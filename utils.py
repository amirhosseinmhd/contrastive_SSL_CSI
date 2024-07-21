# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import scipy.io

import random
import math
import torch
import torch.distributed as dist
from classy_vision.generic.distributed_util import (
    convert_to_distributed_tensor,
    convert_to_normal_tensor,
    is_distributed_training_run,
)
from torch import optim
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
from scipy.interpolate import CubicSpline
from scipy.signal import resample, butter, filtfilt
import matplotlib.pyplot as plt


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """
    Similar to classy_vision.generic.distributed_util.gather_from_all
    except that it does not cut the gradients
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = GatherLayer.apply(tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class TimeSeriesTransform:
    def __init__(self, transformations=None):
        # Define your transforms here
        self.transforms = [
            self.add_jitter,  # less noise
            self.permute_segments,  # more segments
            # self.time_warp,  # same time warp
            self.scale_data#,  # less scaling variation
            # self.channel_shuffle  # same shuffling
        ]

        self.transforms_prime = [
            self.add_jitter,  # less noise
            self.permute_segments,  # more segments
            # self.time_warp,  # same time warp
            self.scale_data#,  # less scaling variation
            # self.channel_shuffle  # same shuffling
        ]

    # self.transforms = [
        #     lambda x: self.add_jitter(x, noise_level=0.8),  # less noise
        #     lambda x: self.permute_segments(x, num_segments=5),  # more segments
        #     self.time_warp,  # same time warp
        #     lambda x: self.scale_data(x, scale_variation=0.05),  # less scaling variation
        #     self.channel_shuffle  # same shuffling
        # ]
        #
        # self.transforms_prime = [
        #     lambda x: self.add_jitter(x, noise_level=0.8),  # less noise
        #     lambda x: self.permute_segments(x, num_segments=5),  # more segments
        #     self.time_warp,  # same time warp
        #     lambda x: self.scale_data(x, scale_variation=0.05),  # less scaling variation
        #     self.channel_shuffle  # same shuffling
        # ]

    def __call__(self, x):
        y1 = x.clone()
        y2 = x.clone()

        for transform in self.transforms:
            y1 = transform(y1)

        for transform in self.transforms_prime:
            y2 = transform(y2)

        return y1, y2

    def add_jitter(self, data, noise_level=1):
        noise = np.random.normal(loc=0.0, scale=noise_level, size=data.shape)
        return data + torch.tensor(noise, dtype=torch.float32)

    def permute_segments(self, data, num_segments=3):
        total_length = data.size(2)
        segment_length = total_length // num_segments
        remainder = total_length % num_segments

        permuted_data = data.clone()
        for sample in range(data.size(0)):
            for channel in range(data.size(1)):
                # Initialize an empty list to collect segments
                segments = []

                # Collect each segment, adjusting the last segment to take any remainder
                for i in range(num_segments):
                    start_idx = i * segment_length
                    if i == num_segments - 1:  # Last segment takes the remainder
                        end_idx = start_idx + segment_length + remainder
                    else:
                        end_idx = start_idx + segment_length

                    segments.append(data[sample, channel, start_idx:end_idx])

                # Shuffle segments
                np.random.shuffle(segments)

                # Concatenate shuffled segments and assign to permuted_data
                permuted_data[sample, channel] = torch.cat(segments, dim=0)

        return permuted_data

    def time_warp(self, data, time_steps=None):
        warped_data = data.clone()
        n = data.size(2)
        for sample in range(data.size(0)):
            for channel in range(data.size(1)):
                original_steps = np.arange(n)
                if time_steps is None:
                    time_steps = original_steps
                new_time_steps = np.sort(np.random.normal(loc=original_steps, scale=1, size=time_steps.size))  # Generate new time steps
                spline = CubicSpline(original_steps, data[sample, channel].to(device='cpu').numpy(), axis=0)
                warped_data[sample, channel] = torch.tensor(spline(new_time_steps), dtype=torch.float32)
                warped_data[sample, channel][0] =  data[sample,channel][0].clone()
                warped_data[sample, channel][n-1] =  data[sample,channel][n-1].clone()

        return warped_data

    def scale_data(self, data):
        scaling_factor = np.random.normal(loc=1.0, scale=0.1, size=(data.size(0), data.size(1), data.size(2)))##########
        return data * torch.tensor(scaling_factor, dtype=torch.float32)
    def channel_shuffle(self, data):
        shuffled_data = data.clone()
        for sample in range(data.size(0)):
            channels = torch.randperm(data.size(1))
            shuffled_data[sample] = data[sample, channels]
        return shuffled_data





def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    base_lr = args.learning_rate #* args.batch_size / 256

    warmup_steps = 10 * len(loader)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr