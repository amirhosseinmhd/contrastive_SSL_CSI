# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import multiprocessing
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from utils import *

from torch.utils.tensorboard import SummaryWriter

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset

import numpy as np
import torch.optim as optim
import torchvision
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

import scipy.io
parser = argparse.ArgumentParser(description='DirectCLR Training')
parser.add_argument('--data', type=Path, metavar='DIR', default="/data/",
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--device', default='mps',
                    help='which device should it be run')

parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batchsize', default=512, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=4.8, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--dim', default=360, type=int,
                    help="dimension of subvector sent to infoNCE")
parser.add_argument('--mode', type=str, default="simclr",
                    choices=["baseline", "simclr", "directclr", "single", "group"],
                    help="project type")
parser.add_argument('--name', type=str, default='test')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class CustomTimeSeriesDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        """
        Args:
            features (Tensor): Tensor containing the input features.
            labels (Tensor): Tensor containing the labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            sample_view1, sample_view2 = self.transform(sample)

        return (sample_view1, sample_view2), label



def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    mat = scipy.io.loadmat('dataset_home_276.mat')
    keys = ['csid_lab', 'csiu_lab', 'label_lab']
    csid_home = mat['csid_home']
    csiu_home = mat['csiu_home']


    # mat_u = scipy.io.loadmat('dataset_home_276_ul.mat')
    # mat_d = scipy.io.loadmat('dataset_lab_276_dl.mat')

    # keys = ['csid_lab', 'csiu_lab', 'label_lab']
    # csid_home = mat['csid_home']
    # csiu_home = mat['csiu_home']
    # csul = mat_u['csiu_lab']
    # csdl = mat_d['csid_lab']
    # label_lab_u = mat_u['label_lab']
    # label_lab_d = mat_d['label_lab']

    # label_lab = np.concatenate((label_lab_d, label_lab_u), axis=0)
    label_home = mat['label_home']
    # csi_data = np.concatenate((csdl, csul), axis=3)
    csi_data = np.abs(csid_home)
    csi_data = np.transpose(csi_data, (3, 2, 0, 1))
    X_tensor = torch.tensor(csi_data, dtype=torch.float32)
    Y_tensor = torch.tensor(label_home, dtype=torch.long).squeeze() - 1
    print(X_tensor.shape)
    total_samples = X_tensor.size(0)

    train_size = int(0.92 * total_samples)  # 80% for training

    valid_size = total_samples - train_size  # 20% for validation
    transform = TimeSeriesTransform()

    train_dataset, valid_dataset = random_split(CustomTimeSeriesDataset(X_tensor, Y_tensor, transform=transform),
                                                [train_size, valid_size])
    trainloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
    valloader = DataLoader(valid_dataset, batch_size=1024, shuffle=False, num_workers=4)
    model = directCLR(args).to(device)

    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    writer = SummaryWriter(f'runs/csi/Model_{args.mode}_epochs{args.epochs}')

    step = 0
    print(f"Training Model for with following arguments: Model_{args.mode}dim{args.dim}_epochs{args.epochs}")
    for epoch in range(args.epochs):
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pth')
        model.train()
        # progress_bar = tqdm(enumerate(trainloader), total=len(trainloader))
        for i, ((y1, y2), label) in enumerate(trainloader):
            y1 = y1.to(device)
            y2 = y2.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            loss, acc = model.forward(y1, y2, label)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Training Loss', loss, global_step=step)
            writer.add_scalar('Accuracy', acc, global_step=step)
            step += 1
            if i % 30 == 0:
                print(f'[{epoch + 1}/{args.epochs}] Loss: {loss.item():.3f}, Acc: {acc.item()*100:.2f}%')

        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for (y1, y2), label in valloader:
                y1 = y1.to(device)
                y2 = y2.to(device)
                label = label.to(device)
                loss, acc = model.forward(y1, y2, label)
                val_loss += loss.item()
                val_acc += acc
            val_loss = val_loss/len(valloader)
            val_acc = val_acc/len(valloader)
            writer.add_scalar('Validation Training Loss', loss, global_step=epoch)
            writer.add_scalar('Validation Accuracy', acc, global_step=epoch)
            print(f'Epoch: {epoch}, Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_acc:.3f}')
            scheduler.step(val_loss)

    print('Finished Training')


class directCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = MyCNN()
        num_classes = 276
        self.online_head = nn.Linear(1024, num_classes)

        if self.args.mode == "simclr":
            sizes = [1024, 512, 276]
            layers = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[-1]))
            self.projector = nn.Sequential(*layers)
        elif self.args.mode == "single":
            self.projector = nn.Linear(1024, 512, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, y1, y2, labels):
        # print(torch.max(labels))
        r1 = self.backbone(y1)
        r2 = self.backbone(y2)

        if self.args.mode == "baseline":
            z1 = r1
            z2 = r2
            loss = infoNCE(z1, z2) / 2 + infoNCE(z2, z1) / 2
        elif self.args.mode == "directclr":
            z1 = r1[:, :self.args.dim]
            z2 = r2[:, :self.args.dim]
            loss = infoNCE(z1, z2) / 2 + infoNCE(z2, z1) / 2

        elif self.args.mode == "simclr" or self.args.mode == "single":
            z1 = self.projector(r1)
            z2 = self.projector(r2)
            loss = infoNCE(z1, z2) / 2 + infoNCE(z2, z1) / 2

        logits = self.online_head(r1.detach())

        # print(logits.shape, labels.shape)
        cls_loss = torch.nn.functional.cross_entropy(logits, labels)
        acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)

        loss = loss + cls_loss

        return loss, acc


def infoNCE(nn, p, temperature=0.1):
    # device = torch.device("cpu")  # Set device to MPS

    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)
    nn = gather_from_all(nn)
    p = gather_from_all(p)
    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).to(device=device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss
class MyCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(MyCNN, self).__init__()
        # 1D convolution along the first dimension (time/sequence)
        self.conv1d_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv1d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.25)
        self.maxpool1d_1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv2d_1 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 30), stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv1d_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 1))
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout2d(0.3)
        self.conv1d_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 1))
        self.bn5 = nn.BatchNorm2d(512)
        self.dropout3 = nn.Dropout2d(0.3)

        self.conv1d_5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(7, 1), stride=3)
        self.bn6 = nn.BatchNorm2d(1024)
        self.dropout4 = nn.Dropout2d(0.4)

        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(1024, 512)  # Ensure to adjust based on the flattened size
        # self.fc2 = nn.Linear(512, 276)  # Assuming 276 classes

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1d_1(x)))
        x = self.dropout1(F.relu(self.bn2(self.conv1d_2(x))))
        x = self.maxpool1d_1(x)
        x = F.relu(self.bn3(self.conv2d_1(x)))
        x = self.maxpool1d_1(x)
        x = self.dropout2(F.relu(self.bn4(self.conv1d_3(x))))
        x = self.maxpool1d_1(x)
        x = self.dropout3(F.relu(self.bn5(self.conv1d_4(x))))
        x = self.maxpool1d_1(x)
        x = self.dropout4(F.relu(self.bn6(self.conv1d_5(x))))
        x = self.flatten(x)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        return x
if __name__ == '__main__':
    # multiprocessing.freeze_support()  # Required for Windows support
    main()  # Call your main function

