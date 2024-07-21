source ~/anaconda3/etc/profile.d/conda.sh

conda activate DL


python main.py --mode directclr --dim 360 --epochs 140

python main.py --mode simclr --dim 360 --epochs 140

python main.py --mode base --dim 360 --epoch 140

# tensorboard --logdir ./runs
