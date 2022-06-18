#!/bin/bash
# Build documentation for display in web browser.

python train_MLOFO.py --total-epoch 200 --d 'CUB' --a 'resnet50' --scheduler 'warmup' --warmup-epochs 5 --nsample 4 --exp-dir '_log'
