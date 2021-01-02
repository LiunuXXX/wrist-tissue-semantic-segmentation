import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from model.unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicMedicalDataset
from torch.utils.data import DataLoader, random_split

def TrainNet(
    Net,
    device,
    epochs = 5,
    batch_size = 1,
    lr = 0.001,
    val_percent = 0.1,
    save_checkpoints = True,
    img_scale = 0.5
):
    dataset = BasicMedicalDataset
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    