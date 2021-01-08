import argparse
import logging
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from model.unet import UNet
from eval import eval_net
from dice_loss import dice_coeff, DiceLoss
from dataset import BasicMedicalDataset
from torch.utils.data import DataLoader, random_split
def TrainNet(
    Net,
    device,
    root_imgs_dir,
    imgs_dir_name,
    mask_dir_name,
    dir_checkpoint,
    epochs = 5,
    batch_size = 1,
    lr = 0.001,
    val_percent = 0.1,
    save_checkpoints = True,
    img_scale = 1,
    custome_loss_fn = None
):
    print(f"training data {root_imgs_dir}")
    dataset = BasicMedicalDataset(root_imgs_dir, imgs_dir_name, mask_dir_name, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last = True
    )
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoints}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    optimizer = optim.RMSprop(
        Net.parameters(), 
        lr = lr,
        weight_decay = 1e-8,
        momentum = 0.9
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        'min' if Net.n_classes > 1 else 'max', patience=2
    )
    if custome_loss_fn is not None:
        criterion = custome_loss_fn()
    elif Net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        # Sets the module in training mode.
        Net.train()
        epoch_loss = 0
        with tqdm(total = n_train,desc=f'Epoch {epoch + 1}/{epochs}',unit = 'img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_mask = batch['mask']
                assert imgs.shape[1] == Net.n_channels, \
                    f'Network has been defined with {Net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                imgs = imgs.to(device = device, dtype = torch.float32)
                mask_type = torch.float32 if Net.n_channels == 1 else torch.long
                true_mask = true_mask.to(device = device, dtype = mask_type)

                masks_pred = Net(imgs)
                
                loss = criterion(masks_pred, true_mask)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                # To avoid gradient explosion during training
                nn.utils.clip_grad_value_(Net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % (n_train // (10 * batch_size)) == 0:
                    val_score = eval_net(Net, val_loader, device)
                    scheduler.step(val_score)
                    if Net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))

    if save_checkpoints:
        try:
            os.mkdir(dir_checkpoint)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        torch.save(Net.state_dict(),
                       dir_checkpoint + f'{mask_dir_name}_CP_epoch{epoch + 1}.pth')
        logging.info(f'Checkpoint {epoch + 1} saved !')

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()
    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        print(cfg_dict)
    # set up network in/out channels details
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    #print(net)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    net.to(device=device)
    # set up custome_loss_fn
    custome_loss_fn = None
    if cfg_dict.get('loss_function', None) == 'dice_loss':
        logging.info(f"\n Use custom loss function-{cfg_dict.get('loss_function', None)}")
        custome_loss_fn = DiceLoss
    # start training
    TrainNet(
        Net = net,
        device = device,
        root_imgs_dir = cfg_dict.get('base_dir', None),
        imgs_dir_name = cfg_dict.get("image_dir_suffix", None),
        mask_dir_name = cfg_dict.get("mask_dir_suffix", None),
        dir_checkpoint = cfg_dict.get("checkpoint_dir", None),
        epochs = cfg_dict.get("epochs", 5),
        batch_size = cfg_dict.get("batch_size", 1),
        lr = cfg_dict.get("learning_rate", 0.0001),
        val_percent = cfg_dict.get("validation", 0.2),
        save_checkpoints = True,
        img_scale = cfg_dict.get("scale", 1),
        custome_loss_fn = custome_loss_fn
    )
if __name__ == "__main__":
    main()