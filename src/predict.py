import argparse
import logging
import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils
from model.unet import UNet
from data_vis import plot_img_and_mask
from dataset import BasicMedicalDataset
from dice_loss import (dice_coeff, DiceCoeff)
from cv2 import cv2

def contours_fn(mask, contours_color : tuple = (255,255,0)):
    mask = (mask * 255).astype(np.uint8)
    mask = mask[..., np.newaxis]
    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        cv2.drawContours(drawing, contours, i, contours_color, 2, cv2.LINE_8, hierarchy, 0)
    return Image.fromarray(drawing)

def threshold_fn(mask):
    mask = np.stack((mask,)*3, axis=-1) # GRAY -> RGB
    mask = (mask * 255).astype(np.uint8)
    mask[...,0] = mask[...,1] = 0
    return Image.fromarray(mask)

def mask_to_image(mask, fn = None, color:tuple = (255,255,0)):
    if fn is None:
        return Image.fromarray((mask * 255).astype(np.uint8))
    else:
        return Image.fromarray((mask * 255).astype(np.uint8)), fn(mask, color)
def merge_pil_image(
    img1_filename:str,
    img2_filename:str,
    weight:float = 0.5,
    color_space = 'RGB' 
):
    img1 = Image.open(img1_filename)
    img2 = Image.open(img2_filename)
    # Make sure two image have the same size
    assert img1.size == img2.size, f'Two images should have the same size, image1 have size {img1.size} while image2 have size {img2.size}'
    blended = Image.blend(img1, img2, weight)
    if color_space == 'RGB':
        blended = np.stack((blended,)*3, axis=-1) # GRAY -> RGB
        blended = (blended * 255).astype(np.uint8)
    return Image.fromarray(blended)

def predict_img(net,
                full_img,
                target_img,
                device,
                scale_factor=1,
                out_threshold=0.5
):
    net.eval()

    img = torch.from_numpy(BasicMedicalDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    target_mask = torch.from_numpy(BasicMedicalDataset.preprocess(target_img, scale_factor))
    target_mask = target_mask.unsqueeze(0)
    target_mask = target_mask.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img) # output mask

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
        pred = torch.sigmoid(output)
        pred = (pred > 0.5).float()
        dc_val = DiceCoeff().forward(pred, target_mask).item()
    return full_mask > out_threshold, dc_val

def get_output_filenames(
    in_files:list,
    output_dir:str,
    suffix:str
    ):
    out_files = []
    print(output_dir)
    if not output_dir:
        logging.error("The folder to which the file location of output is not declared")
        raise SystemExit()
    elif not os.path.isdir(output_dir):
        logging.error("The folder to which the file location of output is not exist")
        raise SystemExit()
    else:
        for in_file in in_files:
            if not in_file.endswith(('jpg', 'jpeg', 'png')):
                logging.error(f"file {in_file} is not a image file")
                raise SystemExit()
            else:
                '''
                In order to be able to easily distinguish the original absolute path of the predicted mask image, 
                we replace all the'/' symbols in the path of the original input image with '-' 
                and prefix the output folder as the output mask image Absolute path
                for example the output masked image of input image 'wrist/data/0/T1/0.jpg' would be 'wrist/eval/data-0-T1-0.jpg'
                '''
                filename = str(in_file).replace("/", "-")
                if filename.endswith(('jpg')):
                    filename = filename.replace(".jpg", suffix +".jpg")
                elif filename.endswith(('png')):
                    filename = filename.replace(".png", suffix +".png")
                else:
                    filename = filename.replace(".jpeg", suffix +".jpeg")
                
                out_files.append(os.path.join(output_dir, filename))
    print(f'output files: {out_files}')
    return out_files


def predict(
    input_images:list = None,
    target_images:list = None,
    config_file:str = None,
    save_file_suffix = None
):
    '''
    Compute output masked and its contours graphs given the "list" of input images filenames.
    Args:
       input_images (list[str]): list of input images filenames, if None then input filenames are given by argument list instead
       target_images (list[str]): list of target images mask filenames, if None then target filenames are given by argument list instead
       config_file  (list[str]): path to the configuation file that specify evaluation detail, 
            if None then config file path are given by argument list instead
    Returns:
        out_files (list[str]): list of output maksed filenames
        countors_outs_files (list[str]): list of countours of output maksed filenames
        dc_val_records (list[float]): list of dice coefficient of each target masked image and output(predicted) masked image
    '''
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # specify configuration file
    if config_file is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config", type=str, help="Path to (.yml) config file."
        )
        parser.add_argument('--input_images', '-i', metavar='INPUT', nargs='+',
                            help='filenames of input images')

        parser.add_argument('--target_images', '-t', metavar='INPUT', nargs='+',
                            help='filenames of target mask images')
        configargs = parser.parse_args()
        config_file = configargs.config

    # Read config file.
    cfg = None
    with open(config_file, "r") as f:
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
    net.to(device=device)
    net.load_state_dict(
        torch.load(cfg_dict.get('model_weights', None),map_location=device)
    )
    # In the case of ignoring parameters, input filenames are given by argument list instead
    if input_images is None:
        input_images = configargs.input_images
    if target_images is None:
        target_images = configargs.target_images
    logging.info("Model loaded !")
    out_files = get_output_filenames(
        in_files = input_images,
        output_dir = cfg_dict.get('output_dir',None),
        suffix = save_file_suffix
    )
    countors_outs_files = []
    dc_val_records = []
    # start evaluating
    for i, (filename, target_filename) in enumerate(zip(input_images, target_images)):
        logging.info(f"\nPredicting image {filename}, Target image {target_filename}")

        img = Image.open(filename)
        target = Image.open(target_filename)

        mask, dc_val = predict_img(net=net,
                           full_img=img,
                           target_img=target,
                           scale_factor=cfg_dict.get('scale', 1),
                           out_threshold=cfg_dict.get('mask_threshold', 0.5),
                           device=device
        )
        if  cfg_dict.get('save', True):
            out_filename = out_files[i]
            result, contours = mask_to_image(mask,fn = contours_fn)
            result.save(out_files[i])
            out_contour = out_files[i].replace(".jpg", "-contour.jpg")
            contours.save(out_contour)
            countors_outs_files.append(out_contour)
            # Record DC value for evaluation
            dc_val_records.append(dc_val)
            logging.info(f"\nMask saved to {out_files[i]}, Countour saved to {out_contour}")
    return out_files, countors_outs_files, dc_val_records
if __name__ == "__main__":
    predict()
