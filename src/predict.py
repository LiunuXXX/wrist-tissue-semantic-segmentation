import argparse
import logging
import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model.unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicMedicalDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicMedicalDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

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

    return full_mask > out_threshold

def get_output_filenames(
    in_files:list,
    output_dir:str 
    ):
    print(in_files)
    out_files = []
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
                for example the output masked image of input image './data/0/T1/0.jpg' would be './eval/data-0-T1-0.jpg'
                '''
                filename = str(in_file).split('./')[1].replace("/", "-")
                out_files.append(os.path.join(output_dir, filename))
    print(f'output files: {out_files}')
    return out_files


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument('--input_images', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
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
    net.to(device=device)
    net.load_state_dict(
        torch.load(cfg_dict.get('model_weights', None),map_location=device)
    )
    
    logging.info("Model loaded !")
    out_files = get_output_filenames(
        in_files = configargs.input_images,
        output_dir = cfg_dict.get('output_dir',None)
    )
    # start evaluating
    for i, filename in enumerate(configargs.input_images):
        logging.info("\nPredicting image {} ...".format(filename))

        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=cfg_dict.get('scale', 1),
                           out_threshold=cfg_dict.get('mask_threshold', 0.5),
                           device=device)

        if not cfg_dict.get('save', True):
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))
if __name__ == "__main__":
    main()
