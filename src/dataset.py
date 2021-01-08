import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicMedicalDataset(Dataset):
    """Create a iterable pytorch dataset using the input root directory.

        Args:
            root_imgs_dir (str): Input data root dictionary, to create iterable pytorch dataset from.
            imgs_dir_name (str): The directory name of the training image data as a subdirectory of `root_imgs_dir`
            masks_dir_name (str): The directory name of the target mask image data(ground truth) as a subdirectory of root_imgs_dir
            scale (float): Represents the ratio to reduce the image, giving a value greater than 1 will cause an error
            mask_suffix(str): Suffix as mask image in masks_dir_name
    """
    def __init__(self,root_imgs_dir, imgs_dir_name, masks_dir_name, scale=1, mask_suffix=''):
        self.scale = scale
        self.mask_suffix = mask_suffix
        image_ids = [] 
        mask_ids = []
        self.image_and_masks = []
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        for path, dirs, filename in os.walk(root_imgs_dir): # omit files, loop through later
            for dirname in dirs:
                fullpath = os.path.join(path,dirname)
                if imgs_dir_name in dirname:
                    image_ids.append([
                            fullpath +
                            '/' +
                            file for file in listdir(fullpath) if not file.startswith('.')
                    ])
                if masks_dir_name in dirname:
                    mask_ids.append([
                            fullpath +
                            '/' +
                            file for file in listdir(fullpath) if not file.startswith('.')
                    ])
        # flatten list and sort the list to make sure each image file are in descending order
        image_ids = [val for sublist in image_ids for val in sublist]
        image_ids.sort()
        mask_ids = [val for sublist in mask_ids for val in sublist]
        mask_ids.sort()
        assert len(mask_ids)==len(image_ids),\
             f'The number of training image is {len(image_ids)} while the number of masked image is {len(mask_ids)}'

        # Merge images and mask filename into list of tuple.
        self.image_and_masks = [(image_ids[i], mask_ids[i]) for i in range(0, len(image_ids))] 
        logging.info(f'Creating dataset with {len(self.image_and_masks)} examples')
        
    def __len__(self):
        return len(self.image_and_masks)

    @classmethod
    def preprocess(
        cls,
        pil_img,
        scale
    ):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        img_file, mask_file = self.image_and_masks[i]
        #print(f'{img_file},{mask_file})')
        
        mask = Image.open(mask_file)
        img = Image.open(img_file)

        assert img.size == mask.size, \
            f'Image {img_file} and mask {mask_file} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }

def cast_to_image(tensor):
    # Input tensor is (1,H,W). Convert to (1, H, W).
    img = tensor.permute(1, 2, 0)
    return img