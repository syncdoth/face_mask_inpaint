import logging
import math
import random
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


def get_reference_dataloader(dir_src_img,
                             dir_ref_img,
                             dir_mask,
                             identity_file,
                             batch_size,
                             val_amount=0.1,
                             num_workers=4,
                             img_scale=1.0):
    dataset = ReferenceDataset(dir_src_img,
                               dir_ref_img,
                               dir_mask,
                               identity_file,
                               scale=img_scale)
    n_train = math.floor(len(dataset) * (1 - val_amount))
    n_val = math.ceil(len(dataset) * val_amount)

    train_set, val_set = random_split(dataset, [n_train, n_val])

    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    return train_loader, val_loader


class BasicDataset(Dataset):

    def __init__(self, images_dir, masks_dir, scale=1.0, mask_suffix=''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        print("Images_dir length: ", len(listdir(images_dir)))
        print("Masks_dir length: ", len(listdir(masks_dir)))

        self.ids = [
            splitext(file)[0].split('_')[0]
            for file in listdir(images_dir)
            if not file.startswith('.')
        ]
        self.ids = random.sample(self.ids, 90000)
        if not self.ids:
            raise RuntimeError(
                f'No input file found in {images_dir}, make sure you put your images there'
            )
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH),
                                 resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = self.masks_dir / Path(name + self.mask_suffix + '.npy')
        img_file = self.images_dir / Path(name + '_surgical' + '.jpg')

        mask = self.load(mask_file)
        img = self.load(img_file)

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class ReferenceDataset(BasicDataset):

    def __init__(self, source_dir, reference_dir, masks_dir, identity_file, scale=1.0):
        self.source_dir = Path(source_dir)
        self.masks_dir = Path(masks_dir)
        self.reference_dir = Path(reference_dir)
        self.identity_map, self.img2identity = self.read_identity_file(identity_file)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.ids = [
            splitext(file)[0].split('_')[0]
            for file in listdir(source_dir)
            if not file.startswith('.')
        ]
        if not self.ids:
            raise RuntimeError(
                f'No input file found in {source_dir}, make sure you put your images there'
            )
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def read_identity_file(self, identity_file):
        identity_map = {}
        img2identity = {}
        with open(identity_file, 'r') as f:
            for line in f:
                img, identity = line.strip().split(' ')
                img_id = splitext(img)[0].split('_')[0]
                identity = int(identity)

                # cache to dictionary
                img2identity[img_id] = identity
                if identity in identity_map:
                    identity_map[identity].append(img_id)
                else:
                    identity_map[identity] = [img_id]
        return identity_map, img2identity

    def sample_reference_image(self, img_name):
        images = self.identity_map[self.img2identity[img_name]]
        reference_image = random.choice(images)
        while reference_image == img_name:
            reference_image = random.choice(images)
        return reference_image

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = self.masks_dir / Path(name + '.npy')
        src_file = self.images_dir / Path(name + '_surgical' + '.jpg')
        gt_file = self.reference_dir / Path(name + '.jpg')
        ref_file = self.sample_reference_image(name)

        mask = self.load(mask_file)
        src_img = self.load(src_file)
        gt_img = self.load(gt_file)
        ref_img = self.load(ref_file)

        assert src_img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        src_img = self.preprocess(src_img, self.scale, is_mask=False)
        gt_img = self.preprocess(gt_img, self.scale, is_mask=False)
        ref_img = self.preprocess(ref_img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'src_img': torch.as_tensor(src_img.copy()).float().contiguous(),
            'gt_img': torch.as_tensor(gt_img.copy()).float().contiguous(),
            'ref_img': torch.as_tensor(ref_img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }