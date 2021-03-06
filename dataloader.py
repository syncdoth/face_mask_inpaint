import logging
import math
import pickle
import random
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

from pytorch_msssim import SSIM


def get_reference_dataloader(dir_src_img,
                             dir_ref_img,
                             dir_mask,
                             identity_file,
                             batch_size,
                             apply_transform=False,
                             val_amount=0.1,
                             num_workers=4,
                             img_scale=1.0,
                             use_ssim=False,
                             device=None):
    dataset = ReferenceDataset(dir_src_img,
                               dir_ref_img,
                               dir_mask,
                               identity_file,
                               apply_transform=apply_transform,
                               scale=img_scale,
                               use_ssim=use_ssim,
                               device=device)
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
        # self.ids = random.sample(self.ids, 90000)
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
        if not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
            img_ndarray = img_ndarray / 255
            img_ndarray = torch.as_tensor(img_ndarray.copy()).float().contiguous()
        else:
            img_ndarray = torch.as_tensor(img_ndarray.copy()).long().contiguous()

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

        return {'image': img, 'mask': mask}


class ReferenceDataset(BasicDataset):

    def __init__(self,
                 source_dir,
                 reference_dir,
                 masks_dir,
                 identity_file,
                 apply_transform=True,
                 scale=1.0,
                 use_ssim=False,
                 device=None,
                 return_id=False):
        self.source_dir = Path(source_dir)
        self.masks_dir = Path(masks_dir)
        self.reference_dir = Path(reference_dir)
        self.identity_map, self.img2identity = self.read_identity_file(identity_file)
        self.filter_id = set()  # identities with only one image should be ignored
        for v in self.identity_map.values():
            if len(v) < 2:
                self.filter_id.update(v)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.ids = []
        for f in listdir(source_dir):
            f_id = splitext(f)[0].split('_')[0]
            if not f.startswith('.') and f_id not in self.filter_id:
                self.ids.append(f_id)

        if not self.ids:
            raise RuntimeError(
                f'No input file found in {source_dir}, make sure you put your images there'
            )
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.use_ssim = use_ssim
        if use_ssim:
            # best_reference_map
            my_file = self.source_dir.parent / Path('best_reference_map.pkl')
            if my_file.is_file():
                with open(my_file, 'rb') as f:
                    self.best_reference_map = pickle.load(f)
            else:
                logging.info(f'Creating best_reference_map')
                self.ssim = SSIM(data_range=1, size_average=True, channel=3)
                self.best_reference_map = self.find_best_reference(device)

        self.apply_transform = apply_transform
        if apply_transform:
            self.transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.return_id = return_id

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

    def find_best_reference(self, device):
        best_reference_map = {}
        device = 'cpu' if device is None else device
        with tqdm(total=len(self.ids), unit='img') as pbar:
            for name in self.ids:
                pbar.update(1)
                gt_file = self.reference_dir / Path(name + '.jpg')
                gt_img = self.load(gt_file)
                gt_img = self.preprocess(gt_img, self.scale, is_mask=False)
                gt_img_tensor = torch.as_tensor(
                    gt_img.copy()).float().contiguous().unsqueeze(0).to(device)
                max_score = -10
                best_ref = None
                for ref_image_name in self.identity_map[self.img2identity[name]]:
                    if (ref_image_name != name):
                        ref_file = self.reference_dir / Path(ref_image_name + '.jpg')
                        ref_img = self.load(ref_file)
                        ref_img = self.preprocess(ref_img, self.scale, is_mask=False)
                        ref_img_tensor = torch.as_tensor(
                            ref_img.copy()).float().contiguous().unsqueeze(0).to(device)
                        score = self.ssim(gt_img_tensor, ref_img_tensor)
                        if (score > max_score):
                            max_score = score
                            best_ref = ref_image_name
                best_reference_map[name] = best_ref
        with open(self.source_dir.parent / Path('best_reference_map.pkl'), 'wb') as f:
            pickle.dump(best_reference_map, f)
        return best_reference_map

    def sample_reference_image(self, img_name):
        if self.use_ssim:
            return self.best_reference_map[img_name]

        images = self.identity_map[self.img2identity[img_name]]
        assert len(images) > 1
        reference_image = random.choice(images)
        while reference_image == img_name:
            reference_image = random.choice(images)
        return reference_image

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = self.masks_dir / Path(name + '.npy')
        src_file = self.source_dir / Path(name + '_surgical' + '.jpg')
        gt_file = self.reference_dir / Path(name + '.jpg')
        ref_file = self.reference_dir / Path(self.sample_reference_image(name) + '.jpg')

        mask = self.load(mask_file)
        src_img = self.load(src_file)
        gt_img = self.load(gt_file)
        ref_img = self.load(ref_file)

        assert src_img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        src_img = self.preprocess(src_img, self.scale, is_mask=False)
        raw_gt_img = self.preprocess(gt_img, self.scale, is_mask=False)
        ref_img = self.preprocess(ref_img, self.scale, is_mask=False)
        if self.apply_transform:
            src_img = self.transform(src_img)
            ref_img = self.transform(ref_img)
            gt_img = self.transform(raw_gt_img)
        else:
            gt_img = raw_gt_img
        mask = self.preprocess(mask, self.scale, is_mask=True)

        items = {
            'src_img': src_img,
            'gt_img': gt_img,
            'raw_gt_img': raw_gt_img,
            'ref_img': ref_img,
            'mask': mask
        }
        if self.return_id:
            items['id'] = torch.LongTensor([int(self.ids[idx])])
        return items
