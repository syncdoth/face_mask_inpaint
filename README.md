# face_mask_inpaint

This is a repository for "Reference Guided Facial Mask Removal" by Sehyun Choi
and Minseok Oh, for the final project of HKUST COMP 4471, 2021 Fall.

## Environment

We recommend using conda environment. First, create a conda environment using

```
conda create -n $env_name python=$py_ver
conda activate $env_name
```

Then, we have prepared a script for setting up the conda env at [env_setup](env_setup.sh).

## Experiments

First, download the CelebA and CelebAHQ dataset from the official project
[page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

* *NOTE: you have contact the authors to get the identity file for the CelebAHQ dataset.*

We expect the dataset folder to be structured as:

```
CelebAHQ
├──images  # all source images
├──images_masked  # masked with MaskTheFace
├──images_masked_test  # pre selected test set
├──binary_map  # mask map .npy files
├──identity.txt

```

* `images_masked` and `binary_map` should be obtained using
[MaskTheFace](https://github.com/aqeelanwar/MaskTheFace) utility.

After you have downloaded the dataset, you need to download the pretrained models
of [PICNet](https://github.com/lyndonzheng/Pluralistic-Inpainting) and
[pSp](https://github.com/eladrich/pixel2style2pixel). Follow the instructions at
their original repos to download the pretrained weights.
*(for pSp, we used StyleGAN inversion checkpoint.)*

Then, you could look at the `scripts` for the various training configurations.

## Evaluation

For evaluation, we use SSIM, MS-SSIM, and FID. To obtain SSIM and MS-SSIM, run
[psp_inference.py](psp_inference.py) or [PICNet_inference.py](psp_inference.py)
depending on the model you want to test. An example for each is:

```
python psp_inference.py --use_ref --use_attention 1 \
--pt_ckpt_path saved_model/RefpSp_train_decoder_attention/G_checkpoint_epoch5.pth \
--batch_size 8 --data_root /path/to/CelebAHQ

python PICNet_inference.py \
--data_root /path/to/CelebAHQ  \
--src_img_path images_masked_test \
--ref_img_path images \
--mask_path binary_map \
--identity_file_path CelebA-HQ-identity.txt \
--mask_detector_path saved_model/new_mask_detector.pth \
--pt_ckpt_path saved_model/PICNet_best_ref_HQ_better_att/G_checkpoint_epoch4.pth \
--img_scale 0.25 \
--use_att 1 \
--batch_size 4 \
--decoder_img_f 256 --decoder_z_nc 256
```

These two scripts generates images from the test set, saves the results in
`test_results/[checkpoint_name]` folder, and calculates SSIM and MS-SSIM. To
calculate FID,

```
python -m pytorch_fid test_results/[checkpoint_name] path/to/test/images
```

## Acknowledgements

This repo borrows heavily from other implementations. Namely:

* [Pixel2Style2Pixel](https://github.com/eladrich/pixel2style2pixel/tree/master)
* [PICNet](https://github.com/lyndonzheng/Pluralistic-Inpainting)
* [PyTorch-UNet](https://github.com/milesial/Pytorch-UNet)
* [DRN](https://github.com/fyu/drn)
* [VGG related loss](https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49)

## License
<br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

This software is for educational and academic research purpose only.