bs=1
run_name=PICNet_best_ref_HQ

# --encoder_type drn
python train_reference_fill.py \
--data_root /data/mohaa/project1/CelebAHQ \
--src_img_path images_masked \
--ref_img_path images \
--mask_path binary_map \
--identity_file_path CelebA-HQ-identity.txt \
--batch_size $bs \
--img_scale 0.25 \
--run_name $run_name \
--eval_options ssim ms_ssim \
--use_best_reference 1 \
--pt_ckpt_path pretrained_models