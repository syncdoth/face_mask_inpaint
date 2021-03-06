bs=1
run_name=drn_best_ref_HQ_better_att

python train_reference_fill.py \
--data_root /data/mohaa/project1/CelebAHQ \
--src_img_path images_masked \
--ref_img_path images \
--mask_path binary_map \
--identity_file_path CelebA-HQ-identity.txt \
--batch_size $bs \
--img_scale 0.25 \
--run_name $run_name \
--encoder_type drn \
--eval_options ssim ms_ssim \
--use_best_reference 1 \
--pt_ckpt_path pretrained_models \
--decoder_img_f 256