# frequent hyperparameters
run_name=pSp_adam_no_crop_from_inversion
optimizer=adam  # or ranger
lr=1e-4
bs=2


python train_psp.py \
--batch_size $bs --learning_rate $lr \
--lpips_lambda=0.8 --l2_lambda=1 --id_lambda=0.1 --w_norm_lambda=0.005 \
--lpips_lambda_crop=0 --l2_lambda_crop=0 \
--run_name $run_name \
--optimizer $optimizer \
--img_scale 0.25 --start_from_latent_avg --randomize_noise \
--data_root /data/mohaa/project1/CelebAHQ \
--src_img_path images_masked --ref_img_path images --mask_path binary_map \
--identity_file_path CelebA-HQ-identity.txt \
--pt_ckpt_path pretrained_models/psp_ffhq_encode.pt
# --use_ref  # uncomment this line and add \ to the line above to use reference images