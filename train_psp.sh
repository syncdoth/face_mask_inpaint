# frequent hyperparameters
run_name=RefpSp_adam_from_inversion_style_ref_loss
optimizer=adam  # or [adam, ranger]
lr=1e-4
bs=2


python train_psp.py \
--batch_size $bs --learning_rate $lr \
--lpips_lambda=0.8 --l2_lambda=1 --id_lambda=0.1 --style_lambda=1000 \
--lpips_lambda_ref=0.8 --l2_lambda_ref=0.8 \
--w_norm_lambda=0.005 \
--run_name $run_name \
--optimizer $optimizer \
--img_scale 0.25 --start_from_latent_avg --randomize_noise \
--data_root /data/mohaa/project1/CelebAHQ \
--src_img_path images_masked --ref_img_path images --mask_path binary_map \
--identity_file_path CelebA-HQ-identity.txt \
--pt_ckpt_path pretrained_models/psp_ffhq_encode.pt \
--use_ref  # uncomment this line and add '\' to the line above to use reference images