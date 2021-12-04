# frequent hyperparameters
run_name=RefpSp_from_inversion_style_cx_decoder_l22
optimizer=adam  # or [adam, ranger]
lr=1e-4
bs=2

#--w_norm_lambda=0.005 \  use this when freezing the stylegan decoder

python train_psp.py \
--train_decoder 1 \
--eval_options ssim ms_ssim \
--batch_size $bs --learning_rate $lr \
--lpips_lambda=0.8 --l2_lambda=2 --id_lambda=0.1 --style_lambda=1000 \
--cx_lambda=1 \
--w_norm_lambda=0 \
--run_name $run_name \
--optimizer $optimizer \
--img_scale 0.25 --start_from_latent_avg --randomize_noise \
--data_root /data/mohaa/project1/CelebAHQ \
--src_img_path images_masked --ref_img_path images --mask_path binary_map \
--identity_file_path CelebA-HQ-identity.txt \
--pt_ckpt_path pretrained_models/psp_ffhq_encode.pt \
--use_ref  # uncomment this line and add '\' to the line above to use reference images
# --use_attention  # uncomment this line and add '\' to the line above