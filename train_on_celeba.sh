run_name=drn_encoder
bs=8
python train_reference_fill.py --encoder_type drn \
--run_name $run_name --batch_size $bs --eval_options ssim