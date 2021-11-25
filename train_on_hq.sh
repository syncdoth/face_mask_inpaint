bs=2
run_name=test_hq


python train_reference_fill.py --data_root /data/mohaa/project1/CelebAHQ \
--src_img_path images_masked \
--ref_img_path images \
--mask_path binary_map \
--identity_file_path CelebA-HQ-identity.txt \
--batch_size $bs \
--img_scale 0.5 \
--run_name $run_name