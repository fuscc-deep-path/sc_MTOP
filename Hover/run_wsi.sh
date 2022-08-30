python run_infer.py \
--gpu='0' \
--nr_types=6 \
--type_info_path=type_info.json \
--batch_size=32 \
--model_mode=fast \
--model_path=/home/xujun/FUSCC/Hover/hovernet_fast_pannuke_type_tf2pytorch.tar \
--nr_inference_workers=4 \
--nr_post_proc_workers=0 \
wsi \
--input_dir=/home/xujun/FUSCC/WSI_example/WSI2 \
--output_dir=/home/xujun/FUSCC/WSI_example/pred2 \
--presplit_dir=/home/xujun/FUSCC/WSI_example/WSI_presplit2 \
--proc_mag 20 \
--save_thumb \
--save_mask
