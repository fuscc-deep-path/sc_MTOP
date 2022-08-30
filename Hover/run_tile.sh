python run_infer.py \
--gpu='1' \
--nr_types=6 \
--type_info_path=type_info.json \
--batch_size=32 \
--model_mode=fast \
--model_path=/media/tiger/Disk1/jhan/code/hover_net-master/hovernet_fast_pannuke_type_tf2pytorch.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=/media/tiger/Disk1/jhan/datasets/sliangProstate/Images/ \
--output_dir=/media/tiger/Disk1/jhan/datasets/sliangProstate/Preds/ \
--mem_usage=0.1 \
--draw_dot \
--save_qupath
