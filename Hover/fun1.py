import os
import torch
# method_args = 
'''
    --input_dir=<path>      Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>     Path to output directory.
    --presplit_dir=<path>   Path to presplit data directory.
    --cache_path=<path>     Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
    --mask_dir=<path>       Path to directory containing tissue masks. 
                            Should have the same name as corresponding WSIs. [default: '']

    --proc_mag=<n>          Magnification level (objective power) used for WSI processing. [default: 40]
    --ambiguous_size=<int>  Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
    --chunk_shape=<n>       Shape of chunk for processing. [default: 10000]
    --tile_shape=<n>        Shape of tiles for processing. [default: 2048]
    --save_thumb            To save thumb. [default: False]
    --save_mask             To save mask. [default: False]
'''
'''python run_infer.py \
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
'''
model_path = './hovernet_fast_pannuke_type_ft2pytorch.tar'
args = {'gpu':0, 'nr_types':6, 'type_info_path':'type_info.json', 'model_mode':'fast',
        'model_path':model_path, 'nr_inference_workers':8, 'nr_post_proc_workers':0,
        'batch_size':16}
sub_args = {}
gpu_list = args.pop('gpu')
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

nr_gpus = torch.cuda.device_count()
nr_types = int(args['nr_types']) if int(args['nr_types']) > 0 else None
method_args = {
    'method' : {
        'model_args' : {
            'nr_types'   : nr_types,
            'mode'       : args['model_mode'],
        },
        'model_path' : args['model_path'],
    },
    'type_info_path'  : None if args['type_info_path'] == '' \
                        else args['type_info_path'],
}

run_args = {
    'batch_size' : int(args['batch_size']) * nr_gpus,

    'nr_inference_workers' : int(args['nr_inference_workers']),
    'nr_post_proc_workers' : int(args['nr_post_proc_workers']),
}

if args['model_mode'] == 'fast':
    run_args['patch_input_shape'] = 256
    run_args['patch_output_shape'] = 164
else:
    run_args['patch_input_shape'] = 270
    run_args['patch_output_shape'] = 80

run_args.update({
    'input_dir'      : sub_args['input_dir'],
    'output_dir'     : sub_args['output_dir'],
    'presplit_dir'   : sub_args['presplit_dir'],
    'input_mask_dir' : sub_args['input_mask_dir'],
    'cache_path'     : sub_args['cache_path'],

    'proc_mag'       : int(sub_args['proc_mag']),
    'ambiguous_size' : int(sub_args['ambiguous_size']),
    'chunk_shape'    : int(sub_args['chunk_shape']),
    'tile_shape'     : int(sub_args['tile_shape']),
    'save_thumb'     : sub_args['save_thumb'],
    'save_mask'      : sub_args['save_mask'],
})

from Hover.infer.wsi import InferManager
infer = InferManager(**method_args)
infer.process_wsi_list(run_args)
