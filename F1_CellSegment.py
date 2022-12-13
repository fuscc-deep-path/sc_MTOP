import os
import torch

def fun1(input_dir, output_dir):
    print(input_dir)
    print(output_dir)
    model_path = 'Hover/hovernet_fast_pannuke_type_tf2pytorch.tar'
    args = {'gpu':'0', 'nr_types':6, 'type_info_path':'Hover/type_info.json', 'model_mode':'fast',
            'model_path':model_path, 'nr_inference_workers':8, 'nr_post_proc_workers':0,
            'batch_size':16}
    sub_args = {'input_dir': input_dir,
                'output_dir': output_dir,
                'presplit_dir': None,
                'cache_path':'cache',
                'input_mask_dir':'',
                'proc_mag':40,
                'ambiguous_size':128,
                'chunk_shape':10000,
                'tile_shape':2048,
                'save_thumb':True,
                'save_mask':True}
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
