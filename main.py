"""main.py

Usage:
    main.py [options] [--help] <command> [<args>...]
    main.py (-h | --help)

Options:
    -h --help       Show this screen
    
Two command mode are `tile` and `wsi` to enter corresponding inference mode
    segment     cell segment from the whole slide image
    feature     feature extract
    visual      visualization of the graph and segmentation

Use 'main.py <command> --help' to show their options and usage. 
"""
segment_cli = """
Arguments for cell segment using HoVer-Net.

Usage:
    segment --input_dir=<path> --output_dir=<path>

Option:
    --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>    Path to output directory.
"""

feature_cli = """
Arguments for feature extract.

Usage:
    feature --json_path=<filepath> --wsi_path=<filepath> --output_path=<folderpath> [--xml_path=<path>]

Option:
    --json_path=<filepath>     Path to HoVer-Net output, it show be a json file.
    --wsi_path=<filepath>      Path to wsi file.
    --output_path=<folderpath>   Path to output.
    --xml_path=<path>      Path to xml. The xml is an annotation file of ImageScope. Only extract the feature in the annotation.[default: None]
"""

visual_cli = """
Arguments for visual.

Usage:
    visual --feature_path=<filepath> --wsi_path=<filepath> --xml_path=<filepath>

Option:
    --feature_path=<filepath>  Path to feature folder, it show be a folder including feature and edge .csv file.
    --wsi_path=<filepath>      Path to wsi file. 
    --xml_path=<filepath>      Path to xml file. The xml is an annotation file of ImageScope.\
                               Only plot in the scale of annotation.[default: None]
"""

from docopt import docopt

if __name__ == '__main__':
    sub_cli_dict = {'segment':segment_cli,
                    'feature':feature_cli,
                    'visual':visual_cli}
    args = docopt(__doc__, help=False, options_first=True)
    sub_cmd = args.pop('<command>')
    sub_cmd_args = args.pop('<args>')

    if args['--help'] and sub_cmd is not None:
        if sub_cmd in sub_cli_dict: 
            print(sub_cli_dict[sub_cmd])
        else:
            print(__doc__)
        exit()
    if args['--help'] or sub_cmd is None:
        print(__doc__)
        exit()
    
    sub_args = docopt(sub_cli_dict[sub_cmd], argv=sub_cmd_args, help=True)
    sub_args = {k.replace('--', '') : v for k, v in sub_args.items()}
    print(sub_args)
    if sub_cmd=='segment':
        from F1_CellSegment import fun1
        import sys
        sys.path.append('Hover')
        fun1(**sub_args)
    elif sub_cmd=='feature':
        from F3_FeatureExtract import fun3
        if sub_args['xml_path'] == 'None':
            sub_args['xml_path'] = None
        fun3(**sub_args)
    elif sub_cmd=='visual':
        from F4_Visualization import fun4
        fun4(**sub_args)