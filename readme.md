# Single Cell Morphological and Topological Profiling Based on Digital Pathology

## Description

sc-MTOP is an analysis framework based on deep learning and computational pathology. It consists of three steps: 1) Hover-Net based nuclear segmentation and classification; 2) Nuclear morphological and texture feature extraction; 3) Multi-level pairwise nuclear graph construction and spatial topological feature extraction. This framework aims to characterize the tumor ecosystem diversity at the single-cell level. We have a [demo](http://101.132.124.14/#/dashboard) website to show this work.

This is the offical pytorch implementation of sc_MTOP. According to the above description, we use three functions to finish three steps: segment, feature and visual. Note that only segment step support batch processing.

<a id="hovernet"></a>
In the segmentation steps, it uses the [HoVer-Net](https://github.com/vqdang/hover_net) model. We doesn't provide the model parameter in the source code because it is large. We use the pretrain model based on PanNuke dataset, you can download it from the this [url](https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR/view) which provided from the repositories of HoVer-Net.  It needs a folder path including WSI files of all samples as input. We use `.ndpi` file in our work, but we have not tried other formats of wsi files. In theory it supports all file formats allowed by HoVer-Net. The step gives a folder with json file which including all information of cell segmentation and classification.

In the feature step, you have to provide the path of `.json` file, WSI file, and output folder. You can also provide an `.xml` annotation file to only compute features for cells within the annotation range. The annotation is with square shape and white color. The code supports multiple annotation to calculate several area in the same sample. The `.xml` annotation file follow the format of [ImageScope](https://www.leicabiosystems.com/zh/digital-pathology/manage/aperio-imagescope/) software's `.xml` annotation. The annotation file from other software may not be support. In the output folder there will be a secondary folder named after the input WSI file. In this folder, it will include three `.csv` files of cell features and one `.csv` file of graph edge information.

In the visual step, we make a visualization of the graph and segmentation. You have to provide the path of feature, WSI file and `.xml` file. The path of feature is the output of feature step, and this step needs the cell ID and the edge information in it. `.xml` file is the annotation file same as the feature step. We only plot the range in the annotation. The visual result is written in the annotation file and can be viewed in the ImageScope software. Note that if the annotation is too large, the ImageScope will failed to open the annotation file.

## Requirements
### Packages and version
The packages required have provided in the file `requirements.txt`
### Operating systems
The code have test in the Windows and Ubuntu 16.04.7 LTS. But the installation in the different operation systems can be different because of some packages.
### Hardware
The code include the deep learning network calculation, so it needs GPU with more than 8GB video memory. And the HoVer-Net needs SSD at least 100GB for cache. The requirement of RAM depends on the data size, but we suggest that it should be more than 128GB. The code have test on the GPU NVIDIA 2080TI, RAM 128GB.

## Installation
To install the environment, you can run the command in the terminal:
```
pip install -r requirements.txt
```
The code require package `openslide python`, but its installation is different between Linux and Windows. Please follow the [offical documentation](https://openslide.org/api/python/) to install and import it in python to make sure it can work correctly.
The HoVer-Net pre-train parameter is not provided in the source code. You can download it follow the [Description](#hovernet). Instead, you can download it in our [release](https://github.com/fuscc-deep-path/sc_MTOP/releases/download/Demo/hovernet_fast_pannuke_type_tf2pytorch.tar).

## Repository Structure
`Hover`: the implementation of HoVer-Net, which clone from the offical [implementation](https://github.com/vqdang/hover_net)  
`main.py`: main function  
`F1_CellSegment.py`: segment step by calling `Hover`.  
`F3_FeatureExtract.py`: feature step by calling `WSIGraph.py`.  
`F4_Visualization.py`: visual step by calling `utils_xml.py`.  
`utils_xml.py`: define some tools to finish visualization.  
`WSIGraph.py`: define the process of feature extract.

## Usage Demo
Here is a demo to use it in the bash terminal of Ubuntu. Some commands may not work in different terminal.
To run the whole demo, you should get the demo data and pre-train parameter first. Download them with the follow command:
Download the pre-train network parameter
```
wget --no-check-certificate --content-disposition -P ./Hover https://github.com/fuscc-deep-path/sc_MTOP/releases/download/Demo/hovernet_fast_pannuke_type_tf2pytorch.tar
```
Download the demo data
```
mkdir ./wsi
wget --no-check-certificate --content-disposition -P ./wsi https://github.com/fuscc-deep-path/sc_MTOP/releases/download/Demo/Example001.ndpi
```
segment step  
```
python main.py segment --input_dir='./wsi' --output_dir='./output'
```

feature step  
```
python main.py feature --json_path='./output/json/Example001.json' --wsi_path='./wsi/Example001.ndpi' --output_path='./feature'
```

visual step  
```
python main.py visual --feature_path='./feature/sample' --wsi_path='./wsi/sample.ndpi' --xml_path='./xml/sample.xml'
```