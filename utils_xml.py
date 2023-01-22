import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
from openslide import OpenSlide
import cv2
import pathlib as plb

clusters_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]# 11个聚类
cellTypes_id = [0, 1, 2, 3, 4, 5]
edges_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
edgeDict = {
    0 : 'T-T',
    1 : 'I-I',
    2 : 'S-S',
    3 : 'N-N',
    4 : 'T-I',
    5 : 'T-S',
    6 : 'T-N',
    7 : 'I-S',
    8 : 'I-N',
    9 : 'S-N',
}
edgeDictReverse = {
    'T-T' : 0,
    'I-I' : 1,
    'S-S' : 2,
    'N-N' : 3,
    'T-I' : 4,
    'T-S' : 5,
    'T-N' : 6,
    'I-S' : 7,
    'I-N' : 8,
    'S-N' : 9,
}
typeDict = {
        0 : "nolabe",
        1 : "neopla",
        2 : "inflam",
        3 : "connec",
        4 : "necros",
        5 : "normal",
}

colormap = mpl.cm.get_cmap('tab20')
clusters_RGB = []
for i in np.linspace(0, 1, 20)[::2]:
        clusters_RGB.append([int(colormap(i, bytes=True)[c]) for c in range(3)])
for i in np.linspace(0, 1, 20)[1::2]:
        clusters_RGB.append([int(colormap(i, bytes=True)[c]) for c in range(3)])
assert len(clusters_id) < len(clusters_RGB), f"the num of classes ({len(clusters_id)}) can't more than the num of RGB list ({len(clusters_RGB)})"

cellType_RGB = [(0  ,   0,   0),#nolabe no color 
                (255,   0,   0),#neopla red
                (0  , 255,   0),#inflam green
                (0  ,   0, 255),#connec blue
                (255, 255,   0),#necros
                (255, 165,   0)]#normal

colormap = mpl.cm.get_cmap('tab20')
edge_RGB= []
for i in np.linspace(0, 1, 20)[::2]:
        edge_RGB.append([int(colormap(i, bytes=True)[c]) for c in range(3)])
for i in np.linspace(0, 1, 20)[1::2]:
        edge_RGB.append([int(colormap(i, bytes=True)[c]) for c in range(3)])
HEX2RGB = lambda x:(int(x[1:3], 16), int(x[3:5], 16), int(x[5:7], 16))
edge_RGB[0] = HEX2RGB('#FF0000')
edge_RGB[1] = HEX2RGB('#00FF00')
edge_RGB[2] = HEX2RGB('#0000FF')
edge_RGB[4] = HEX2RGB('#FDB462') ##set3-6 浅橙色
edge_RGB[5] = HEX2RGB('#F781BF') ##set1-8 粉色
edge_RGB[7] = HEX2RGB('#8DD3C7')  ##set3-1 青色
assert len(edges_id) < len(edge_RGB), f"the num of classes ({len(edges_id)}) can't more than the num of RGB list ({len(edge_RGB)})"

# 进制转换，xml文件的颜色RGB通道数是反着的且是十进制
# Convert RGB to OCT
# clusters_RGBOCT = []
# for rgb in clusters_RGB:
#     digit = '0123456789ABCDEF'
#     rgb.reverse()
#     rgb_hex = ''
#     for i in rgb:
#         rgb_hex += digit[i//16] + digit[i%16]
#     clusters_RGBOCT.append(int(rgb_hex, 16))
# 
# cellType_RGBOCT = []
# for rgb in cellType_RGB:
#     digit = '0123456789ABCDEF'
#     rgb.reverse()
#     rgb_hex = ''
#     for i in rgb:
#         rgb_hex += digit[i//16] + digit[i%16]
#     cellType_RGBOCT.append(int(rgb_hex, 16))
#     
# edge_RGBOCT = []
# for rgb in edge_RGB:
#     digit = '0123456789ABCDEF'
#     rgb.reverse()
#     rgb_hex = ''
#     for i in rgb:
#         rgb_hex += digit[i//16] + digit[i%16]
#     edge_RGBOCT.append(int(rgb_hex, 16))

def get_windows(xml_path):
    window_bbox = []
    assert os.path.exists(xml_path), f"Can't find {xml_path}"
    tree = ET.parse(xml_path)
    annos = tree.getroot()
    for anno in annos.findall('Annotation'):
        if anno.get('LineColor')=='16777215':
            for vs in anno.iter('Vertices'):
                wx = []
                wy = []
                for v in vs.iter('Vertex'):
                    wx.append(float(v.get('X')))
                    wy.append(float(v.get('Y')))
                window_bbox.append([[min(wx), min(wy)], 
                                    [max(wx), max(wy)]])
    return window_bbox

####################################
class make_graph_img():
    def __init__(self, data=None):
        self.data = data
        if self.data is None:
            self.data = dict()
            self.data['bbox'] = np.empty((0, 2, 2))
            self.data['class'] = np.empty((0))
            self.data['type'] = np.empty((0))
            self.data['offset'] = np.empty((0, 1, 2))
            self.data['group'] = np.empty((0))
        self.annos = ET.Element('Annotations')
        self.tree = ET.ElementTree(self.annos)
        self.layer_id = 1
        self.window_bbox = None
    
    def read_csv(self, feature_path,
                 wsi_path,
                 xml_path,
                 specell_names=None,
                 bbox_size=None,
                 omit_cell = [],
                 omit_edge = []):
        
        sample_name = plb.Path(wsi_path).stem
        self.sample_name = sample_name
        data = self.data
        wsi = OpenSlide(wsi_path)

        # %%
        # Read csv file
        # csv_file = ['T', 'I', 'S', 'N']
        csv_file = ['T', 'I', 'S']
        csv_data = None
        n_data = 0
        for c in csv_file:
            if csv_data is None:
                csv_data = pd.read_csv(os.path.join(feature_path, sample_name+'_Feats_'+c+'.csv'))
            else:
                temp = pd.read_csv(os.path.join(feature_path, sample_name+'_Feats_'+c+'.csv'))
                csv_data = pd.concat([csv_data, temp])
        inbox_name = np.empty((0), dtype=np.int64)
        inbox_centroid = np.empty((0, 2), dtype=np.float32)
        centroid = np.array(list(map(lambda x:eval(x), csv_data.Centroid)))
        name = csv_data.name.values

        # %%
        # Get window bbox
        if specell_names is None:
            if os.path.exists(xml_path):
                window_bbox = get_windows(xml_path)
            else:
                raise OSError('xml file is not exist')
                # window_bbox = [[[0, 0], [wsi.level_dimensions[0][0], wsi.level_dimensions[0][1]]]]
        else:
            assert bbox_size is not None, 'Must give bbox size when cell is not None.'
            window_bbox = []
            for specell_name in specell_names:
                specell_centroid = centroid[np.array(csv_data['name'])==specell_name][0]
                bbox_size = np.array(bbox_size)
                window_bbox.append([specell_centroid - bbox_size/2,
                                    specell_centroid + bbox_size/2])

        # window_size = [(bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1]) for bbox in window_bbox]
        # maxindex = np.argmax(window_size)
        # roi_window_bbox = window_bbox.pop(maxindex)
        window_bbox = np.array(window_bbox)
        self.window_bbox = window_bbox
        try:
            self.img = [wsi.read_region(location=i[0].astype('int'),
                                        level=0,
                                        size=(i[1]-i[0]).astype('int')) for i in window_bbox]
            self.img = [np.array(i)[:, :, 0:3].copy() for i in self.img]
        except:
            self.img = []
            for window in window_bbox:
                window_size = window[1] - window[0]
                patch = 1000
                self.img.append(np.zeros([window_size[1], window_size[0], 3], dtype=np.uint8))
                for x in range(window[0][0], window[1][0], patch):
                    for y in range(window[0][1], window[1][1], patch):
                        y_end = min(y+patch, window[1][1])
                        x_end = min(x+patch, window[1][0])
                        patch_size = [x_end-x, y_end-y]
                        self.img[-1][y-window[0][1]:y_end-window[0][1],
                                     x-window[0][0]:x_end-window[0][0]] = np.array(wsi.read_region(location=(x, y),
                                                                                                   level=0,
                                                                                                   size=patch_size))[:, :, 0:3]
        self.rawimg = self.img.copy()
        
        # %%
        # Read vertex information
        ## Prepare index
        index = np.zeros((len(centroid), len(window_bbox)), dtype=np.bool_)
        for i in range(len(window_bbox)):
            index[:, i] = ((window_bbox[i, 0, 0]<centroid[:, 0]) & (centroid[:,0]<window_bbox[i, 1, 0])) &\
                ((window_bbox[i, 0, 1]<centroid[:,1]) & (centroid[:,1]<window_bbox[i, 1, 1]))
        index_x, index_y = np.where(index)

        ## Get data
        group = index_y
        offset = window_bbox[group, 0:1, :]
        name = name[index_x]
        centroid = centroid[index_x]
        cellType = csv_data.CellType.values[index_x]
        bbox = np.stack([centroid-3, centroid+3], 1)
        bbox -= offset

        if len(omit_cell)>0:
            index = np.zeros((len(cellType), len(omit_cell)))
            for i in range(len(omit_cell)):
                index[:, i] = cellType!=omit_cell[i]
            index = index.sum(axis=1)==len(omit_cell)
            
            group = group[index]
            offset = offset[index]
            name = name[index]
            centroid = centroid[index]
            cellType = cellType[index]
            bbox = bbox[index]
        
        data['bbox'] = np.concatenate([data['bbox'], bbox], 0)
        data['class'] = np.concatenate([data['class'], cellType])
        data['type'] = np.concatenate([data['type'], 2*np.ones(len(bbox), dtype=np.int8)])
        data['offset'] = np.concatenate([data['offset'], offset])
        data['group'] = np.concatenate([data['group'], group])

        # %%
        # Read edge csv
        ## Prepare index from table data
        csv_data = pd.read_csv(os.path.join(feature_path, sample_name+'_Edges'+'.csv'))
        source = csv_data.source
        target = csv_data.target

        ## Preapare table data
        inbox_name = np.concatenate([inbox_name, name])
        inbox_centroid = np.concatenate([inbox_centroid, centroid])
        n_data = max(max(source), max(target))
        name_inbox_tab = np.zeros((n_data + 1), dtype=np.bool_)
        name_inbox_tab[inbox_name] = True
        offset_tab = np.zeros((n_data + 1, 1, 2))
        offset_tab[inbox_name] = offset
        group_tab = np.zeros((n_data + 1), dtype=np.int16)
        group_tab[inbox_name] = group
        centroid_inbox_tab = np.zeros((n_data + 1, 2))
        centroid_inbox_tab[inbox_name] = inbox_centroid
        index = name_inbox_tab[source] & name_inbox_tab[target] & (group_tab[source] == group_tab[target])
        

        ## Get data
        group = group_tab[source][index]
        edgeType = csv_data.featype[index]
        bbox = np.stack([centroid_inbox_tab[source], centroid_inbox_tab[target]], axis=1)[index]
        offset = offset_tab[source][index]
        bbox -= offset

        if len(omit_edge)>0:
            index = np.zeros((len(edgeType), len(omit_edge)))
            for i in range(len(omit_edge)):
                index[:, i] = edgeType!=omit_edge[i]
            index = index.sum(axis=1)==len(omit_edge)
            
            group = group[index]
            edgeType = edgeType[index]
            bbox = bbox[index]
            offset = offset[index]

        data['bbox'] = np.concatenate([data['bbox'], bbox], axis = 0)
        data['class'] = np.concatenate([data['class'], edgeType])
        data['type'] = np.concatenate([data['type'], np.zeros(len(bbox), dtype=np.int8)])
        data['offset'] = np.concatenate([data['offset'], offset])
        data['group'] = np.concatenate([data['group'], group])
        
        self.data = data

    
    def make_img_element(self, type, draw_fun):
        index = self.data['type'] == type
        bboxes = self.data['bbox'][index]
        classes = self.data['class'][index]
        groups = self.data['group'][index]
        for i in range(len(self.window_bbox)):
            for bbox, myclass in zip(bboxes[groups==i], classes[groups==i]):
                # cv2.circle(img[i], center, radius = 3, color=cellType_RGB[myclass], thickness='FILLED')
                draw_fun(self.img[i], bbox, myclass)
            
    
    def make_img_circle(self):
        def draw_circle(img, bbox, myclass, radius=7):
            center = bbox.mean(axis=0).astype('int')
            center[center<0] = 0
            cv2.circle(img, tuple(center), radius=radius, color=cellType_RGB[int(myclass)], thickness=-1)
        self.make_img_element(type=2, draw_fun=draw_circle)

    def make_img_line(self, thickness=5):
        def draw_line(img, bbox, myclass):
            pt1 = bbox[0].astype('int')
            pt2 = bbox[1].astype('int')
            cv2.line(img, tuple(pt1), tuple(pt2), color=edge_RGB[edgeDictReverse[myclass]], thickness=thickness)
        self.make_img_element(type=0, draw_fun=draw_line)

    def make_img_rect(self, thickness=5):
        def draw_line(img, bbox, myclass):
            pt1 = bbox[0].astype('int')
            pt2 = bbox[1].astype('int')
            cv2.rectangle(img, tuple(pt1), tuple(pt2), color=clusters_RGB[myclass], thickness=thickness)
        self.make_img_element(type=1, draw_fun=draw_line)

    def make_img(self, line_size=5, rect_size=5):
        self.img = self.rawimg.copy()
        self.make_img_line(line_size)
        self.make_img_circle()
        self.make_img_rect(rect_size)

    def write_img(self):
        from io import BytesIO
        for i in range(len(self.window_bbox)):
            # code = cv2.imencode('.png', self.img[i][:, :, ::-1])[1]  # 保存图片
            # byte_stream = BytesIO(code.tobytes())
            # with open(f'./fun_fig/graph_{self.sample_name}_plot{i}.png','wb') as p: # 可以保存任意路径
            #     p.write(code.tobytes())
            cv2.imwrite(f'./fun_fig/graph_{self.sample_name}_plot{i}.jpg', self.img[i][:, :, ::-1])