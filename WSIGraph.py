# coding=utf-8
"""
2021-11-10
Jineng Han
FUSCC
"""
from collections import defaultdict
from tracemalloc import start
from tqdm import tqdm
from skimage.measure import regionprops
from scipy import stats
from scipy.spatial import cKDTree
from openslide import OpenSlide
import skimage.feature as skfeat
import cv2
import numpy as np
import igraph as ig
import json
import time
import os
import multiprocessing as mp
import xml.etree.ElementTree as et
import pandas as pd

try:
    mp.set_start_method('spawn')
except:
    pass


def getRegionPropFromContour(contour, bbox, extention=2):
    (left, top), (right, bottom) = bbox
    height, width = bottom - top, right - left
    # image = np.zeros((height + extention * 2, width + extention * 2), dtype=np.uint8)
    image = np.zeros((height + extention * 2,
                      width + extention * 2),
                     dtype=np.uint8)
    contour = np.array(contour)
    contour[:, 0] = contour[:, 0] - left + extention  ## 这里是因为整个image往外面扩了2个extension所以每个点相较于左边好上边都要加1个extension
    contour[:, 1] = contour[:, 1] - top + extention
    cv2.drawContours(image, [contour], 0, 1, -1)
    # TODO: check contour coords
    regionProp = regionprops(image)[0]
    return regionProp


def getCurvature(contour, n_size=5):
    contour = np.array(contour)
    contour_circle = np.concatenate([contour, contour[0:1]], axis=0)
    dxy = np.diff(contour_circle, axis=0)

    # 计算保留哪些需要计算的节点，紧密的节点会导致计算结果粗糙
    samplekeep = np.zeros((len(contour)), dtype=np.bool_)
    samplekeep[0] = True
    flag = 0
    for i in range(1, len(contour)):
        if np.abs(contour[i] - contour[flag]).sum() > 2:
            flag = i
            samplekeep[flag] = True

    contour = contour[samplekeep]
    contour_circle = np.concatenate([contour, contour[0:1]], axis=0)
    dxy = np.diff(contour_circle, axis=0)

    ds = np.sqrt(np.sum(dxy ** 2, axis=1, keepdims=True))
    ddxy = dxy / ds
    ds = (ds + np.roll(ds, shift=1)) / 2
    Cxy = np.diff(np.concatenate([ddxy, ddxy[0:1]], axis=0), axis=0) / ds
    Cxy = (Cxy + np.roll(Cxy, shift=1, axis=0)) / 2
    k = (ddxy[:, 1] * Cxy[:, 0] - ddxy[:, 0] * Cxy[:, 1]) / ((ddxy ** 2).sum(axis=1) ** (3 / 2))

    curvMean = k.mean()
    curvMin = k.min()
    curvMax = k.max()
    curvStd = k.std()

    n_protrusion = 0
    n_indentation = 0
    if n_size > len(k):
        n_size = len(k) // 2
    k_circle = np.concatenate([k[-n_size:], k, k[:n_size]], axis=0)
    for i in range(n_size, len(k_circle) - n_size):
        neighbor = k_circle[i - 5:i + 5]
        if k_circle[i] > 0:
            if k_circle[i] == neighbor.max():
                n_protrusion += 1
        elif k_circle[i] < 0:
            if k_circle[i] == neighbor.min():
                n_indentation += 1
    n_protrusion /= len(contour)
    n_indentation /= len(contour)

    return curvMean, curvStd, curvMax, curvMin, n_protrusion, n_indentation


def SingleMorphFeatures(args):
    ids, name, contours, bboxes = args
    featuresDict = defaultdict(list)
    featuresDict['name'] = name
    for contour, bbox in zip(contours, bboxes):
        regionProps = getRegionPropFromContour(contour, bbox)
        featuresDict['Area'] += [regionProps.area]
        featuresDict['AreaBbox'] += [regionProps.bbox_area]
        # featuresDict['AreaConvex'] += [regionProps.convex_area]
        # featuresDict['EquialentDiameter'] += [regionProps.equivalent_diameter]
        featuresDict['CellEccentricities'] += [regionProps.eccentricity]
        featuresDict['Circularity'] += [(4 * np.pi * regionProps.area) / (regionProps.perimeter ** 2)]
        featuresDict['Elongation'] += [regionProps.major_axis_length / regionProps.minor_axis_length]
        featuresDict['Extent'] += [regionProps.extent]
        # featuresDict['FeretDiameterMax'] += [regionProps.feret_diameter_max]
        featuresDict['MajorAxisLength'] += [regionProps.major_axis_length]
        featuresDict['MinorAxisLength'] += [regionProps.minor_axis_length]
        # featuresDict['Orientation'] += [regionProps.orientation]
        featuresDict['Perimeter'] += [regionProps.perimeter]
        featuresDict['Solidity'] += [regionProps.solidity]

        curvMean, curvStd, curvMax, curvMin, n_protrusion, n_indentation = getCurvature(contour)
        featuresDict['CurvMean'] += [curvMean]
        featuresDict['CurvStd'] += [curvStd]
        featuresDict['CurvMax'] += [curvMax]
        featuresDict['CurvMin'] += [curvMin]
        # featuresDict['NProtrusion'] += [n_protrusion]
        # featuresDict['NIndentation'] += [n_indentation]

    return featuresDict


def getMorphFeatures(name, contours, bboxes, desc, process_n=1):
    name = [int(i) for i in name]
    if process_n == 1:
        return SingleMorphFeatures([0, name, contours, bboxes])
    else:
        featuresDict = defaultdict(list)
        vertex_len = len(name)
        batch_size = vertex_len // 8
        for batch in range(0, vertex_len, batch_size):
            p_slice = [slice(batch + i, min(batch + batch_size, vertex_len), process_n) for i in range(process_n)]
            args = [[ids, name[i], contours[i], bboxes[i]] for ids, i in enumerate(p_slice)]
            with mp.Pool(process_n) as p:
                ans = p.map(SingleMorphFeatures, args)
            for q_info in ans:
                for k, v in zip(q_info.keys(), q_info.values()):
                    featuresDict[k] += v
    return featuresDict


def getCellImg(slidePtr, bbox, pad=2, level=0):
    bbox = np.array(bbox)
    bbox[0] = bbox[0] - pad
    bbox[1] = bbox[1] + pad
    cellImg = slidePtr.read_region(location=bbox[0] * 2 ** level, level=level, size=bbox[1] - bbox[0])
    cellImg = np.array(cv2.cvtColor(np.asarray(cellImg), cv2.COLOR_RGB2GRAY))
    return cellImg


def getCellMask(contour, bbox, pad=2, level=0):
    if level != 0:
        raise KeyError('Not support level now')
    (left, top), (right, bottom) = bbox
    height, width = bottom - top, right - left
    # image = np.zeros((height + extention * 2, width + extention * 2), dtype=np.uint8)
    cellMask = np.zeros((height + pad * 2,
                         width + pad * 2),
                        dtype=np.uint8)
    contour = np.array(contour)
    contour[:, 0] = contour[:, 0] - left + pad  ## 这里是因为整个image往外面扩了2个extension所以每个点相较于左边好上边都要加1个extension
    contour[:, 1] = contour[:, 1] - top + pad
    cv2.drawContours(cellMask, [contour], 0, 1, -1)
    return cellMask


def mygreycoprops(P):
    # reference https://murphylab.web.cmu.edu/publications/boland/boland_node26.html
    (num_level, num_level2, num_dist, num_angle) = P.shape
    if num_level != num_level2:
        raise ValueError('num_level and num_level2 must be equal.')
    if num_dist <= 0:
        raise ValueError('num_dist must be positive.')
    if num_angle <= 0:
        raise ValueError('num_angle must be positive.')

    # normalize each GLCM
    P = P.astype(np.float64)
    glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums

    Pxplusy = np.zeros((num_level + num_level2 - 1, num_dist, num_angle))
    Ixplusy = np.expand_dims(np.arange(num_level + num_level2 - 1), axis=(1, 2))
    P_flip = np.flip(P, axis=0)
    for i, offset in enumerate(range(num_level - 1, -num_level2, -1)):
        Pxplusy[i] = np.trace(P_flip, offset)
    SumAverage = np.sum(Ixplusy * Pxplusy, axis=0)
    Entropy = - np.sum(Pxplusy * np.log(Pxplusy + 1e-15), axis=0)
    SumVariance = np.sum((Ixplusy - Entropy) ** 2 * Pxplusy, axis=0)

    Ix = np.tile(np.arange(num_level).reshape(-1, 1, 1, 1), [1, num_level2, 1, 1])
    Average = np.sum(Ix * P, axis=(0, 1))
    Variance = np.sum((Ix - Average) ** 2 * P, axis=(0, 1))
    return SumAverage, Entropy, SumVariance, Average, Variance


def SingleGLCMFeatures(args):
    ids, wsiPath, name, contours, bboxes, pad, level = args
    slidePtr = OpenSlide(wsiPath)
    # Use wsipath as parameter because multiprocess can't use pointer like the object OpenSlide() as parameter
    featuresDict = defaultdict(list)
    featuresDict['name'] = name
    for contour, bbox in zip(contours, bboxes):
        cellImg = getCellImg(slidePtr, bbox, pad, level)
        cellmask = getCellMask(contour, bbox, pad).astype(np.bool_)
        # 去除背景
        cellImg[~cellmask] = 0

        outMatrix = skfeat.graycomatrix(cellImg, [1], [0])
        # 去除共生矩阵中与背景相关的值
        outMatrix[0, :, ...] = 0
        outMatrix[:, 0, ...] = 0

        dissimilarity = skfeat.graycoprops(outMatrix, 'dissimilarity')[0][0]
        homogeneity = skfeat.graycoprops(outMatrix, 'homogeneity')[0][0]
        # energy = skfeat.greycoprops(outMatrix, 'energy')[0][0]
        ASM = skfeat.graycoprops(outMatrix, 'ASM')[0][0]
        contrast = skfeat.graycoprops(outMatrix, 'contrast')[0][0]
        correlation = skfeat.graycoprops(outMatrix, 'correlation')[0][0]
        SumAverage, Entropy, SumVariance, Average, Variance = mygreycoprops(outMatrix)

        featuresDict['ASM'] += [ASM]
        featuresDict['Contrast'] += [contrast]
        featuresDict['Correlation'] += [correlation]
        # featuresDict['Dissimilarity'] += [dissimilarity]
        featuresDict['Entropy'] += [Entropy[0][0]]
        featuresDict['Homogeneity'] += [homogeneity]
        # featuresDict['Energy'] += [energy] #Delete because similar with ASM
        # featuresDict['Average'] += [Average[0][0]]
        # featuresDict['Variance'] += [Variance[0][0]]
        # featuresDict['SumAverage'] += [SumAverage[0][0]]
        # featuresDict['SumVariance'] += [SumVariance[0][0]]

        featuresDict['IntensityMean'] += [cellImg[cellmask].mean()]
        featuresDict['IntensityStd'] += [cellImg[cellmask].std()]
        featuresDict['IntensityMax'] += [cellImg[cellmask].max().astype('int16')]
        featuresDict['IntensityMin'] += [cellImg[cellmask].min().astype('int16')]
        # featuresDict['IntensitySkewness'] += [stats.skew(cellImg.flatten())] # Plan to delete this feature
        # featuresDict['IntensityKurtosis'] += [stats.kurtosis(cellImg.flatten())] # Plan to delete this feature
    return featuresDict


def getGLCMFeatures(wsiPath, name, contours, bboxes, pad=2, level=0, process_n=1):
    name = [int(i) for i in name]
    if process_n == 1:
        return SingleGLCMFeatures([0, wsiPath, name, contours, bboxes, pad, level])
    else:
        featuresDict = defaultdict(list)
        vertex_len = len(name)
        batch_size = vertex_len // 8
        for batch in range(0, vertex_len, batch_size):
            p_slice = [slice(batch + i, min(batch + batch_size, vertex_len), process_n) for i in range(process_n)]
            args = [[ids, wsiPath, name[i], contours[i], bboxes[i], pad, level] for ids, i in enumerate(p_slice)]
            with mp.Pool(process_n) as p:
                ans = p.map(SingleGLCMFeatures, args)
            for q_info in ans:
                for k, v in zip(q_info.keys(), q_info.values()):
                    featuresDict[k] += v
    return featuresDict


def getGraphDisKnnFeatures(name, disKnnList):
    result = defaultdict(list)
    result['name'] = name
    disKnnList[np.isinf(disKnnList)] = np.nan
    disKnnList_valid = np.ma.masked_invalid(disKnnList)
    result['minEdgeLength'] += np.min(disKnnList_valid, axis=1).tolist()
    # result['maxEdgeLength'] += np.max(disKnnList_valid, axis=1).tolist()
    result['meanEdgeLength'] += np.mean(disKnnList_valid, axis=1).tolist()
    # result['stdEdgeLength'] += np.std(disKnnList_valid, axis=1).tolist()
    # result['skewnessEdgeLength'] += stats.skew(disKnnList, axis=1, nan_policy='omit').tolist()
    # result['kurtosisEdgeLength'] += stats.kurtosis(disKnnList, axis=1, nan_policy='omit').tolist()
    return result


def getSingleGraphFeatures(args):
    subgraph, cmd = args
    result = defaultdict(list)
    n = subgraph.vcount()
    if cmd == 'name':
        result['name'] += [int(i) for i in subgraph.vs['name']]
    elif cmd == 'Nsubgraph':
        result['Nsubgraph'] += [n] * n
    elif cmd == 'Degrees':
        result['Degrees'] += subgraph.degree()
    # elif cmd == 'Eigenvector':
    #     result['Eigenvector'] += subgraph.eigenvector_centrality()  # katz别人文章里面还用过katz centerality和cluster coef
    # Slow
    elif cmd == 'Closeness':
        result['Closeness'] += subgraph.closeness()
    # Slow
    elif cmd == 'Betweenness':
        betweenness = np.array(subgraph.betweenness())
        result['Betweenness'] += betweenness.tolist()
        if n != 1 and n != 2:
            betweenness = betweenness / ((n - 1) * (n - 2) / 2)
        result['Betweenness_normed'] += betweenness.tolist()
    # elif cmd == 'AuthorityScore':
    #     result['AuthorityScore'] += subgraph.authority_score()
    elif cmd == 'Coreness':
        result['Coreness'] += subgraph.coreness()
    # elif cmd == 'Diversity':
    #     result['Diversity'] += subgraph.diversity()
    # Slow
    elif cmd == 'Eccentricity' or cmd == 'Eccentricity_normed':
        eccentricity = np.array(subgraph.eccentricity())
        result['Eccentricity'] += eccentricity.tolist()
        result['Eccentricity_normed'] += (eccentricity / n).tolist()
    # Slow
    elif cmd == 'HarmonicCentrality':
        result['HarmonicCentrality'] += subgraph.harmonic_centrality()
    # elif cmd == 'HubScore':
    #     result['HubScore'] += subgraph.hub_score()
    # elif cmd == 'NeighborhoodSize':
    #     result['NeighborhoodSize'] += subgraph.neighborhood_size()
    # elif cmd == 'Strength':
    #     result['Strength'] += subgraph.strength()
    elif cmd == 'ClusteringCoefficient':
        result['ClusteringCoefficient'] += subgraph.transitivity_local_undirected()
    return result


def getGraphCenterFeatures(graph: ig.Graph):
    result = defaultdict(list)
    # norm_cmds = ['name', 'Nsubgraph', 'Eigenvector', 'Degrees', 'AuthorityScore', 'Coreness', 'Diversity',
    #             'HubScore', 'NeighborhoodSize', 'Strength', 'ClusteringCoefficient']
    norm_cmds = ['name', 'Nsubgraph', 'Degrees',
                 # 'AuthorityScore', 'HubScore', 'Eigenvector',
                 'Coreness', 'ClusteringCoefficient']
    multi_cmds = ['Eccentricity', 'HarmonicCentrality', 'Closeness', 'Betweenness']
    for subgraph in tqdm(graph.decompose()):
        for cmd in norm_cmds:
            args = [subgraph, cmd]
            ans = getSingleGraphFeatures(args)
            for k, v in zip(ans.keys(), ans.values()):
                result[k] += v
        if subgraph.vcount() > 50000:  # Huge graph, use multiprocessing
            args = [[subgraph, cmd] for cmd in multi_cmds]
            with mp.Pool() as p:
                ans = p.map(getSingleGraphFeatures, args)
            for q_info in ans:
                for k, v in zip(q_info.keys(), q_info.values()):
                    result[k] += v
        else:  # Small graph, directly calucate
            for cmd in multi_cmds:
                args = [subgraph, cmd]
                ans = getSingleGraphFeatures(args)
                for k, v in zip(ans.keys(), ans.values()):
                    result[k] += v
    return result


def constructGraphFromDict(
        wsiPath: str, nucleusInfo: dict, distanceThreshold: float,
        knn_n: int = 5, level: int = 0, offset=np.array([0, 0])
):
    r"""Construct graph from nucleus information dictionary

    Parameters
    ----------
    nucleusInfo : dict
        'mag': int
            magnification of the result
        'nuc': dict
            nucleus information
            'nuclei ID' : dict
                note that ID generated from HoverNet is not continuous
                'bbox' : list
                    [[left, top], [right, bottom]]
                'centroid' : list
                    [column, row]
                'contour' : list, from cv2.findContours
                    [[column1, row1], [column2, row2], ... ]
                'type_prob' : float
                    The probability of current nuclei belonging to type 'type'
                'type' : int

    distanceThreshold : maximum distance in magnification of 40x

    typeDict : dict
        "0" : "nolabe"
        "1" : "neopla"
        "2" : "inflam"
        "3" : "connec"
        "4" : "necros"
        "5" : "no-neo"

    cellSize : int, odd
        size of cell cropped for extracting GLCM features

    level : int
        level for reading WSI
        0 : 40x
        1 : 20x
        ...

    Returns
    -------
    graph :

    """
    typeDict2 = {
        'neolabe': 0,
        'neopla': 1,
        'inflame': 2,
        'connect': 3,
        'necros': 4,
        'normal': 5
    }
    offset = np.array([0, 0])
    print(f"{'Total 9 steps: 0 ~ 8':*^30s}")
    mag = nucleusInfo['mag']
    distanceThreshold = distanceThreshold / (40.0 / mag)

    bboxes, centroids, contours, types = [], [], [], []

    for nucInfo in tqdm(nucleusInfo['nuc'].values(),
                        desc="0. Preparing"):  ##nucleusInfo['nuc']得到的是一个序号作为键，各种信息作为value的字典，这里只取后者
        # ! nucInfo['bbox'] doesn't match nucInfo['contour']  ##nucInfo里面本身有bbox，为什么要从新算 并且算出来的和本身的的还不一样？？？可视化代码是否也是从新计算？
        tmpCnt = np.array(nucInfo[
                              'contour'])  ##contour是等高线，一个N行2列的数组，每一行是一点二维平面的点，勾勒细胞的轮廓？？？ x坐标最小值是left，y坐标最小值是top；x坐标最大值和y坐标最大值是right和bottom
        left, top = tmpCnt.min(0)  ## 这里的0应该是axis=0,也就是按照行来看最小值，所以最小值有2个
        right, bottom = tmpCnt.max(0)
        bbox = [[left + offset[0], top + offset[1]], [right + offset[0], bottom + offset[1]]]
        bboxes.append(bbox)  # [[[, ],[, ]], [[, ],[, ]], ......]
        centroids.append(nucInfo['centroid'])  ## [[, ], [, ], ......]
        contours.append(nucInfo['contour'])
        types.append(nucInfo['type'])  ## [, , , ......]
    assert len(bboxes) == len(centroids) == len(
        types), 'The attribute of nodes (bboxes, centroids, types) must have same length'
    vertex_len = len(bboxes)
    globalGraph = ig.Graph()
    names = [str(i) for i in range(vertex_len)]  ##按理说这里的name应该又变成了连续的？？？

    globalGraph.add_vertices(vertex_len, attributes={
        'name': names, 'Bbox': bboxes, 'Centroid': centroids,
        'Contour': contours, 'CellType': types})

    print('Getting morph features')
    t1 = time.time()
    morphFeats = getMorphFeatures(names, contours, bboxes, 'MorphFeatures', process_n=8)
    for k, v in zip(morphFeats.keys(),
                 morphFeats.values()):  ##morphFeats这个字典是按照特征作为键，每个特征名称作为键对应的值是特征列表，包含若干细胞这个特征的值，例如第一个键elongation就有472726个,和globalGraph的vs长度一致
     if k != 'name':
         globalGraph.vs[morphFeats['name']][
             'Morph_' + k] = v  ## 这里这个vs确实不太清楚，但是结合217-219行那里都是字典的形式，来存储特征值，这里也可以理解为往globalGraph上面加字典？
    print(f"{'morph features cost':#^40s}, {time.time() - t1:*^10.2f}")

    print('Getting GLCM features')
    t2 = time.time()
    GLCMFeats = getGLCMFeatures(wsiPath, names, contours, bboxes, pad=2, level=level, process_n=8)
    for k, v in zip(GLCMFeats.keys(),
                 GLCMFeats.values()):  ##morphFeats这个字典是按照特征作为键，每个特征名称作为键对应的值是特征列表，包含若干细胞这个特征的值，例如第一个键elongation就有472726个,和globalGraph的vs长度一致
        if k != 'name':
            globalGraph.vs[GLCMFeats['name']][
                'Texture_' + k] = v  ## 这里这个vs确实不太清楚，但是结合217-219行那里都是字典的形式，来存储特征值，这里也可以理解为往globalGraph上面加字典？
    print(f"{'GLCM features cost':#^40s}, {time.time() - t2:*^10.2f}")

    t3 = time.time()
    nolabeIDs, neoplaIDs, inflamIDs, connecIDs, necrosIDs, normalIDs = \
        [np.where(np.array(types) == i)[0].tolist() for i in range(6)]  ## 这里的6个IDs，每个都是一个列表，总的顺序是连续的从0开始
    # timeMorph, timeGLCM = 0, 0

    edge_info = pd.DataFrame({'source': [], 'target': [], 'featype': []})

    # Neopla->T, Inflam->I, Connec->S, Normal->N
    featype_dict = {'T-T': [neoplaIDs, neoplaIDs],
                    ## 字典的前3个value，都是[[1,2,3,......]]这样列表套列表的list；后面则是列表里面2个列表，类似于[[1,2,3,....],[2,3,4,...]]
                    'I-I': [inflamIDs, inflamIDs],
                    'S-S': [connecIDs, connecIDs],
                    # 'N-N': [normalIDs, normalIDs],
                    'T-I': [neoplaIDs, inflamIDs],
                    'T-S': [neoplaIDs, connecIDs],
                    # 'T-N': [neoplaIDs, normalIDs],
                    'I-S': [inflamIDs, connecIDs],
                    # 'I-N': [inflamIDs, normalIDs],
                    # 'S-N': [connecIDs, normalIDs]
    }

    for featype, featype_index_list in zip(featype_dict.keys(),
                                           featype_dict.values()):  ##这里对featype_dict的value进行迭代，featype_index_list要么是列表套列表，要么是列表里面套2个列表
        print(f'Getting {featype} graph feature')
        print(f'---Creating edges')
        # Treat neopla and normal as the same cell type by making the same cellTypeMark,
        # and delete the edge between vertexs which have the same cellTypeMark
        pairs = np.array([]).reshape((0, 2))
        disKnnList = np.array([]).reshape((0, knn_n))
        subgraph_names = []
        featype_index = []

        for index in featype_index_list:
            featype_index += index
        for src, i_src in enumerate(featype_index_list):
            for tar, i_tar in enumerate(featype_index_list):
                if src != tar:
                    centroid_tar = globalGraph.induced_subgraph(i_tar).vs['Centroid']
                    centroid_src = globalGraph.induced_subgraph(i_src).vs['Centroid']
                    n_tar = len(i_tar)
                    n_src = len(i_src)
                    tree = cKDTree(centroid_tar)

                    if i_src == i_tar:
                        disknn, vertex_index = tree.query(centroid_src, k=knn_n + 1,
                                                          distance_upper_bound=distanceThreshold, p=2)
                        disknn = disknn[..., -knn_n:]
                        vertex_index = vertex_index[..., -knn_n:]
                    else:
                        disknn, vertex_index = tree.query(centroid_src, k=knn_n, distance_upper_bound=distanceThreshold,
                                                          p=2)

                    knn_mask = vertex_index != n_tar  # delete the vertex whose distance upper bound
                    v_src = np.tile(np.array(i_src, dtype='str').reshape((n_src, -1)), (1, knn_n))[knn_mask]
                    v_tar = np.array(i_tar, dtype='str')[vertex_index[knn_mask]]

                    pairs = np.concatenate([pairs, np.stack((v_src, v_tar), axis=1)], axis=0)
                    disKnnList = np.concatenate([disKnnList, disknn], axis=0)
                    subgraph_names += i_src

        subgraph = globalGraph.induced_subgraph(featype_index)

        subgraph.add_edges(pairs[:, 0:2])
        multiple_edge = subgraph.es[np.where(np.array(subgraph.is_multiple()))[0].tolist()]
        subgraph.delete_edges(multiple_edge)  # delete the multiple edge

        subgraph_edge = subgraph.get_edge_dataframe()
        subgraph_vname = subgraph.get_vertex_dataframe()['name']
        subgraph_edge['source'] = [subgraph_vname[a] for a in subgraph_edge['source'].values]
        subgraph_edge['target'] = [subgraph_vname[a] for a in subgraph_edge['target'].values]
        subgraph_edge.insert(subgraph_edge.shape[1], 'featype', featype)

        edge_info = pd.concat([edge_info, subgraph_edge])

        print(f'---Getting DisKnn features')
        feats = getGraphDisKnnFeatures(subgraph_names, disKnnList)
        for k, v in zip(feats.keys(), feats.values()):
            if k != 'name':
                globalGraph.vs[feats['name']]['Graph_' + featype + '_' + k] = v

        print(f'---Getting GraphCenter features')
        feats = getGraphCenterFeatures(subgraph)
        for k, v in zip(feats.keys(), feats.values()):
            if k != 'name':
                globalGraph.vs[feats['name']]['Graph_' + featype + '_' + k] = v

    # print(f"{'Graph features cost':#^40s}, {time.time() - t3:*^10.2f}")

    # Stroma barrier
    # For each inflam node, add it to neoplaAddConnecGraph, compute barrier and delete
    # ! Why select the maximum subgraph, if distanceThreshold wasn't set appropriately, shortestPathsLymCancer would be empty
    t4 = time.time()
    centroid_T = globalGraph.induced_subgraph(neoplaIDs).vs['Centroid']
    centroid_I = globalGraph.induced_subgraph(inflamIDs).vs['Centroid']
    centroid_S = globalGraph.induced_subgraph(connecIDs).vs['Centroid']
    Ttree = cKDTree(centroid_T)
    STree = cKDTree(centroid_S)
    dis, pairindex_T = Ttree.query(centroid_I, k=1)  # 距离I最近的T的距离和索引
    paircentroid_T = np.array(centroid_T)[pairindex_T]  # 最近的T细胞坐标
    barrier = []
    for Tcoor, Icoor, r in tqdm(zip(centroid_I, paircentroid_T, dis), total=len(centroid_I)):
        # 分别计算在r距离内间质细胞的数量
        set1 = set(STree.query_ball_point(Tcoor, r))
        set2 = set(STree.query_ball_point(Icoor, r))
        barrier.append(len(set1 & set2))
    globalGraph.vs[inflamIDs]['stromaBarrier'] = barrier
    print(f"{'stroma barrier cost':#^40s}, {time.time() - t4:*^10.2f}")

    return globalGraph, edge_info


typeDict = {
    0: "nolabe",
    1: "neopla",
    2: "inflam",
    3: "connec",
    4: "necros",
    5: "normal"
}

typeDict2 = {
    'neolabe': 0,
    'neopla': 1,
    'inflame': 2,
    'connect': 3,
    'necros': 4,
    'normal': 5
}

if __name__ == '__main__':
    json_dir = 'D:\\immune_DL\\DATA\\InferRes_Q2T\\'
    WSI_dir = 'D:\\immune_DL\\DATA\\WSI_Q2T\\'
    presplit_dir = 'D:\\immune_DL\\DATA\\CBCGA_WSIxml_forhovernet\\'
    FinalFeats_dir = 'D:\\immune_DL\\DATA\\FinalFeats0503_Q2T\\'
    distanceThreshold = 100
    level = 0
    k = 5

    for jsonres in os.listdir(json_dir):
        sampleres_dir = os.path.join(FinalFeats_dir, jsonres.split('.')[0])

        if os.path.exists(sampleres_dir):
            print(jsonres, 'has been processed')
            continue
        else:
            print('--------------------now process', jsonres, '--------------------')
            presplit_file = os.path.join(presplit_dir, jsonres.split('.')[0] + '.xml')
            if os.path.exists(presplit_file):
                tree = et.parse(presplit_file)
                root = tree.getroot()
                box = np.array([[float(vertex.get('X')), float(vertex.get('Y'))] for vertex in
                                root.find('Annotation').find('Regions').find('Region').find('Vertices').findall(
                                    'Vertex')])
                start_point = np.array([box[:, 0].min(), box[:, 1].min()])
            else:
                start_point = np.array([0, 0])

            with open(os.path.join(json_dir, jsonres)) as fp:
                print(f"{'Loading json':*^30s}")
                nucleusInfo = json.load(fp)
                wsiPath = os.path.join(WSI_dir, jsonres.replace('.json', '.ndpi'))

            globalgraph, edge_info = constructGraphFromDict(wsiPath, nucleusInfo, distanceThreshold, k, level,
                                                            start_point)
            vertex_dataframe = globalgraph.get_vertex_dataframe()

            col_dist = defaultdict(list)
            cellType = ['T', 'I', 'S', 'N']
            for featname in vertex_dataframe.columns.values:
                if 'Graph' not in featname:
                    # public feature, including cell information, Morph feature and GLCM feature
                    for cell in cellType:
                        col_dist[cell] += [featname] if featname != 'Contour' else []
                else:
                    # Graph feature, format like 'Graph_T-I_Nsubgraph'
                    for cell in cellType:
                        featype = featname.split('_')[1]  # Graph feature type like 'T-T', 'T-I'
                        col_dist[cell] += [featname] if cell in featype else []
            cellType_save = {'T': [1],  # Neopla
                             'I': [2],  # Inflam
                             'S': [3],  # Connec
                             'N': [5]}  # Normal

            os.makedirs(sampleres_dir)
            for i in col_dist.keys():
                vertex_csvfile = os.path.join(sampleres_dir, jsonres.split('.')[0] + '_vertex_' + i + '.csv')
                save_index = vertex_dataframe['CellType'].isin(cellType_save[i]).values
                vertex_dataframe.iloc[save_index].to_csv(vertex_csvfile, index=False, columns=col_dist[i])
            edge_csvfile = os.path.join(sampleres_dir, jsonres.split('.')[0] + '_edge.csv')
            # globalgraph.get_edge_dataframe().to_csv(edge_csvfile, index=False)
            edge_info.to_csv(edge_csvfile, index=False)

