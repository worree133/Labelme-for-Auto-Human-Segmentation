'''
lanhuage: python
Descripttion: 
version: beta
Author: xiaoshuyui
Date: 2020-06-12 09:44:19
LastEditors: xiaoshuyui
LastEditTime: 2021-02-19 16:45:57
'''
try:
    from labelme import __version__ as labelme_version
except:
    labelme_version = '4.2.9'

from labelme.convertmask import baseDecorate
import sys

sys.path.append("..")

import json
import copy
import json
import math
import os
from shapely.geometry import Polygon, LineString, Point, box
import cv2
import numpy as np
import skimage.io as io
import yaml

from labelme.convertmask.utils.methods.get_shape import *
from labelme.convertmask.utils.methods.img2base64 import imgEncode
from labelme.shape import Shape
# import warnings
from labelme.convertmask.utils.methods.logger import logger
from operator import add
from qtpy import QtCore

def rs(st:str):
    s = st.replace('\n','').strip()
    return s

def readYmal(filepath):
    if os.path.exists(filepath) and filepath.endswith('info.txt'):

        f = open(filepath,'r',encoding='utf-8')
        classList = f.readlines()
        f.close()
        l3 = [rs(i) for i in classList]

        return l3
    else:
        raise FileExistsError('label file not found. Make sure you create a info.txt file in the current working directory.')



def getMultiShapes(label_img,
                   oriImgPath='',
                   x=0,y=0,x2=0,y2=0,matting=None,groupid=None
                   ):

    BASE_DIR = os.path.abspath(os.getcwd())

    labels = readYmal(BASE_DIR + '/info.txt')

    shapes = []



    region = process(label_img.astype(np.uint8))

    if len(region)>1:
        region.sort(key=len,reverse=True)
        cache=[]
        for subregion in region:
            shape = dict()
            points = []
            for i in range(0, subregion.shape[0]):
                points.append(list(map(add,subregion[i][0].tolist(),[x,y])))

            p=Polygon(points)
            cache.append(p)

            shape['flags'] = {}
            shape['group'] = groupid
            shape['group_id'] = None

            if len(cache) > 1:

                if cache[0].intersects(cache[1]) == True:
                    shape['label'] = labels[0]
                    #shape['matting'] = None
                else:
                    shape['label'] = labels[1]
                    #shape['matting'] = matting
                cache.pop()
            else:
                shape['label'] = labels[1]
                #shape['matting'] = matting

            shape['model'] = 'vitae'
            shape['points'] = points
            shape['rectpts'] = [[x, y], [x2, y2]]
            shape['shape_type'] = 'polygon'


            shapes.append(shape)


    else:
        for subregion in region:
            points = []
            for i in range(0, subregion.shape[0]):
                points.append(list(map(add, subregion[i][0].tolist(), [x, y])))

            shape = dict()
            shape['flags'] = {}
            shape['group'] = groupid
            shape['group_id'] = None
            shape['label'] = labels[1]
            shape['model'] = 'vitae'
            shape['points'] = points
            shape['rectpts'] = [[x, y], [x2, y2]]
            shape['shape_type'] = 'polygon'


            shapes.append(shape)


    (ImgPath,imgname) = os.path.split(oriImgPath)

    saveJsonPath = ImgPath + os.sep + imgname[:-4] + '.json'

    shapesave = None

    if os.path.exists(saveJsonPath):
        with open(saveJsonPath, 'r') as f:
            obj = json.load(f)
            shapesave = copy.deepcopy(obj['shapes'])
            for shape in shapes:
                obj['shapes'].append(shape)

    else:
        shapesave = []
        obj = dict()
        obj['shapes'] = shapes
        obj['imagePath'] = imgname
        obj['version'] = labelme_version
        obj['flags'] = {}
        obj['imageData'] = str(imgEncode(oriImgPath))


    j = json.dumps(obj, sort_keys=True, indent=4)


    with open(saveJsonPath, 'w') as f:
        f.write(j)




    return shapesave, groupid+1


def matte2shape(label_img,matting=None,groupid=None,
                   oriImgPath='',
                   x=0,y=0,x2=0,y2=0,
                   ):

    BASE_DIR = os.path.abspath(os.getcwd())

    labels = readYmal(BASE_DIR + '/info.txt')

    shapes = []

    matting = cv2.cvtColor(matting, cv2.COLOR_RGB2GRAY)


    matting = matting.tolist()



    region = process(label_img.astype(np.uint8))

    if len(region)>1:
        region.sort(key=len,reverse=True)
        cache=[]
        for subregion in region:
            shape = dict()
            points = []
            for i in range(0, subregion.shape[0]):
                points.append(list(map(add,subregion[i][0].tolist(),[x,y])))

            p=Polygon(points)
            cache.append(p)

            shape['flags'] = {}
            shape['group'] = groupid
            shape['group_id'] = None

            if len(cache) > 1:

                if cache[0].intersects(cache[1]) == True:
                    shape['label'] = labels[0]
                    #shape['matting'] = None
                else:
                    shape['label'] = labels[1]
                    shape['matting'] = matting
                cache.pop()
            else:
                shape['label'] = labels[1]
                shape['matting'] = matting

            shape['model'] = 'matteformer'
            shape['points'] = points
            shape['rectpts'] = [[x, y], [x2, y2]]
            shape['shape_type'] = 'polygon'


            shapes.append(shape)


    else:
        for subregion in region:
            points = []
            for i in range(0, subregion.shape[0]):
                points.append(list(map(add, subregion[i][0].tolist(), [x, y])))

            shape = dict()
            shape['flags'] = {}
            shape['group'] = groupid
            shape['group_id'] = None
            shape['label'] = labels[1]
            shape['matting'] = matting
            shape['model'] = 'matteformer'
            shape['points'] = points
            shape['rectpts'] = [[x, y], [x2, y2]]
            shape['shape_type'] = 'polygon'


            shapes.append(shape)

    for shape in shapes:
        if shape['label'] == labels[1]:
            p = Polygon(shape['points'])
            shape['bound'] = p.bounds


    (ImgPath,imgname) = os.path.split(oriImgPath)

    saveJsonPath = ImgPath + os.sep + imgname[:-4] + '.json'

    shapesave = None

    if os.path.exists(saveJsonPath):
        with open(saveJsonPath, 'r') as f:
            obj = json.load(f)
            shapesave = copy.deepcopy(obj['shapes'])
            for shape in shapes:
                obj['shapes'].append(shape)

    else:
        shapesave = []
        obj = dict()
        obj['shapes'] = shapes
        obj['imagePath'] = imgname
        obj['version'] = labelme_version
        obj['flags'] = {}
        obj['imageData'] = str(imgEncode(oriImgPath))

    j = json.dumps(obj, sort_keys=True, indent=4)


    with open(saveJsonPath, 'w') as f:
        f.write(j)


    return shapesave, groupid+1

def outputNewshape(oriImgPath, nshape, orishape):


    (ImgPath, imgname) = os.path.split(oriImgPath)

    BASE_DIR = os.path.abspath(os.getcwd())
    labels = readYmal(BASE_DIR + '/info.txt')
    saveJsonPath = ImgPath + os.sep + imgname[:-4] + '.json'



    with open(saveJsonPath, 'r') as f:
        obj = json.load(f)
        shapesave = copy.deepcopy(obj['shapes'])
        for shape in obj['shapes']:
            if shape['points'] == orishape:
                shape['points'] = nshape



    j = json.dumps(obj, sort_keys=True, indent=4)

    with open(saveJsonPath, 'w') as f:
        f.write(j)



    return shapesave


def getNewShapes(label_img,groupid,rectpts,model,
                   oriImgPath='',
                   x1=0,y1=0,x2=0,y2=0
                   ):

    (ImgPath,imgname) = os.path.split(oriImgPath)


    #with open(savePath, 'w') as f:
        #f.write(str(lastshapes))

    BASE_DIR = os.path.abspath(os.getcwd())
    labels = readYmal(BASE_DIR + '/info.txt')

    shapes = []
    p0 = Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    l0 = LineString([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])

    region = process(label_img.astype(np.uint8))

    if len(region)>1:
        region.sort(key=len,reverse=True)
        cache=[]
        for subregion in region:
            shape = dict()
            points = []
            skipsignal = []
            for i in range(0, subregion.shape[0]):

                pt = list(map(add, subregion[i][0].tolist(), [x1, y1]))
                points.append(pt)
                for x, y in [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]:
                    if math.hypot(pt[0] - x, pt[1] - y) < 5:
                        skipsignal.append(i)
                        if len(skipsignal) == 2:
                            if skipsignal[0] + 1 == i:
                                points.pop()
                                points.pop()
                                skipsignal = []
                        break

            p=Polygon(points)
            cache.append(p)

            shape['flags'] = {}
            shape['group'] = groupid
            shape['group_id'] = None

            if len(cache) > 1:

                if cache[0].intersects(cache[1]) == True:
                    shape['label'] = labels[0]

                else:
                    shape['label'] = labels[1]
                cache.pop()
            else:
                shape['label'] = labels[1]

            shape['model'] = model
            shape['points'] = points
            shape['rectpts'] = rectpts
            shape['shape_type'] = 'polygon'

            shapes.append(shape)


    else:
        for subregion in region:
            points = []
            skipsignal = []
            for i in range(0, subregion.shape[0]):

                pt = list(map(add,subregion[i][0].tolist(),[x1,y1]))
                points.append(pt)
                for x, y in [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]:
                    if math.hypot(pt[0]-x,pt[1]-y)<5:
                        skipsignal.append(i)
                        if len(skipsignal) == 2:
                            if skipsignal[0]+1 == i:
                                points.pop()
                                points.pop()
                        break


            shape = dict()
            shape['flags'] = {}
            shape['group'] = groupid
            shape['group_id'] = None
            shape['label'] = labels[1]
            shape['model'] = model
            shape['points'] = points
            shape['rectpts'] = rectpts
            shape['shape_type'] = 'polygon'

            shapes.append(shape)

    saveJsonPath = ImgPath + os.sep + imgname[:-4] + '.json'

    with open(saveJsonPath, 'r') as f:
        obj = json.load(f)
        chshapes = []
        changepts = []
        remove = []
        shapesave = copy.deepcopy(obj['shapes'])
        for orishape in obj['shapes']:

            if orishape['group'] != groupid:

                continue


            l1 = LineString(orishape['points'])
            p1 = Polygon(orishape['points'])
            if p0.contains(p1):

                remove.append(obj['shapes'].index(orishape))
            elif l0.intersects(l1):

                chshapes.append(orishape)
                #p0p1 = p0.intersection(p1)
                changepts = []
                changept = []
                for idx in range(len(orishape['points']) - 1):

                    A = orishape['points'][idx]
                    B = orishape['points'][idx + 1]
                    line1 = LineString([A, B])

                    if line1.intersects(l0) == True:
                        int_pt = line1.intersection(l0)

                        if int_pt.geom_type == 'Point':

                            gate1 = p0.contains(Point(A))
                            gate2 = p0.contains(Point(B))
                            if gate1 == True and gate2 == False:
                                changept.append(idx + 1)
                                changepts.append(changept)
                                changept = []
                            elif gate1 == False and gate2 == True:
                                changept.append(idx + 1)

                    if idx == len(orishape['points']) - 2 and changept != []:
                        changepts.append(changept)

                remove.append(obj['shapes'].index(orishape))

        obj['shapes'] = [obj['shapes'][idx] for idx in range(len(obj['shapes'])) if idx not in remove]

        if changepts != []:
            if len(changepts[0]) == 1:
                changepts[0]=[changepts[-1][0],changepts[0][0]]
                changepts.pop()
        l0 = l0.buffer(10)

        for shape in shapes:

            p = LineString(shape['points'])

            if  p.contains(p0):
                obj['shapes'].append(shape)
            elif p.intersects(l0):
                if changepts == []:
                    obj['shapes'].append(shape)
                    continue
                intershape = l0.intersection(p)

                if intershape.geom_type == 'MultiLineString':
                    intersectpts = list(intershape.geoms)
                    bdpts = []
                    for i in intersectpts:
                        for j in list(i.coords):
                            bdpts.append(j)

                elif intershape.geom_type == 'Polygon':
                    bdpts = list(intershape.exterior.coords)
                elif intershape.geom_type == 'LineString':
                    bdpts = list(intershape.coords)


                bdpts = [list(elem) for elem in bdpts if  list(elem) in shape['points']]

                for chshape in chshapes:
                    shapecopy = copy.deepcopy(chshape['points'])

                    for pt1, pt2 in changepts:

                        try:
                            line = LineString([chshape['points'][pt1+1],chshape['points'][pt1-1]])
                            linexl0 = line.intersection(l0).coords[0]
                            dis = [math.hypot(linexl0[0]-bpt[0], linexl0[1]-bpt[1]) for bpt in bdpts]
                            stpt = shape['points'].index(bdpts[dis.index(min(dis))])
                            line = LineString([chshape['points'][pt2+1], chshape['points'][pt2-1]])
                            linexl0 = line.intersection(l0).coords[0]
                            dis = [math.hypot(linexl0[0] - bpt[0], linexl0[1] - bpt[1]) for bpt in bdpts]
                            edpt = shape['points'].index(bdpts[dis.index(min(dis))])
                            pt1 = shapecopy.index(chshape['points'][pt1])
                            pt2 = shapecopy.index(chshape['points'][pt2])
                            if edpt<stpt:
                                if pt1>pt2:

                                    shapecopy[:pt2] = shape['points'][stpt:]+shape['points'][:edpt]
                                    shapecopy[shapecopy.index(chshape['points'][pt1]):] = []
                                else:
                                    shapecopy[pt1:pt2] = shape['points'][stpt:]+shape['points'][:edpt]
                            else:
                                if pt1>pt2:
                                    shapecopy[:pt2] = shape['points'][stpt:edpt+1]
                                    shapecopy[shapecopy.index(chshape['points'][pt1]):] = []
                                else:
                                    shapecopy[pt1:pt2] = shape['points'][stpt:edpt+1]
                        except:
                            continue

                    chshape['points']=shapecopy
                    obj['shapes'].append(chshape)

            else:

                obj['shapes'].append(shape)


    j = json.dumps(obj, sort_keys=True, indent=4)


    with open(saveJsonPath, 'w') as f:
        f.write(j)



    return shapesave


def findcorrespondingrect(filename, x1, y1, x2, y2):

    (ImgPath, imgname) = os.path.split(filename)
    saveJsonPath = ImgPath + os.sep + imgname[:-4] + '.json'
    b0 = box(x1, y1, x2, y2)
    maxarea = 0
    rectpts = None
    groupid = None
    modeltype = None
    matting = None

    with open(saveJsonPath, 'r') as f:
        obj = json.load(f)
        for shape in obj['shapes']:
            if shape['model'] == 'vitae':
                b1 = box(shape['rectpts'][0][0],shape['rectpts'][0][1],shape['rectpts'][1][0],shape['rectpts'][1][1])

            elif shape['model'] == 'matteformer' and 'bound' in shape:
                b1 = box(shape['bound'][0],shape['bound'][1],shape['bound'][2],shape['bound'][3])

            else:
                continue

            if b1.intersection(b0).area > maxarea:
                rectpts = shape['rectpts']
                groupid = shape['group']
                modeltype = shape['model']
                if modeltype == 'matteformer':
                    matting = np.array(shape['matting'])

    return rectpts, groupid, modeltype, matting

