import os, sys, json, copy, random
import pandas as pd
import numpy as np
from fastai.vision import *
from fastai.utils.mem import  gpu_mem_get_free_no_cache


def scaled_pts(pts):
    ''' use a helper function to convert coords flow->absolute
        input  pts - an ImagePoints object
        output tensor(4,2)
    '''
    return scale_flow(
                FlowField(pts.size, pts.data)
                            ,to_unit=False
                        ).flow.flip(1)

def scaled_pts_2(data_slice, item_num):
    ''' use a secondary pattern to convert coords flow->absolute:
        input data_slice=data_final.valid_dl
              item_num=0
        output tensor(4,2)
    '''
    pll = PointsLabelList([])
    re_pts = pll.reconstruct( data_slice.y.items[item_num]
                             ,data_slice.x.get(item_num))
    return re_pts.data

def get_ip(img,pts): 
    ''' converts plain tensor representing points to an 
        ImagePoints obj; useful for inserting into show(y=?)
    '''
    return ImagePoints(FlowField(img.size, pts), 
                       scale=True,  
                       y_first=True)

def clean_preds(pred_pts):
    '''remove out-of-bounds from coords'''
    pass


def pred_compare(data, learn, size=(216, 324), i=None, b_print=False):
    
    if i is None:
        i = np.random.randint(0, len(data.valid_dl.x.items) - 1)

    fn =  data.valid_dl.x.items[i]
    img = data.valid_dl.x.get(i)
    truth_pts = data.valid_dl.y.items[i]

    pred_pts = learn.predict(img)  #outputs tuple(ImgPts obj, data)
    # pred_pts2 = learn2.predict(img)  #outputs tuple(ImgPts obj, data)
    scaled_pred_pts = scaled_pts(pred_pts[0])
    # scaled_pred_pts2 = scaled_pts(pred_pts2[0])

    flip_pts = tensor([[e[1],e[0]] for e in list(scaled_pred_pts)])

    img2 = img.clone()
    img2.resize(size=(3,*size))

    if b_print:
        print(i, fn.name)
        # print(truth_pts); print('----')
        # print(scaled_pred_pts) #TODO - put on same terms as ground-truth, round
        # print(flip_pts)

    #TODO - is get_ip(img or img2, pts)?

    img2.show( 
              # y=get_ip(img, truth_pts)
            #   y=get_ip(img2, scaled_pred_pts)
              y=get_ip(img2, flip_pts)
             ,figsize=(10, 10)
             ,c='y' ,marker='o', s=100
            )

    # img2.show( 
    #           # y=get_ip(img, truth_pts)
    #           y=get_ip(img2, scaled_pred_pts2)
    #          ,figsize=(10, 10)
    #          ,c='y' ,marker='o', s=100
    #         )