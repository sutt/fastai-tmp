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


class MultiIp:

    ''' for displaying multiple sets of ImagePoints 

        univ_p = {'marker':'o', 's':100}

        mip = MultiIp(list_ips=[ip1, ip2], 
                      list_params=[{'c':'y', **univ_p, },
                                   {'c':'r', **univ_p, }
                                  ]
                    )
    
    '''

    def __init__(self, list_ips, list_params=[]):
        self.ips = list_ips
        self.params = list_params
    
    def show(self, ax, **kwargs):
        kwargs_save = kwargs.copy()
        for _ip, _param in zip(self.ips, self.params):
            kwargs = {**_param, **kwargs_save}
            _ip.show(ax, **kwargs)


def pred_compare(datasets, learners, size=(216, 324), i=None, b_print=False
                ,b_one_image=True):

    ''' 
        visually compare perf of different learners on same scoring image

    '''
    
    if i is None:
        i = np.random.randint(0, len(datasets[0].valid_dl.x.items) - 1)

    fn = datasets[0].valid_dl.x.items[i]
    for _dataset in datasets:
        assert fn == _dataset.valid_dl.x.items[i]  # you're comparing diff imgs; stop
    
    img = datasets[0].valid_dl.x.get(i)
    
    pred_pts, truth_pts = [], []
    for _dataset, _learner in zip(datasets, learners):
        pred_pts.append( _learner.predict(img)  )  #outputs tuple(ImgPts obj, data)
        truth_pts = _dataset.valid_dl.y.items[i]
    
    scaled_pred_pts = [scaled_pts(_pts[0]) for _pts in pred_pts]
    
    flip_pts = [tensor([[e[1],e[0]] for e in list(_pts)])
                for _pts in scaled_pred_pts]

    img2 = img.clone()
    img2.resize(size=(3,*size))
    

    if b_print:
        pass
        #TODO - error amount
        #TODO - is get_ip(img or img2, pts)?
        # y=get_ip(img, truth_pts)
        #   y=get_ip(img2, scaled_pred_pts)


    if b_one_image:
        
        univ_p = {'marker':'o', 's':100}

        mip = MultiIp(list_ips=[get_ip(img2, _pt) for _pt in flip_pts], 
                      list_params=[{'c':'y', **univ_p, },
                                   {'c':'r', **univ_p, }
                                  ]
                    )

        img2.show( 
                 y=mip
                ,title=(str(i) + ' - ' + fn.name)
                ,figsize=(10, 10)
                )

    else:

        for _pts in flip_pts:

            img2.show( 
                    y=get_ip(img2, _pts)
                    ,title = str(i) + ' - ' + fn.name
                    ,figsize=(10, 10)
                    ,c='y' ,marker='o', s=100
                    )