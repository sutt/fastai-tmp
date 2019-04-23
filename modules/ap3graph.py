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
        see example notebook here: TODO
    '''

    def __init__(self, list_ips, list_params=[], labels=None, legend=None, **kwargs):
        ''' required input: list_ips - list of ImagePoint object(s)
            optional input: list_params - list of dict(s);        len=len(list_ips)
                            legend - true, false, or list of str  len=len(list_ip))
                            labels - true, false, or list of str  len=list_ips[0].shape[0]
                            label_offset - int                    default=3
                            label_enumerate - bool                default=false
        '''
        self.ips =    list_ips
        self.params = list_params
        
        # if labels/legend is False or not specified, feature is off
        # if its passed True, feature is on with default titling strategy
        # if its passed a list of string, feature is on with that list's titling strategy
        self.b_legend =   legend if isinstance(legend, bool) else (legend is not None)
        self.d_legend =   None if isinstance(legend, bool) else legend
        self.b_labels =   labels if isinstance(labels, bool) else (labels is not None)
        self.d_labels =   None if isinstance(labels, bool) else labels
        
        # extra args
        self.label_offset =     kwargs.get('label_offset', 3)
        self.label_enumerate =  kwargs.get('label_enumerate', False)
        
    def show(self, ax, **kwargs):
    # show(self, ax:plt.Axes=None, figsize:tuple=(3,3), title:Optional[str]=None, hide_axis:bool=True, **kwargs):
        ''' when this class is passed as y, this method get called inside Image.show'''

        kwargs_save = kwargs.copy()

        for _i, (_ip, _param) in enumerate(zip(self.ips, self.params)):
            
            unique_kwargs = {**_param, **kwargs_save}
            _ip.show(ax, **unique_kwargs)   #TODO, pass in args like figsize to this call to ImagePoints.show

            if self.b_labels:
                self._label(ax, _ip, set_index=_i)

        if self.b_legend:
            if self.d_legend is None:
                legend = [str(x) for x in range(len(self.ips))]
            else:
                legend = self.d_legend
            ax.legend(legend)

    def _label(self, ax, ips, set_index='?'):
        '''apply a label to each point, based on params passed-in during init'''
        
        # same proc as ImagePoints.show() + offset so labels are centered in marker
        pnts = scale_flow(FlowField(ips.size, ips.data), to_unit=False).flow.flip(1)
        pnts[:,0] = pnts[:,0].sub_(self.label_offset)
        pnts[:,1] = pnts[:,1].add_(self.label_offset)
        pnts = pnts.tolist()
        
        for _ptindex, _pt in enumerate(pnts):
            
            # decide the labelling strategy
            if self.d_labels is not None:
                point_label = self.d_labels[_ptindex]
            elif self.label_enumerate:
                point_label = _ptindex    
            else:
                point_label = set_index
            point_label = str(point_label)
            
            ax.annotate(point_label, _pt)

    

def pred_compare(datasets, learners, size=(216, 324), i=None, b_print=False
                ,b_one_image=True, labels=False, legend=False):

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
                                  ],
                      labels=labels,
                      legend=legend,
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