import os, sys, json, copy, random
import pandas as pd
import numpy as np
from fastai.vision import *
from fastai.utils.mem import  gpu_mem_get_free_no_cache

# gpu_mem_get_free_no_cache()

### setup data building environment -------------------------------------

data_dir = Path('data/alphapilot/')
raw_fn = data_dir/'data_training'
label_fn = data_dir/'truth_df.csv'

if os.name == 'nt':      ##local
    # data_dir = Path('../../../../alphapilot/')
    data_dir = Path('c:/users/wsutt/desktop/files/alphapilot/')  # absolute
    raw_fn = data_dir/'Data_Training/Data_Training/'
    label_fn = data_dir/'truth_df.csv'

#load ground truth
truth_df = pd.read_csv(label_fn, index_col=0)
assert truth_df.shape == (8, 5827)

TRUTH_INDS  = list(truth_df.columns)


#utility functions -----
def filter_img_by_truth(fn):
    ''' only use data records in truth_df'''
    return fn.name in TRUTH_INDS

def filter_mini(fn):
    ''' only use data records in truth_df'''
    return fn.name in TRUTH_INDS and 'IMG_00' in fn.name

filter_records = filter_img_by_truth

def label_points(fn):
    '''
        input:  x0,y0,...x3,y3 (list)
        output: [y0,x0],...[y3,x3] (list) 
         
        >use y_first=True in label-load-func
    '''
    p = truth_df[fn.name]
    return tensor([ [float(p[i*2+1]), float(p[i*2+0])] for i in range(4)])


def get_truth_df():
    ''' utility for jupyter user to access the base ground truth '''
    return truth_df


### Main notebook helper functions --------------------------------------------------

def avg_prediction(dataset, as_int=False, as_flat=False):
    '''
        input: ImageDataBunch
        returns mean of each y-point in the training-set 
                as float 
                    (as_int: as int)
                as list if list: [[x0,y0],...[yx3,y3]]
                    (as_flat: [x0,y0,...,x3,y3]), 
        notes:
            y.items are in tensor(4by2 floats)
            we maintain coord order from input-dataset; 
            expecting & returning x-first(?)
    '''

    y_coords =  [_item.tolist() for _item in dataset.train_dl.y.items]

    y_flat =    [[item for sublist in _box for item in sublist]
                  for _box in y_coords]

    sum_y_flat = [sum( [_row[_col] for _row in y_flat] )
                  for _col, _ in enumerate(y_flat[0])
                 ]
    
    n =         len(y_coords)

    avg_y_flat = [x / n for x in sum_y_flat]

    if as_int:
        avg_y_flat = [int(x) for x in avg_y_flat]

    if not(as_flat):
        avg_y_flat = [  [avg_y_flat[_icoord*2+0], avg_y_flat[_icoord*2+1]]
                        for _icoord in range(len(avg_y_flat) // 2)
                     ]

    return avg_y_flat



def build_data(
                batch_size = None,
                size = None,
                num_workers = None,
                bypass_validation = True,
                seed = None,
                mini_data = False,
                ):

    _numworkers = {}
    if os.name == 'nt':
        _numworkers['num_workers'] = 0
    if num_workers is not None:
        _numworkers['num_workers'] = num_workers
        
    _batchsize = 4
    if batch_size is not None:
        _batchsize = batch_size

    _size = (216, 324)
    if size is not None:
        _size = size

    _seed = 42
    if seed is not None:
        _seed = seed

    np.random.seed(_seed)  # called each time you eneter function

    filter_records = filter_img_by_truth
    if mini_data:
        filter_records = filter_mini

    data = (PointsItemList.from_folder(raw_fn)
            .filter_by_func(filter_records)
            .split_by_rand_pct(valid_pct=0.2)
            .label_from_func(label_points)
            .transform(get_transforms()
                                ,tfm_y=True
                                ,size=_size
                                ,remove_out=False
                            )    
            .databunch(bs=_batchsize, **_numworkers)
            .normalize(imagenet_stats)
        )

    if bypass_validation:
        return data
    
    assert isinstance(data, ImageDataBunch)
    assert len(data.train_dl.x.items) == 4662
    assert len(data.valid_dl.x.items) == 1165
    assert list(data.valid_dl.y.items[0].shape) == [4,2]
    assert list(data.valid_dl.x.get(0).shape) == [3, 864, 1296]
    assert data.num_workers == (0 if os.name == 'nt' else 8)
    assert str(data.path) == ( 'data/alphapilot/data_training'
                            if os.name != 'nt' else
                            '..\\..\\..\\..\\alphapilot\\Data_Training\\Data_Training'
                                )

    return data