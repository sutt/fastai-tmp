import os, sys, json, copy, random, time, pickle, datetime, dateutil
import pandas as pd
import numpy as np
import fastai
import torch
from fastai.vision import *
from fastai.utils.mem import  gpu_mem_get_free_no_cache
from .miscutils import fastai_version, is_local

loss_each_ = torch.nn.MSELoss(reduction='none')
loss_all_ = torch.nn.MSELoss(reduction='sum')

def calc_sse(pred, target):
    '''return sse from output of get_preds(); pred=output[0], target=output[1] '''
    syn_err = [loss_all_(_pred, _target.flatten()) 
                    for _pred, _target in zip(pred, target) ]
    return sum(syn_err)

def calc_mse(pred, target, n_pts=8):
    '''return mse from output of get_preds(); pred=output[0], target=output[1] '''
    sse = calc_sse(pred, target)
    return sse / float(len(pred) * n_pts)

def err_to_np(errs, n_pts=8):
    ''' return the errors in postion [2] of output from get_preds() 
        as an n x p np ndarray
    '''
    errs = errs.clone()
    arr = np.array(errs.tolist())
    
    n = int(len(arr) / n_pts)
    tbl = arr.reshape((n, n_pts))

    return tbl

class ModelHome:

    ''' for re-intializing already trained models '''

    def __init__(self, 
                model_arch,
                dataset,
                pth_fn,
                name=None,
                history_fn=None,
                preds_fn=None,
                ):

        self.name = name
        self.dataset = dataset
        self.model = None
        self.history = None
        self.preds = None
        self.avg_pred_time = None
        self.valid_err = None
        self.y_first = None

        self._init(model_arch, dataset, pth_fn, history_fn, preds_fn)

    def _init(self, model_arch, dataset, pth_fn, history_fn, preds_fn):
        ''' the workhouse of class init '''
        
        self.model = cnn_learner(data=dataset, base_arch=model_arch)

        model_dir, model_fn = os.path.split(pth_fn)
        # self.model.model_dir = os.path.relpath(model_dir, start = self.model.path)
        self.model.model_dir = os.path.abspath(model_dir)
        self.model.load(model_fn.split('.')[0])

        if history_fn is not None:
            self.history = pd.read_csv(history_fn)

        if preds_fn is not None:
            with open(preds_fn, 'rb') as fn:
                self.preds = pickle.load(fn)

        if self.preds is not None:
            self.valid_err = calc_mse(self.preds[0], self.preds[1])
        elif self.history is not None:
            self.valid_err = list(self.history['valid_loss'])[-1]

        

    def get_prediction(self, i, b_train=False):
        ''' return prediction for item i in validation-set; or from training-set if
            b_train is True. 
            prediction as flat tensor flow array
        '''
        if not(b_train):
            if self.preds is not None:
                return self.preds[0][i]
        else:
            return self.model.predict(self.dataset.train_dl.get(i))[2]

    def get_truth(self, i, b_train=False):
        if not(b_train):
            if self.preds is not None:
                return self.preds[1][i]
        else:
            #TODO - still wrong
            return self.dataset.train_dl.y.items[0]

    def get_split(self, b_train=False):
        ''' using a command to switch b/w train_dl and valid_dl; default=valid'''
        if b_train:
            return self.dataset.train_dl
        return self.dataset.valid_dl

    def build_preds(self, dest_dir='misc-data', pickle_fn=None):
        ''' run get_preds on validation set; save to pickle '''
        preds = self.model.get_preds(with_loss=True)
        self.preds = preds.copy()

        fn = 'preds_'
        if self.name is not None:
            fn += self.name
        else:
            now = datetime.datetime.now()
            fn+=  '_'.join( [str(e) for e in 
                             [now.month(), now.day(), now.hour(), now.minute()]
                             ])
        fn += '.pickle'
        pickle_fn = os.path.join(dest_dir, fn)

        try:
            with open(pickle_fn, 'wb') as fn:
                pickle.dump(preds, fn)
            print('success - output preds to: %s' % pickle_fn)
        except:
            print('failed - to pickle preds to %s' % pickle_fn)


    def build_avg_pred_time(self, n=5):
        ''' perform a prediction n times, take avg of time elpased'''        
        time_tmp = []
        
        for _i in range(n):
            _record = self.dataset.valid_dl.get(_i)
            t0 = time.time()
            _ = self.model.predict(_record)
            time_tmp.append(time.time() - t0)
        
        self.avg_pred_time = sum(time_tmp) / float(len(time_tmp))

        ret = [ int(round(e, 3) * 1000) for e in
                    [self.avg_pred_time, min(time_tmp), max(time_tmp)]
              ]

        print('avg pred time (ms): %i  [min: %i | max: %i]' % 
                (ret[0],ret[1],ret[2])
                )


class Residuals:

    def __init__(self, errs):
        # assert isintance()
        self.record_err = None
        self.pt_err = None

    def foo():
        pass

class Predictions:

    ''' transforms for/ be able to query preds, actuals, and errs on-demand'''
    
    def __init__(self):
        pass



        