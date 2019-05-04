import os, sys, json, copy, random, time, pickle, datetime, dateutil
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
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

def get_ip(img,pts): 
    ''' converts plain tensor representing points to an 
        ImagePoints obj; useful for inserting into show(y=?)
    '''
    return ImagePoints(FlowField(img.size, pts), 
                       scale=True,  
                       y_first=True)

class ModelHome:

    ''' for re-intializing already trained models 
        TODO
        [ ] load training preds
        [ ] facts about history, e.g. num of epochs
            this can include non-history facts: e.g. training_time, epoch_time

    
    '''

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

        self.residuals = None
        self.predictions = None
        self.groundtruth = None

        self.d_facts = {}

        self._init(model_arch, dataset, pth_fn, history_fn, preds_fn)

    def _init(self, model_arch, dataset, pth_fn, history_fn, preds_fn):
        ''' the workhouse of class init '''
        
        self.model = cnn_learner(data=dataset, base_arch=model_arch)

        model_dir, model_fn = os.path.split(pth_fn)
        self.model.model_dir = os.path.abspath(model_dir)
        self.model.load(model_fn.replace('.pth', ''))

        if history_fn is not None:
            self.history = pd.read_csv(history_fn)

        if preds_fn is not None:
            self.load_preds(preds_fn, split_type='valid')

        if self.preds is not None:
            self.valid_err = calc_mse(self.preds[0], self.preds[1])
        elif self.history is not None:
            self.valid_err = list(self.history['valid_loss'])[-1]

    def __repr__(self):

        def ns(x):
            if x is None: return 'None'
            return str(x)
        def nn(x):
            if x is None: return 'false'
            return 'true'
        
        s = '''
        %s
        valid_err:          %s
        training_err:       %s
        avg predict time:   %s
        histroy loaded:     %s
        (validation) preds loaded: %s
        (training) preds loaded:   %s ''' % (
        
                ns(self.name),
                ns(self.valid_err),
                'not implemented',
                ns(self.avg_pred_time),
                nn(self.history),
                nn(self.preds),
                'not implemented',
        )
        return str(s).strip()
                
    def load_preds(self, preds_fn, split_type='valid'):
        ''' load the pickled output from model.get_preds into cached preds'''
        
        with open(preds_fn, 'rb') as fn:
            preds = pickle.load(fn)
        
        assert len(preds) == 3, 'preds_tbl not 3-wide; run with `with_loss=True`'
        assert len(preds[0]) == len(self.dataset.valid_dl.items)
        
        self.preds = preds


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
        ''' return truth for item i in validation-set; or from training-set if
            b_train is True.  
            truth as 4 by 2 flow tensor
        '''
        if not(b_train):
            if self.preds is not None:
                return self.preds[1][i]
        else:
            demo_img = self.dataset.train_dl.x.get(0)
            return get_ip(demo_img, self.dataset.train_dl.y.items[i]).data

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

    def __init__(self, errs, n_pts=8):
        ''' errs - tensor of each coord loss len=(n * n_pts)
                   loss values from coords as flow
        '''
        self.flow = errs
        self.flow_np = err_to_np(errs, n_pts=n_pts)

        self.n = int(len(errs) / n_pts)
        self.mse = sum(errs) / (self.n * n_pts)
        

    def plot_losses(ax=None, per_coord=False):
        ''' historgram of losses '''
        if ax is None: fig,ax = plt.subplots(figsize=figsize)
        ax.hist(self.get_loss())
        return ax

    def get_loss(self, i=None, per_coord=False, per_point=False, as_np=True):
        ''' return losses, as np array by default, by total for all coords '''
        if per_coord:
            if i is None:
                return self.flow_np.copy()
            else:
                return self.flow_np[i,:]
        elif per_point:
            pass # calc cartesian loss
        else:
            if i is None:
                return self.flow_np.sum(axis=1).copy()
            else:
                return sum(self.flow_np[i,:])


class GroundTruth:

    ''' do transforms and operations on truth table'''

    def __init__(self, y_flow, verify_order=False):
        ''' y_flow - output of get_preds[1], tensor of (n, 4, 2) with y_first'''
        self.y_flow = y_flow

        self.order = self.get_order(self.y_flow[0])  

        if verify_order:
            try:
                assert self.order_verify(), 'order verified returned False'
            except:
                pass

    @staticmethod
    def get_order(truth_vec, y_first=True):
        ''' use y_flow[0] to deduce order
            convetion: top left (x1,y1), clockwise around
            assumes in true four corners that there are two lo's and two hi's
            for both x and y...hard to explain here. 
        '''

        y_vec, x_vec = truth_vec[:,0], truth_vec[:,1]        

        y_sort = sorted(enumerate(y_vec), key=lambda e:e[1])
        y_sort = [e[0] for e in y_sort]
        y_lo, y_hi = y_sort[:2], y_sort[2:]
        
        x_sort = sorted(enumerate(x_vec), key=lambda e:e[1])
        x_sort = [e[0] for e in x_sort]
        x_lo, x_hi = x_sort[:2], x_sort[2:]
        
        tl = [e for e in range(4) if e in x_lo and e in y_lo][0]
        tr = [e for e in range(4) if e in x_hi and e in y_lo][0]
        br = [e for e in range(4) if e in x_hi and e in y_hi][0]
        bl = [e for e in range(4) if e in x_lo and e in y_hi][0]
        
        order = (tl, tr, br, bl)

        return order
    
    def verify_order(self, ret_all=False):
        ''' verify points position follows the same ordering as 
            the first truth_vec (index=0) that was used to set self.order
            if ret_all=True, return a list of all mis-ordered elements
         '''
        tmp = []
        
        for _i, _v in enumerate(self.y_flow):
        
            _order = self.get_order(_v)
            
            if _order != self.order:
                
                if (ret_all): 
                    
                    tmp.append((_i,_v,_order))
                    continue
                
                else:

                    print('order at ind %i is %s not %s \n y_flow:\n %s' % 
                        (_i, str(_order), str(self.order), str(_v)) 
                        )
                    return False
        
        if (ret_all): return tmp
        else: return True

    
    def get_interior_size(self):
        ''' calculate rough area of ground truth'''
        for _record in self.y_flow:
            
            v_len = _record / 2

    def get_perspective_ratio(self, b_vertical=True):
        ''' compare the two distances; return the ration of small/large'''
        
        coord_j = ([['y1', 'y4'],['y2', 'y3']] if b_vertical else 
                   [['x1', 'x4'],['x2', 'x3']])

        for _i, _yflow in enumerate(self.y_flow):

            dsts = [
                    abs(_yflow[self.order(coord_j[0][0])] -
                        _yflow[self.order(coord_j[0][1])]
                    ),
                    abs(_yflow[self.order(coord_j[1][0])] -
                        _yflow[self.order(coord_j[1][1])]
                    ),
                   ]

        return min(dsts) / max(dsts)

class Predictions:

    ''' transforms for/ be able to query preds, actuals, and errs on-demand'''
    
    def __init__(self):
        pass

class ModelCmp:

    def __init__(self, list_mh):
        self.list_mh = list_mh

    def cmp_err(self):
        '''return a df with each mh name and err + supplementary info '''
        pass
        # {_mh}
    
    def plot_losses(self):
        ''' for each mh, plot a two hists of the residuals:
                - a log hist of full amount of losses
                - a regular hist of non-outliers
        '''
        num_mh = len(list_mh)

        fig,ax = plt.subplots(num_mh, 2, figsize=figsize)
        
        for _mh in self.list_mh:

            _mh.residuals.plot_losses(ax=ax)

        return ax




        