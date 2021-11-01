#%%
import argparse
import datetime
import os
import copy

import numpy as np
import torch
import torch.utils.data as Data
from munch import Munch
from numpy.lib.shape_base import expand_dims
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.types import Number
from torch.utils.data.dataset import T
from torchsummaryX import summary
from tqdm import tqdm

from perspect_classInit3 import *

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=str, default='2007', help='PDBbinding dataset v(2007, 2013, 2016)')
parser.add_argument('--model', type=str, default='Resnet', help='Deep backbones to extract features from Mol-PSIs.\
                    Densenet, Resnet for 2D/1D Mol-PSIs, (Image) Transformer for 1D Mol-PSIs')
parser.add_argument('--sigma', type=float, default=None, help='Dimensionality of scale parameter sigma (0.0001, 0.75, 1.5, 2.25)')
parser.add_argument('--datatype', type=str, default='es0', help='ES-IDM 1D (es0), ES-IEM 1D (elec0), ES-IDM 2D (es) and ES-IEM 2D (elec)')

parser.add_argument('--num_epoch', type=int, default=1001, help='number of training epoch')
parser.add_argument('--batch_size', type=int, default=1001, help='number of data points for each itration')
parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=37, help='Random seed.')
parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
parser.add_argument('--transfer_model', type=bool, default=False, help='Training from the transfer model')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--num_repeat', type=int, default=5, help='Number of tries for the same experiments')

parser.add_argument('--valid_model', type=bool, default=True, help='Training from the transfer model')
parser.add_argument('--valid_method', type=str, default='bins_pred', help='Stratified the train and valid subsets by random/bins/bins_pred/exist_model')
parser.add_argument('--valid_path', type=str, default='', help='load valid model state dict when transfering on exist model')
parser.add_argument('--valid_size', type=int, default=0.2, help='Number of training dataset used for transfer')
parser.add_argument('--valid_seed', type=int, default=13, help='Random seed for validation dataset selection')
parser.add_argument('--valid_num_epoch', type=int, default=51, help='number of transfer epoch')
parser.add_argument('--valid_batch_size', type=int, default=256, help='number of data points for each transfering itration')
parser.add_argument('--valid_lr', type=float, default=5e-4, help='Initial transfer learning rate.')


# args = parser.parse_args()
args = parser.parse_known_args()[0]
print(args)
args.cuda = args.cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.model == 'trans' and (args.datatype == 'es0' or args.datatype == 'elec0'):
    raise ValueError('Image Transformer is not applicable for 2D Mol-PSIs')
if args.sigma and (args.datatype == 'es0' or args.datatype == 'elec0'):
    raise ValueError('1D Mol-PSIs is not applicable with sigmas')
if not args.sigma and (args.datatype != 'es0' and args.datatype != 'elec0'):
    raise ValueError('2D Mol-PSIs requires a specified sigma')

###
now = datetime.datetime.now()
date = now.strftime("%B%d")
log_dir = f'log_{date}_30bins_2007tries'
if not os.path.exists(f"log_{date}_30bins_2007tries/best_models/"):
    os.makedirs(f"log_{date}_30bins_2007tries/best_models/")
if not os.path.exists(f"log_{date}_30bins_2007tries/valid_models/"):
    os.makedirs(f"log_{date}_30bins_2007tries/valid_models/")
# if not os.path.exists(f"log_{date}/cls_models/"):
#     os.makedirs(f"log_{date}/cls_models/")

#%%
def transfer_model(method, t, load_tranfer=True):
    if t == 'es' and method == 'Densenet':
        model = Dense_back(36)
        if load_tranfer:
            model.load_state_dict(torch.load("./transfer_model/es_dense.pt"))
    elif t == 'es' and method == 'Resnet':
        model = Res_back(36)
        if load_tranfer:       
            model.load_state_dict(torch.load("./transfer_model/es_res.pt"))        
    elif t == 'es' and method == 'Transformer':
        model = ImageTransformer(channels=36)
        if load_tranfer:
            model.load_state_dict(torch.load("./transfer_model/es_trans.pt")) 
        
    elif t == 'elec' and method == 'Densenet':
        model = Dense_back(50)
        if load_tranfer:
            model.load_state_dict(torch.load("./transfer_model/elec_dense.pt"))  
    elif t == 'elec' and method == 'Resnet':
        model = Res_back(50)
        if load_tranfer:
            model.load_state_dict(torch.load("./transfer_model/elec_res.pt"))  
    elif t == 'elec' and method == 'Transformer':
        model = ImageTransformer(channels=50)
        if load_tranfer:
            model.load_state_dict(torch.load("./transfer_model/elec_trans.pt"))  
        
    elif t == 'es0' and method == 'Densenet':
        model = Dense_back(1,3072)
        if load_tranfer:
            model.load_state_dict(torch.load("./transfer_model/es0_dense.pt"))  
    elif t == 'es0' and method == 'Resnet':
        model = Res_back(1)
        if load_tranfer:
            model.load_state_dict(torch.load("./transfer_model/es0_res.pt"))
    
    elif t == 'elec0' and method == 'Densenet':
        model = Dense_back(1,3072)
        if load_tranfer:
            model.load_state_dict(torch.load("./transfer_model/elec0_dense.pt"))  
    elif t == 'elec0' and method == 'Resnet':
        model = Res_back(1)
        if load_tranfer:
            model.load_state_dict(torch.load("./transfer_model/elec0_res.pt")) 
    else:
        raise Exception("No such a model")
    
    model = model.cuda() if args.cuda else model
    print(f"load model {method} {t} load_tranfer_paras is {load_tranfer}")
    return model

def valid_model(method, t, sigma, log_dir):
    if t == 'es' and method == 'Densenet':
        model = Dense_back(36)

    elif t == 'es' and method == 'Resnet':
        model = Res_back(36)
    
    elif t == 'es' and method == 'Transformer':
        model = ImageTransformer(channels=36)

    elif t == 'elec' and method == 'Densenet':
        model = Dense_back(50)

    elif t == 'elec' and method == 'Resnet':
        model = Res_back(50)

    elif t == 'elec' and method == 'Transformer':
        model = ImageTransformer(channels=50)

        
    elif t == 'es0' and method == 'Densenet':
        model = Dense_back(1,3072)

    elif t == 'es0' and method == 'Resnet':
        model = Res_back(1)

    
    elif t == 'elec0' and method == 'Densenet':
        model = Dense_back(1,3072)

    elif t == 'elec0' and method == 'Resnet':
        model = Res_back(1)

    else:
        raise Exception("No such a model")
    
    model = model.cuda() if args.cuda else model
    print(f"load initial model {method} {t}")
    
    print(f"Transfer on (train_ = train + validation) {2016} {t} {str(sigma) if sigma else ''}:")
    X_train_, X_test, y_train_, y_test = tensors_by_year_type('2016', sigma, t)
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=args.valid_size, random_state=args.valid_seed)

    n_splits = int(1/args.valid_size)
    # bins = np.linspace(2, 12, 11, endpoint=True)
    # bins = np.linspace(2, 12, 31, endpoint=True)
    bins = np.linspace(2, 12, 51, endpoint=True)
    # bins = np.linspace(2, 12, 101, endpoint=True)
    
    if args.valid_method == 'exist_model':
        model.load_state_dict(torch.load(f'{args.valid_path}'))
        print(f"load exist model {args.valid_path} successfully!")
        return model
        
    if args.valid_method == 'bins' or args.valid_method == 'random':
        if args.valid_method == 'bins':
            y_train_ind = np.digitize(y_train_, bins)
            split_folder = StratifiedKFold(n_splits=n_splits, random_state=args.valid_seed, shuffle=True)
            enumerater = enumerate(split_folder.split(X_train_, y_train_ind))

        if args.valid_method == 'random':
            split_folder = KFold(n_splits=n_splits, random_state=args.valid_seed, shuffle=True)
            enumerater = enumerate(split_folder.split(X_train_, y_train_))

        valid_pearson = 0
        best_model = 0
        for j, (train_index, valid_index) in enumerater: # y_bins y_train_ y_train_ind
            X_valid = X_train_[valid_index]
            X_train = X_train_[train_index]
            y_valid = y_train_[valid_index]
            y_train = y_train_[train_index]

            hist_train, _ = np.histogram(y_train, bins=bins)
            hist_valid, _ = np.histogram(y_valid, bins=bins) 
            hist_ratio = np.around(hist_train/hist_valid,3)

            print(f'bins for train and valid: {bins}, i.e. 2≤?<3')
            print('hist_train: number of samples in bins',hist_train)
            print('hist_valid: number of samples in bins',hist_valid)
            print(f'hist_ratio: sample size ratio in bins (train/valid) n_split = {n_splits}\n{hist_ratio}')
            valid_model, max_pearson = valid_info(model, X_train, X_valid, y_train, y_valid, log_dir,\
            f"{method}_{t}{'_'+str(sigma) if sigma else ''}_{2016}_valid_fold_{j}.txt",\
            epoch_num=args.valid_num_epoch, batch_size=args.valid_batch_size, lr=args.valid_lr,)

            if max_pearson > valid_pearson:
                valid_pearson = max_pearson
                best_model = valid_model

        print(f"{method} {t} {str(sigma) if sigma else ''} best validation model with pcc {valid_pearson}\n")
        return best_model

    if args.valid_method == 'bins_pred':
        valid_inds = np.load("../Initialization_bins/y_2016_split_by_pred_test_bins30_checked.npy",allow_pickle=True)
        valid_ind = np.hstack(valid_inds)
        train_ind = np.array([x for x in range(len(y_train_)) if x not in valid_ind])
        
        y_train = y_train_[train_ind]
        y_valid = y_train_[valid_ind]
        
        X_train = X_train_[train_ind]
        X_valid = X_train_[valid_ind]
        
        hist_valid, _ = np.histogram(y_valid, bins=bins) 
        print(f'bins for train and valid from test_pred bins, i.e. 2≤?<3')
        print('hist_valid: number of samples in bins',hist_valid)    
        print(f'sample size ratio of (train/valid) {y_train.shape[0]}:{y_valid.shape[0]}')

        valid_model, max_pearson = valid_info(model, X_train, X_valid, y_train, y_valid, log_dir,\
        f"{method}_{t}{'_'+str(sigma) if sigma else ''}_{2016}_valid_test_pred.txt",\
        epoch_num=args.valid_num_epoch, batch_size=args.valid_batch_size, lr=args.valid_lr,)

        print(f"{method} {t} {str(sigma) if sigma else ''} best validation model with pcc {max_pearson}\n")
        return valid_model

#%%

X_train, X_test, y_train, y_test = tensors_by_year_type(args.year,args.sigma,args.datatype)
model_valid = valid_model(args.model,args.datatype,args.sigma,log_dir)

for i in range(1,args.num_repeat+1): #number of repeats
    # model = transfer_model(args.model,args.datatype,args.transfer_model)
    model = copy.deepcopy(model_valid)
    train_info(model,X_train,X_test,y_train,y_test, log_dir,\
        f"try_{i}_{args.model}_{args.datatype}{'_'+str(args.sigma) if args.sigma else ''}_{args.year}.txt",\
            epoch_num=args.num_epoch,batch_size=args.batch_size,lr=args.lr)
