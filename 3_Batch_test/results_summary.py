import importlib
import os
from math import sqrt

import numpy as np
from scipy.stats.stats import SigmaclipResult
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from skorch import NeuralNetRegressor
from skorch.callbacks import Checkpoint, LoadInitState
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from tqdm import tqdm

import perspect

importlib.reload(perspect)
from perspect import (ComboLoss, Dense_back, ImageTransformer,
                    InputShapeSetter_Dense_1d, InputShapeSetter_Dense_2d,
                    InputShapeSetter_Res, InputShapeSetter_Trans, Res_back,
                    lrs, normalize, pcc_test, pcc_train, opt_evaluate)

years = ['2007','2013','2016']
types = ['es','elec']
sigmas = [0.0001, 0.75, 1.5, 2.25]

y_train_dict = {year: np.load(f'./PDB_withNTU/PDB_data/y_train_{year}.npy').astype(np.float32).reshape(-1, 1)\
                for year in years}

y_test_dict = {year: np.load(f'./PDB_withNTU/PDB_data/y_test_{year}.npy').astype(np.float32).reshape(-1, 1)\
                for year in years}

X_dict_1d = {f'{year}_{t}0': np.load(f'./PDB_withNTU/Temp_tensor_Dec9/tensor_{year}_norm_numzero.npy').astype(np.float32)[:,np.newaxis,:,:]\
            if t=='es' else normalize(np.load(f'./PDB_withNTU/Temp_tensor_Dec9/tensor_{year}_elec_numzero.npy')).astype(np.float32)[:,np.newaxis,:,:]\
            for year in years for t in types}

X_dict_2d = {f'{year}_{t}_{sigma}': sparse.load_npz(f'./PDB_withNTU/Temp_tensor_Dec9/{year}_normDist_sigma_{sigma}/tensor_{year}_norm.npz').astype(np.float32)\
            if t=='es' else sparse.load_npz(f'./PDB_withNTU/Temp_tensor_Dec9/{year}_elecDist_sigma_{sigma}/tensor_{year}_elec.npz').astype(np.float32)\
            for year in years for t in types for sigma in tqdm(sigmas)} # np.load npy npz

def load_save_dir_1d(load_dir):
    # load_dir = 'log_skorch/2007_elec0_res'

    year = load_dir.split('/')[-1].split('_')[-3]
    t = load_dir.split('_')[-2]
    method = load_dir.split('_')[-1]

    y_train = y_train_dict[year]
    y_test = y_test_dict[year]
    y = np.vstack([y_train, y_test])

    X = X_dict_1d[f'{year}_{t}'] 
    X_train = X[:len(y_train)]
    X_test = X[len(y_train):]

    cp = Checkpoint(dirname=load_dir)
    # load_state = LoadInitState(cp)

    if method == 'res':
        net = NeuralNetRegressor(
            Res_back,
            max_epochs=500,
            batch_size=512,
            optimizer=torch.optim.Adam,
            lr=5e-3,
            criterion = ComboLoss,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            train_split=predefined_split(Dataset(X_test, y_test)),
            device='cuda',
            callbacks=[InputShapeSetter_Res(), cp, pcc_test, pcc_train, ] 
            #lrs cp, load_state,
        )
        
    if method == 'dense':
        net = NeuralNetRegressor(
            Dense_back,
            max_epochs=500,
            batch_size=512,
            optimizer=torch.optim.Adam, 
            #SGD Adam
            lr=5e-6,
            criterion = ComboLoss,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            train_split=predefined_split(Dataset(X_test, y_test)),
            device='cuda',
            callbacks=[InputShapeSetter_Dense_1d(), cp, pcc_test, pcc_train,] 
            #         _=net.initialize()
        )

    print(year,t,method)
    _=net.initialize()
    if method == 'dense':
        _=net.set_params(module__flatten=3072)
    net.load_params(checkpoint=cp)
    
    y_pred_test = net.predict(X_test).squeeze()    
    pcc = pearsonr(y_pred_test,y_test.squeeze())[0]
    rmse = sqrt(mean_squared_error(y_pred_test, y_test))

    y_pred = net.predict(X).squeeze()
    np.save(f'./results_skorch/{year}_{t}_{method}_test_{pcc:.4f}_{rmse:.4f}.npy', y_pred)

    print('save model',year,t,method,'test',pcc,rmse)

def load_save_dir_2d(load_dir):
    # load_dir = 'log_skorch/2007_es_1.5_dense'

    year = load_dir.split('/')[-1].split('_')[-4]
    sigma = load_dir.split('_')[-2]
    t = load_dir.split('_')[-3]
    method = load_dir.split('_')[-1]


    y_train = y_train_dict[year]
    y_test = y_test_dict[year]
    y = np.vstack([y_train, y_test])

    X_sp = X_dict_2d[f'{year}_{t}_{sigma}'] 
    X = X_sp.toarray().reshape(y.shape[0],36,100,100) if t=='es' else X_sp.toarray().reshape(y.shape[0],50,100,100)
    X_train = X[:len(y_train)]
    X_test = X[len(y_train):]

    cp = Checkpoint(dirname=load_dir)
    load_state = LoadInitState(cp)

    if method == 'res':
        net = NeuralNetRegressor(
            Res_back,
            max_epochs=300,
            batch_size=384,
            optimizer=torch.optim.Adam,
            lr=7e-4,
            criterion = ComboLoss,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            train_split=predefined_split(Dataset(X_test, y_test)),
            device='cuda',
            callbacks=[InputShapeSetter_Res(), load_state, cp, pcc_test, pcc_train, ] 
        )   # lrs cp, load_state, load_state 
        
    if method == 'dense':
        net = NeuralNetRegressor(
            Dense_back,
            max_epochs=300,
            batch_size=384,
            optimizer=torch.optim.Adam, #SGD Adam
            lr=7e-4,
            criterion = ComboLoss,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            train_split=predefined_split(Dataset(X_test, y_test)),
            device='cuda',
            callbacks=[InputShapeSetter_Dense_2d(), load_state, cp, pcc_test, pcc_train, ]
        ) # load_state lrs
        
    if method == 'trans':
        net = NeuralNetRegressor(
            ImageTransformer,
            max_epochs=300,
            batch_size=384,
            optimizer=torch.optim.Adam, #SGD
            lr=7e-4,
            criterion = ComboLoss,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            train_split=predefined_split(Dataset(X_test, y_test)),
            device='cuda',
            callbacks=[InputShapeSetter_Trans(), load_state, cp, pcc_test, pcc_train, ]
        ) #  lrs
        

    print(year,t,sigma,method)
    _=net.initialize()
    if method == 'res' or method == 'dense':
        _=net.set_params(module__combo=X.shape[1])
    if method == 'trans':
        net.set_params(module__channels=X.shape[1])
    net.load_params(checkpoint=cp)
    
    y_pred_test = net.predict(X_test).squeeze()
    pcc = pearsonr(y_pred_test,y_test.squeeze())[0]
    rmse = sqrt(mean_squared_error(y_pred_test, y_test))
    
    y_pred = net.predict(X).squeeze()
    np.save(f'./results_skorch/{year}_{t}_{sigma}_{method}_test_{pcc:.4f}_{rmse:.4f}.npy', y_pred)
    
    print('save model',year,t,method,'test',pcc,rmse)

dirs = os.listdir('./log_skorch')
for dir in dirs:
    load_dir = f'./log_skorch/{dir}'
    if 'es0' in dir or 'elec0' in dir:
        print('load 1d',load_dir)
        load_save_dir_1d(load_dir)
    if 'es_' in dir or 'elec_' in dir:
        load_save_dir_2d(load_dir)
        print('load 2d',load_dir)

#%% 2007

combo_es = ('results_skorch/2007_es_0.0001_trans_test_0.7376_1.6966.npy',\
        'results_skorch/2007_es_0.75_res_test_0.7823_1.6746.npy',\
        'results_skorch/2007_es_1.5_dense_test_0.8099_1.4896.npy',\
        'results_skorch/2007_es_2.25_trans_test_0.7809_1.5229.npy')

combo_elec = ('results_skorch/2007_elec_0.0001_res_test_0.6948_1.9539.npy',\
        'results_skorch/2007_elec_0.75_res_test_0.7864_1.6005.npy',\
        'results_skorch/2007_elec_1.5_res_test_0.8245_1.4058.npy',\
        'results_skorch/2007_elec_2.25_dense_test_0.8065_1.4733.npy')

es0_elec0 = ('results_skorch/2007_es0_dense_test_0.8029_1.4984.npy',\
            'results_skorch/2007_elec0_res_test_0.7954_1.4838.npy')

y_es,*_ = opt_evaluate(*combo_es, y_train_dict['2007'].squeeze(), y_test_dict['2007'].squeeze())
print(_)
y_elec,*_ = opt_evaluate(*combo_elec, y_train_dict['2007'].squeeze(), y_test_dict['2007'].squeeze())
print(_)
y,*_ =  opt_evaluate(*es0_elec0,y_elec,y_es, y_train_dict['2007'].squeeze(), y_test_dict['2007'].squeeze())
print(_)

#%% 2013

combo_es = ('results_skorch/2013_es_0.0001_dense_test_0.6928_1.8001.npy',\
        'results_skorch/2013_es_0.75_res_test_0.7435_1.6101.npy',\
        'results_skorch/2013_es_1.5_dense_test_0.7282_1.5935.npy',\
        'results_skorch/2013_es_2.25_dense_test_0.7325_1.5609.npy')

combo_elec = ('results_skorch/2013_elec_0.0001_res_test_0.7262_1.6769.npy',\
        'results_skorch/2013_elec_0.75_res_test_0.7838_1.4877.npy',\
        'results_skorch/2013_elec_1.5_res_test_0.7872_1.5302.npy',\
        'results_skorch/2013_elec_2.25_res_test_0.7714_1.5709.npy')

es0_elec0 = ('results_skorch/2013_es0_dense_test_0.7526_1.5717.npy',\
            'results_skorch/2013_elec0_dense_test_0.7544_1.5727.npy')

y_es,*_ = opt_evaluate(*combo_es, y_train_dict['2013'].squeeze(), y_test_dict['2013'].squeeze())
print(_)
y_elec,*_ = opt_evaluate(*combo_elec, y_train_dict['2013'].squeeze(), y_test_dict['2013'].squeeze())
print(_)
y,*_ =  opt_evaluate(*es0_elec0,y_elec,y_es, y_train_dict['2013'].squeeze(), y_test_dict['2013'].squeeze())
print(_)
#%% 2016

combo_es = ('results_skorch/2016_es_0.0001_res_test_0.7931_1.7465.npy',\
        'results_skorch/2016_es_0.75_dense_test_0.7902_1.4325.npy',\
        'results_skorch/2016_es_1.5_trans_test_0.8120_1.3150.npy',\
        'results_skorch/2016_es_2.25_trans_test_0.8087_1.3817.npy')

combo_elec = ('results_skorch/2016_elec_0.0001_dense_test_0.7841_1.4917.npy',\
        'results_skorch/2016_elec_0.75_trans_test_0.8067_1.3120.npy',\
        'results_skorch/2016_elec_1.5_trans_test_0.8219_1.2779.npy',\
        'results_skorch/2016_elec_2.25_dense_test_0.8213_1.3127.npy')

es0_elec0 = ('results_skorch/2016_es0_res_test_0.7689_1.4585.npy',\
            'results_skorch/2016_elec0_dense_test_0.7664_1.4901.npy')

y_es,*_ = opt_evaluate(*combo_es, y_train_dict['2016'].squeeze(), y_test_dict['2016'].squeeze())
print(_)
y_elec,*_ = opt_evaluate(*combo_elec, y_train_dict['2016'].squeeze(), y_test_dict['2016'].squeeze())
print(_)
y,*_ =  opt_evaluate(*es0_elec0,y_elec,y_es, y_train_dict['2016'].squeeze(), y_test_dict['2016'].squeeze())
print(_)

