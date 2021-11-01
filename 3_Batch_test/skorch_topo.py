import importlib
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.stats import pearsonr
from skorch import NeuralNetRegressor
from skorch.callbacks import Checkpoint, LoadInitState
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from tqdm import tqdm

import perspect

importlib.reload(perspect)
from perspect import (ComboLoss, Dense_back, ImageTransformer, InputShapeSetter_Res,
                    InputShapeSetter_Dense_1d, InputShapeSetter_Dense_2d, InputShapeSetter_Trans,
                    Res_back, lrs, normalize, pcc_test, pcc_train)

#####################
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

######################
def transfer_on_dir_1d(transfer_dir):
    # transfer_dir = 'log_skorch/2007_elec0_res'

    year = transfer_dir.split('/')[-1].split('_')[-3]
    t = transfer_dir.split('_')[-2]
    method = transfer_dir.split('_')[-1]

    y_train = y_train_dict[year]
    y_test = y_test_dict[year]
    y = np.vstack([y_train, y_test])

    X = X_dict_1d[f'{year}_{t}'] 
    X_train = X[:len(y_train)]
    X_test = X[len(y_train):]

    cp = Checkpoint(dirname=transfer_dir)
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
            callbacks=[InputShapeSetter_Dense_1d(), cp, pcc_test, pcc_train, ] 
            #         _=net.initialize()

        )
        
    try:
        print(year,t,method)
        _=net.initialize()
        _=net.fit(X_train, y_train)
    
    finally:
        print(year,t,method)
        net.load_params(checkpoint=cp)
        print('final model demo',net.predict(X_test).squeeze()[::16].round(2),y_test.squeeze()[::16].round(2),sep='\n')        
        pcc=pearsonr(net.predict(X_test).squeeze(),y_test.squeeze())[0]
        print('final model pcc',pcc,'\nfinish',year,t,method,)

def transfer_on_dir_2d(transfer_dir):
    # transfer_dir = 'log_skorch/2007_es_1.5_dense'
    
    year = transfer_dir.split('/')[-1].split('_')[-4]
    sigma = transfer_dir.split('_')[-2]
    t = transfer_dir.split('_')[-3]
    method = transfer_dir.split('_')[-1]

    y_train = y_train_dict[year]
    y_test = y_test_dict[year]
    y = np.vstack([y_train, y_test])

    X_sp = X_dict_2d[f'{year}_{t}_{sigma}'] 
    X = X_sp.toarray().reshape(y.shape[0],36,100,100) if t=='es' else X_sp.toarray().reshape(y.shape[0],50,100,100)
    X_train = X[:len(y_train)]
    X_test = X[len(y_train):]

    cp = Checkpoint(dirname=transfer_dir) 
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
        
    try:
        print(year,t,sigma,method)
        _=net.initialize()
        _=net.fit(X_train, y_train) 
    
    finally:
        print(year,t,sigma,method)
        net.load_params(checkpoint=cp)
        print('final model demo',net.predict(X_test).squeeze()[::16].round(2),y_test.squeeze()[::16].round(2),sep='\n')        
        pcc=pearsonr(net.predict(X_test).squeeze(),y_test.squeeze())[0]
        print('final model pcc',pcc,'\nfinish',year,t,method,)

# dirs_1d = sorted([f'./log_skorch/{x}' for x in os.listdir('./log_skorch') if 'es0_' in x or 'elec0_' in x])
# dir_1d_generator = (x for x in dirs_1d)

# dir_id = dir_generator.__next__()
# transfer_on_dir_1d(dir_id)

dirs_2d = sorted([f'./log_skorch/{x}' for x in os.listdir('./log_skorch') if 'es_' in x or 'elec_' in x])
dir_2d_generator = (x for x in dirs_2d)

dir_id = dir_2d_generator.__next__()
transfer_on_dir_2d(dir_id)

