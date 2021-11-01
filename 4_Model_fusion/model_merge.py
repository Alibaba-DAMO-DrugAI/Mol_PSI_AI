#import numpy as np
#import os
#import torch
#from utils import load_pc, load_sub_data
import scipy.stats
from math import sqrt
#from sklearn.metrics import mean_squared_error
import torch.nn.functional as F


import argparse
import datetime
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from munch import Munch
from numpy.lib.shape_base import expand_dims
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch.utils.data.dataset import T
from torchsummaryX import summary
from tqdm import tqdm

from perspect_classInit2 import *

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=str, default='2016', help='PDBbinding dataset v(2007, 2013, 2016)')
##parser.add_argument('--model', type=str, default='Resnet', help='Deep backbones to extract features from Mol-PSIs.\
##                    Densenet, Resnet for 2D/1D Mol-PSIs, (Image) Transformer for 1D Mol-PSIs')
##parser.add_argument('--sigma', type=float, default=1.5, help='Dimensionality of scale parameter sigma (0.0001, 0.75, 1.5, 2.25)')
##parser.add_argument('--datatype', type=str, default='es', help='ES-IDM 1D (es0), ES-IEM 1D (elec0), ES-IDM 2D (es) and ES-IEM 2D (elec)')

##parser.add_argument('--num_epoch', type=int, default=1001, help='number of training epoch')
##parser.add_argument('--transfer_model', type=int, default=True, help='Training from the transfer model')
parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=24, help='Random seed.')
##parser.add_argument('--lr', type=float, default=3e-5, help='Initial learning rate.')
##parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--num_repeat', type=int, default=1, help='Number of tries for the same experiments')

# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda = args.cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
#if args.model == 'trans' and (args.datatype == 'es0' or args.datatype == 'elec0'):
#    raise ValueError('Image Transformer is not applicable for 2D Mol-PSIs')
#if args.sigma and (args.datatype == 'es0' or args.datatype == 'elec0'):
#    raise ValueError('1D Mol-PSIs is not applicable with sigmas')



def optimize(elec, dist, y_train):
    a, b = compute_coef(elec,dist,y_train)
    print('elec+dist ',compute_loss_pcc(a,b,elec,dist,y_train))
    y_pred = (a*elec + b*dist)
    combo_pcc_train = scipy.stats.pearsonr(y_1_2_3_4[:len(y_train)], y_train)[0]
    combo_pcc_test = scipy.stats.pearsonr(y_1_2_3_4[len(y_train):], y_test)[0]
    print(combo_pcc_train, combo_pcc_test)
    return y_pred

def compute_loss_pcc(m,n,a,b,y):
    loss_a = sqrt(mean_squared_error(a,y))
    pcc_a =scipy.stats.pearsonr(a,y)
    loss_b = sqrt(mean_squared_error(b,y))
    pcc_b = scipy.stats.pearsonr(b,y)

    y_combo = m*a+n*b
    loss = sqrt(mean_squared_error(y_combo,y))
    pcc = scipy.stats.pearsonr(y_combo,y)

    return pcc_a[0],pcc_b[0],pcc[0]

def compute_coef(a,b,c):
    va, ab, ac = np.cov((a,b,c),bias=True)[0]
    _, vb, bc = np.cov((a,b,c),bias=True)[1]

    m = bc*ab-ac*vb
    n = ac*ab-bc*va
    return m/(m+n),n/(m+n)

def elec_result_pcdata(model_path, data_path):
    model = torch.load(model_path)
    _, _, charge_train, _, _, _, _, charge_test, _, _ = load_pc(data_path)
    test_pred = model(charge_test.cuda(), None).cpu().detach().numpy().squeeze()
    train_pred = model(charge_train.cuda(), None).cpu().detach().numpy().squeeze()
    train_pred = np.insert(train_pred, 395, 4.15)  # 395
    result = np.concatenate([train_pred, test_pred], axis=0)
    return result

def dist_result_pcdata(model_path, data_path):
    model = torch.load(model_path)
    dist_train, _, _, _, _, dist_test, _, _, _, _ = load_pc(data_path)
    test_pred = model(dist_test.cuda(), None).cpu().detach().numpy().squeeze()
    train_pred = model(dist_train.cuda(), None).cpu().detach().numpy().squeeze()
    train_pred = np.insert(train_pred, 395, 4.15)  # 395
    result = np.concatenate([train_pred, test_pred], axis=0)
    np.save('results/%s' % model_path.split('/')[1], result)
    return test_pred, train_pred

def result_graphdata(model_path, data_path):
    model = torch.load(model_path)
    adj_train, dist_train, label_train, adj_test, dist_test, label_test = load_sub_data(data_path)

    # idx = np.argsort(label_test.reshape(-1))
    # dist_test = dist_test[idx]; adj_test = adj_test[idx]
    # idx = np.argsort(label_train.reshape(-1))
    # dist_train = dist_train[idx]; adj_train = adj_train[idx]

    test_pred = model(dist_test.cuda(), adj_test.cuda()).cpu().detach().numpy().squeeze()
    train_pred = model(dist_train.cuda(), adj_train.cuda()).cpu().detach().numpy().squeeze()
    #if train_pred.shape[0] == 2763: train_pred = np.insert(train_pred, 395, 4.15)
    #if train_pred.shape[0] == 4056: train_pred = np.insert(train_pred, 2017, 7.27)
    result = np.concatenate([train_pred, test_pred], axis=0)
    return result


def opt_evaluate(path1, path2, path3, path4, y_train, y_test):
    y_1 = np.load(path1) if isinstance(path1, str) else path1
    y_2 = np.load(path2) if isinstance(path2, str) else path2
    y_3 = np.load(path3) if isinstance(path3, str) else path3
    y_4 = np.load(path4) if isinstance(path4, str) else path4

    y_1_train = y_1[:len(y_train)]
    y_2_train = y_2[:len(y_train)]
    y_3_train = y_3[:len(y_train)]
    y_4_train = y_4[:len(y_train)]

    #### combo1
    a, b = compute_coef(y_1_train, y_2_train, y_train)
    print(compute_loss_pcc(a, b, y_1_train, y_2_train, y_train))
    y_1_2_train = (a * y_1_train + b * y_2_train)

    c, d = compute_coef(y_3_train, y_4_train, y_train)
    print(compute_loss_pcc(c, d, y_3_train, y_4_train, y_train))
    y_3_4_train = (c * y_3_train + d * y_4_train)

    m, n = compute_coef(y_1_2_train, y_3_4_train, y_train)
    print(compute_loss_pcc(m, n, y_1_2_train, y_3_4_train, y_train))

    y_1_2 = a * y_1 + b * y_2
    y_3_4 = c * y_3 + d * y_4
    y_1_2_3_4 = m * y_1_2 + n * y_3_4

    combo_pcc_train = scipy.stats.pearsonr(y_1_2_3_4[:len(y_train)], y_train)[0]
    combo_pcc_test = scipy.stats.pearsonr(y_1_2_3_4[len(y_train):], y_test)[0]

    pcc_1 = scipy.stats.pearsonr(y_1[len(y_train):], y_test)[0]
    pcc_2 = scipy.stats.pearsonr(y_2[len(y_train):], y_test)[0]
    pcc_12 = scipy.stats.pearsonr(y_1_2[len(y_train):], y_test)[0]
    pcc_3 = scipy.stats.pearsonr(y_3[len(y_train):], y_test)[0]
    pcc_4 = scipy.stats.pearsonr(y_4[len(y_train):], y_test)[0]
    pcc_34 = scipy.stats.pearsonr(y_3_4[len(y_train):], y_test)[0]
    print('1:%.4f, 2:%.4f, 12:%.4f' % (pcc_1, pcc_2, pcc_12))
    print('3:%.4f, 4:%.4f, 34:%.4f' % (pcc_3, pcc_4, pcc_34))
    print('12:%.4f, 34:%.4f, 1234:%.4f' % (pcc_12, pcc_34, combo_pcc_test))
    return y_1_2_3_4


def joint_opt(npy_path,y_train,y_test):
    pred = [np.load(path) for path in npy_path]
    print(len(y_train),len(pred[0]))
    pccs = [scipy.stats.pearsonr(p[:len(y_train)], y_train)[0] for p in pred]
    pccs = np.around(pccs, decimals=4)
    mse_train = [F.mse_loss(torch.FloatTensor(pred[i][:len(y_train)]), torch.FloatTensor(y_train))
                 for i in range(0, len(npy_path))]
    rmse_train = np.around(np.sqrt(mse_train), decimals=4)
    mse_train = np.around(mse_train, decimals=4)
    print('training set pcc/mse/rmse:\n', pccs, '\n', mse_train,'\n', rmse_train)

    mean_w = np.zeros_like(pccs).reshape([1, -1])
    for i in range(0, len(npy_path)): mean_w[0, i] = 1 / len(npy_path)
    pcc_w = np.zeros_like(pccs).reshape([1, -1])
    for i in range(0, len(npy_path)): pcc_w[0, i] = pccs[i] / np.sum(pccs)
    mse_w = np.zeros_like(mse_train).reshape([1, -1])
    for i in range(0, len(npy_path)): mse_w[0, i] = mse_train[i] / np.sum(mse_train)
    rmse_w = np.zeros_like(rmse_train).reshape([1, -1])
    for i in range(0, len(npy_path)): rmse_w[0, i] = rmse_train[i] / np.sum(rmse_train)
    pcc_mse = np.average(np.concatenate([pcc_w, mse_w], axis=0), axis=0)
    pcc_rmse = np.average(np.concatenate([pcc_w, rmse_w], axis=0), axis=0)

    print('mean weight', end=' ')
    result = np.zeros_like(pred[0])
    for i in range(0, len(npy_path)): result += pred[i] * mean_w[0, i]
    pcc = scipy.stats.pearsonr(result[len(y_train):], y_test)[0]
    rmse = np.sqrt(F.mse_loss(torch.FloatTensor(result[len(y_train):]), torch.FloatTensor(y_test)))
    print('pcc: %f, rmse: %f' % (pcc, np.float(rmse)))

    print('pcc weight', end=' ')
    result = np.zeros_like(pred[0])
    for i in range(0, len(npy_path)): result += pred[i] * pcc_w[0,i]
    pcc = scipy.stats.pearsonr(result[len(y_train):], y_test)[0]
    rmse = np.sqrt(F.mse_loss(torch.FloatTensor(result[len(y_train):]), torch.FloatTensor(y_test)))
    print('pcc: %f, rmse: %f'%(pcc, np.float(rmse)))

    print('mse weight', end=' ')
    result = np.zeros_like(pred[0])
    for i in range(0, len(npy_path)): result += pred[i] * mse_w[0,i]
    pcc = scipy.stats.pearsonr(result[len(y_train):], y_test)[0]
    rmse = np.sqrt(F.mse_loss(torch.FloatTensor(result[len(y_train):]), torch.FloatTensor(y_test)))
    print('pcc: %f, rmse: %f'%(pcc, np.float(rmse)))

    print('rmse weight', end=' ')
    result = np.zeros_like(pred[0])
    for i in range(0, len(npy_path)): result += pred[i] * rmse_w[0,i]
    pcc = scipy.stats.pearsonr(result[len(y_train):], y_test)[0]
    rmse = np.sqrt(F.mse_loss(torch.FloatTensor(result[len(y_train):]), torch.FloatTensor(y_test)))
    print('pcc: %f, rmse: %f'%(pcc, np.float(rmse)))

    print('pcc+mse weight', end=' ')
    result = np.zeros_like(pred[0])
    for i in range(0, len(npy_path)): result += pred[i] * pcc_mse[i]
    pcc = scipy.stats.pearsonr(result[len(y_train):], y_test)[0]
    rmse = np.sqrt(F.mse_loss(torch.FloatTensor(result[len(y_train):]), torch.FloatTensor(y_test)))
    print('pcc: %f, rmse: %f'%(pcc, np.float(rmse)))

    print('pcc+rmse weight', end=' ')
    result = np.zeros_like(pred[0])
    for i in range(0, len(npy_path)): result += pred[i] * pcc_rmse[i]
    pcc = scipy.stats.pearsonr(result[len(y_train):], y_test)[0]
    rmse = np.sqrt(F.mse_loss(torch.FloatTensor(result[len(y_train):]), torch.FloatTensor(y_test)))
    print('pcc: %f, rmse: %f'%(pcc, np.float(rmse)))

def tensors_by_year_type(year, sigma, t,):
    #y_train = np.load(f'./PDB_withNTU/PDB_data/y_train_{year}.npy').astype(np.float32).reshape(-1, 1)
    y_train = np.load(f'../PDB_data/y_train_{year}.npy').astype(np.float32).reshape(-1, 1)
    #y_test = np.load(f'./PDB_withNTU/PDB_data/y_test_{year}.npy').astype(np.float32).reshape(-1, 1)
    y_test = np.load(f'../PDB_data/y_test_{year}.npy').astype(np.float32).reshape(-1, 1)
    y = np.vstack([y_train, y_test])

    if t=='es':
        X_sp = sparse.load_npz(f'./PDB_withNTU/Temp_tensor_Dec9/{year}_normDist_sigma_{sigma}/tensor_{year}_norm.npz').astype(np.float32)
        X = X_sp.toarray().reshape(y.shape[0],36,100,100)

    if t=='elec':
        X_sp = sparse.load_npz(f'./PDB_withNTU/Temp_tensor_Dec9/{year}_elecDist_sigma_{sigma}/tensor_{year}_elec.npz').astype(np.float32)
        X = X_sp.toarray().reshape(y.shape[0],50,100,100)

    if t =='es0':
        X = np.load(f'../Temp_tensor_Dec9/tensor_{year}_norm_numzero.npy').astype(np.float32)[:,np.newaxis,:,:]

    if t=='elec0':
        X = normalize(np.load(f'../Temp_tensor_Dec9/tensor_{year}_elec_numzero.npy')).astype(np.float32)[:,np.newaxis,:,:]

    X_train = X[:len(y_train)]
    X_test = X[len(y_train):]
    print(f"load data {year} {t} {sigma}")
    return X_train, X_test, y_train.squeeze(), y_test.squeeze()



def best_model(method, t, sigma='1.5', year=2007, i=1):
    rootNo = 5
    if t == 'es' and method == 'Densenet':
        model = Dense_back(36)
        test_path = ('./best_model%d/try_%d_Densenet_es_%s_%d.pth'%(rootNo,i,sigma,year))
    elif t == 'es' and method == 'Resnet':
        model = Res_back(36)
        test_path = ('./best_model%d/try_%d_Resnet_es_%s_%d.pth'%(rootNo,i,sigma,year))
    elif t == 'es' and method == 'Transformer':
        test_path = ('./best_model%d/try_%d_Transformer_es_%s_%d.pth'%(rootNo,i,sigma,year)) 
    elif t == 'elec' and method == 'Densenet':
        model = Dense_back(50)
        test_path = ('./best_model%d/try_%d_Densenet_elec_%s_%d.pth'%(rootNo,i,sigma,year))
    elif t == 'elec' and method == 'Resnet':
        model = Res_back(50)
        test_path = ('./best_model%d/try_%d_Resnet_elec_%s_%d.pth'%(rootNo,i,sigma,year)) 
    elif t == 'elec' and method == 'Transformer':
        test_path = ('./best_model%d/try_%d_Transformer_elec_%s_%d.pth'%(rootNo,i,sigma,year)) 
    elif t == 'es0' and method == 'Densenet':
        model = Dense_back(1,3072)
        test_path = ('./best_model%d/try_%d_Densenet_es0_%d.pth'%(rootNo,i,year)) 
    elif t == 'es0' and method == 'Resnet':
        model = Res_back(1)
        test_path = ('./best_model%d/try_%d_Resnet_es0_%d.pth'%(rootNo,i,year))
    elif t == 'elec0' and method == 'Densenet':
        model = Dense_back(1,3072)
        test_path = ('./best_model%d/try_%d_Densenet_elec0_%d.pth'%(rootNo,i,year)) 
    elif t == 'elec0' and method == 'Resnet':
        model = Res_back(1)
        test_path = ('./best_model%d/try_%d_Resnet_elec0_%d.pth'%(rootNo,i,year))
    else:
        raise Exception("No such a model")

    model.load_state_dict(torch.load(test_path))
    model = model.cuda() if args.cuda else model
    print(f"load model {method} {t} {sigma} {year} try {i}")
    return model, test_path



#if __name__ == '__main__':
year = 2007 #args.year
y_train = np.load(f'../PDB_data/y_train_{year}.npy').astype(np.float32).reshape(-1, 1)
y_test = np.load(f'../PDB_data/y_test_{year}.npy').astype(np.float32).reshape(-1, 1)
if not os.path.exists(os.path.join('model_predictions5')): os.mkdir(os.path.join('model_predictions5'))


method = 'Resnet'

sigma = '0'
for datatype in ('elec0', 'es0'):   
    X_train, X_test, y_train, y_test = tensors_by_year_type(year,sigma,datatype)

    for i in range(args.num_repeat,args.num_repeat+1): #number of test round
        model,test_path = best_model(method,datatype,sigma,year,i)
        model.eval()
        train_pred=model(torch.from_numpy(X_train).float().cuda())
        train_pred=train_pred.cpu().detach().squeeze().numpy()
        test_pred=model(torch.from_numpy(X_test).float().cuda())
        test_pred=test_pred.cpu().detach().squeeze().numpy()
        #y_pred=model(torch.from_numpy(X_test).float()).detach().squeeze().numpy()
        y_pred = np.concatenate([train_pred, test_pred], axis=0)
        np.save(test_path.replace('best_model5/','model_predictions5/').replace('.pth','.npy'), y_pred)
        pcc = sp.stats.pearsonr(test_pred, y_test)[0]
        rmse = sqrt(mean_squared_error(test_pred, y_test))
        print('year:',year, 'method:',method,'datatype:', datatype,'sigma:', sigma,'test_pcc:', pcc,'test_rmse:', rmse)


for datatype in ('elec', 'es'):
    for sigma in ('0.0001', '0.75', '1.5', '2.25'):   
        X_train, X_test, y_train, y_test = tensors_by_year_type(year,sigma,datatype)

        for i in range(args.num_repeat,args.num_repeat+1): 
            model,test_path = best_model(method,datatype,sigma,year,i)
            model.eval()
            train_pred=model(torch.from_numpy(X_train).float().cuda())
            train_pred=train_pred.cpu().detach().squeeze().numpy()
            test_pred=model(torch.from_numpy(X_test).float().cuda())
            test_pred=test_pred.cpu().detach().squeeze().numpy()
            #y_pred=model(torch.from_numpy(X_test).float()).detach().squeeze().numpy()
            y_pred = np.concatenate([train_pred, test_pred], axis=0)
            np.save(test_path.replace('best_model5/','model_predictions5/').replace('.pth','.npy'), y_pred)
            pcc = sp.stats.pearsonr(test_pred, y_test)[0]
            rmse = sqrt(mean_squared_error(test_pred, y_test))
            print('year:',year, 'method:',method,'datatype:', datatype,'sigma:', sigma,'test_pcc:', pcc,'test_rmse:', rmse)



for i in range(args.num_repeat,args.num_repeat+1):
    npy_path = [
                './model_predictions5/try_%d_Resnet_elec_0.0001_%d.npy'%(i,year), 
                './model_predictions5/try_%d_Resnet_elec_0.75_%d.npy'%(i,year),
                './model_predictions5/try_%d_Resnet_elec_1.5_%d.npy'%(i,year),
                './model_predictions5/try_%d_Resnet_elec_2.25_%d.npy'%(i,year),
                './model_predictions5/try_%d_Resnet_es_0.0001_%d.npy'%(i,year),
                './model_predictions5/try_%d_Resnet_es_0.75_%d.npy'%(i,year),
                './model_predictions5/try_%d_Resnet_es_1.5_%d.npy'%(i,year),
                './model_predictions5/try_%d_Resnet_es_2.25_%d.npy'%(i,year),
                './model_predictions5/try_%d_Resnet_es0_%d.npy'%(i,year),
                './model_predictions5/try_%d_Resnet_elec0_%d.npy'%(i,year),
                ]
joint_opt(npy_path,y_train,y_test)

