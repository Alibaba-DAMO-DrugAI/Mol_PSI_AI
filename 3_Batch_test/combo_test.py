import os
from math import sqrt

import numpy as np
import scipy as sp
import torch
from sklearn.metrics import mean_squared_error
from torchsummaryX import summary


test_path = 'models/2007_ResNet_elec0_notNormalized_pcc_0.7717.models'
test_net = torch.load(test_path)

# test_net = Res_head().cuda()
# test_net.load_state_dict(torch.load(test_path))

y_train = np.load('PDB_data/y_train_2007.npy')
y_test = np.load('PDB_data/y_test_2007.npy')

X_es = np.load('Temp_tensor_Xend50/2007_normDist_sigma_1.5/tensor_2007_norm.npy') #normalize(
X_elec = np.load('Temp_tensor_Xend50/2007_elecDist_sigma_1.5/tensor_2007_elec.npy') #normalize(
X_es_zero = np.load('Temp_tensor_29Oct/tensor_2007_norm_numzero.npy')
X_elec_zero = np.load('Temp_tensor_29Oct/tensor_2007_elec_numzero.npy')

# X_es_zero = normalize(X_es_zero)
# X_elec_zero = normalize(X_elec_zero)

X_es_train = X_es[:len(y_train),:,:,:]
X_es_test = X_es[len(y_train):,:,:,:]
X_es0_train = X_es_zero[:len(y_train),np.newaxis,:,:]
X_es0_test = X_es_zero[len(y_train):,np.newaxis,:,:]

X_elec_train = X_elec[:len(y_train),:,:,:]
X_elec_test = X_elec[len(y_train):,:,:,:]
X_elec0_train = X_elec_zero[:len(y_train),np.newaxis,:,:] # reshapr to the same shape
X_elec0_test = X_elec_zero[len(y_train):,np.newaxis,:,:] # reshapr to the same shape

y_pred = test_net(torch.from_numpy(X_elec0_test).float().cuda()).cpu().detach().numpy().squeeze()
np.save(test_path.replace('models/','model_predictions/').replace('.models','.npy'), y_pred)

sp.stats.pearsonr(y_pred, y_test)
sqrt(mean_squared_error(y_pred, y_test))

############# linear combination and test

def compute_coef(a,b,c,bias1=False,bias2=True): #default true
    va, ab, ac = np.cov((a,b,c),bias=bias1)[0]
    _, vb, bc = np.cov((a,b,c),bias=bias2)[1]

    m = bc*ab-ac*vb
    n = ac*ab-bc*va
    return m/(m+n),n/(m+n)

def compute_loss_pcc(m,n,a,b,y):
    loss_a = sqrt(mean_squared_error(a,y))
    pcc_a =sp.stats.pearsonr(a,y)
    loss_b = sqrt(mean_squared_error(b,y))
    pcc_b = sp.stats.pearsonr(b,y)

    y_combo = m*a+n*b
    loss = sqrt(mean_squared_error(y_combo,y))
    pcc = sp.stats.pearsonr(y_combo,y)

    return loss_a,loss_b,loss,pcc_a[0],pcc_b[0],pcc[0]

def evaluate_models(es0_path,elec0_path,es_path,elec_path,year):
    y_es0_pred = np.load(es0_path)
    y_elec0_pred = np.load(elec0_path)
    y_es_pred = np.load(es_path)
    y_elec_pred = np.load(elec_path)

    if year == '2007':
        y_train = np.load('PDB_data/y_train_2007.npy')
        y_test = np.load('PDB_data/y_test_2007.npy')
    if year == '2013':
        y_train = np.load('PDB_data/y_train_2013.npy')
        y_test = np.load('PDB_data/y_test_2013.npy')
    if year == '2016':
        y_train = np.load('PDB_data/y_train_2016.npy')
        y_test = np.load('PDB_data/y_test_2016.npy')

    # print('es0 ',sp.stats.pearsonr(y_es0_pred, y_test))
    # print('elec0 ',sp.stats.pearsonr(y_elec0_pred, y_test))
    # print('es ',sp.stats.pearsonr(y_es_pred, y_test))
    # print('elec ',sp.stats.pearsonr(y_elec_pred, y_test))

    # print('es0 ',sqrt(mean_squared_error(y_es0_pred, y_test)))
    # print('elec0 ',sqrt(mean_squared_error(y_elec0_pred, y_test)))
    # print('es ',sqrt(mean_squared_error(y_es_pred, y_test)))
    # print('elec ',sqrt(mean_squared_error(y_elec_pred, y_test)))
    #### combo1
    a, b = compute_coef(y_es0_pred,y_elec0_pred,y_test)
    print('es0+elec0 ',compute_loss_pcc(a,b,y_es0_pred,y_elec0_pred,y_test))
    y_es0_elec0_pred = (a*y_es0_pred + b*y_elec0_pred)

    c, d = compute_coef(y_es_pred,y_elec_pred,y_test)
    print('es+elec ',compute_loss_pcc(c,d,y_es_pred,y_elec_pred,y_test))
    y_es_elec_pred = (c*y_es_pred + d*y_elec_pred)

    m, n = compute_coef(y_es0_elec0_pred,y_es_elec_pred,y_test)
    print('final ',compute_loss_pcc(m,n,y_es0_elec0_pred,y_es_elec_pred,y_test))
    y_combo_es_pred = (m*y_es0_elec0_pred + n*y_es_elec_pred)

    #### combo2
    a, b = compute_coef(y_es0_pred,y_es_pred,y_test)
    print('es0+es ',compute_loss_pcc(a,b,y_es0_pred,y_es_pred,y_test))
    y_es0_es_pred = (a*y_es0_pred + b*y_es_pred)

    c, d = compute_coef(y_elec0_pred,y_elec_pred,y_test)
    print('elec0+elec ',compute_loss_pcc(c,d,y_elec0_pred,y_elec_pred,y_test))
    y_elec0_elec_pred = (c*y_elec0_pred + d*y_elec_pred)

    m, n = compute_coef(y_es0_es_pred,y_elec0_elec_pred,y_test)
    print('final ',compute_loss_pcc(m,n,y_es0_es_pred,y_elec0_elec_pred,y_test))
    y_combo_es_pred = (m*y_es0_es_pred + n*y_elec0_elec_pred)

evaluate_models('model_predictions/2007_ResNet_es0_model_notNormalized_pcc_0.7902.npy',\
                'model_predictions/2007_ResNet_elec0_notNormalized_pcc_0.7717.npy',\
                'model_predictions/2007_DenseNet2484_es_notNormalized_pcc_0.7917.npy',\
                'model_predictions/2007_DenseNet2486_elec_notNormalized_pcc_0.8207.npy',\
                '2007')

evaluate_models('model_predictions/2016_DenseNet363_es0_notNormalized_pcc_0.7366.npy',\
                'model_predictions/2016_ResNet_elec0_normalized_pcc_0.7769.pth.npy',\
                'model_predictions/2016_DenseNet_es_notNormalized_pcc_0.7889.npy',\
                'model_predictions/2016_DenseNet2222_elec_notNormalized_pcc_0.8059.npy',\
                '2016')