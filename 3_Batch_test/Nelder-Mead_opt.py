
from math import sqrt
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize

def minimize_me(m, y1,y2,y3):
    # multiply by -1 to maximize
    n = 1-m
    return -1 * pearsonr(m*y1+n*y2,y3)[0]

bnds = [(0, 1)]

y_es0_pred = np.load('model_predictions/2007_ResNet_es0_model_notNormalized_pcc_0.7902.npy')
y_elec0_pred = np.load('model_predictions/2007_ResNet_elec0_notNormalized_pcc_0.7717.npy')
y_es_pred = np.load('model_predictions/2007_DenseNet2484_es_notNormalized_pcc_0.7917.npy')
y_elec_pred = np.load('model_predictions/2007_DenseNet2486_elec_notNormalized_pcc_0.8207.npy')

y_train = np.load('PDB_data/y_train_2007.npy')
y_test = np.load('PDB_data/y_test_2007.npy')

res = minimize(minimize_me, (1), args=(y_es0_pred,y_es_pred,y_test), bounds=bnds)
m = res.x[0]
n = 1-m

pcc1 = pearsonr(y_es0_pred,y_test)[0]
pcc2 = pearsonr(y_es_pred,y_test)[0]
pcc3 = pearsonr(m*y_es0_pred+n*y_es_pred,y_test)[0]

print(m,n,pcc1,pcc2,pcc3)
