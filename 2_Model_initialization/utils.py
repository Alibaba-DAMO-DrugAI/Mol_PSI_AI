from math import sqrt

import numpy as np
import scipy as sp
import torch
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def sample_by_int_bins(y,bins,x,n_sample):
    y_bin = np.digitize(y, bins)-1
    inds = np.arange(len(y_bin))
    y_c = y[y_bin == x]
    c = inds[y_bin == x]
    if n_sample>len(c):
        n_sample = len(c)
    return np.random.choice(c,n_sample,replace=False)

def normalize(x,ran=1):
    ma=x.max()
    mi=x.min()
    x[x!=0]=(x[x!=0]-mi)/(ma-mi)*ran
    return x

def pearsonr_torch(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    
    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    
    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def compute_coef(a,b,c):
    va, ab, ac = np.cov((a,b,c),bias=True)[0]
    _, vb, bc = np.cov((a,b,c),bias=True)[1]

    m = bc*ab-ac*vb
    n = ac*ab-bc*va
    return m/(m+n),n/(m+n)

def minimize_me(m,a,b,c):
    # multiply by -1 to maximize
    n = 1-m
    return -1 * pearsonr(m*a+n*b,c)[0]

def opt_coef(a,b,c):
    res = minimize(minimize_me, (1), args=(a,b,c), bounds=[(0,1)])
    m = res.x[0]
    n = 1-m
    return m, n

def compute_loss_pcc(m,n,a,b,y):
    loss_a = sqrt(mean_squared_error(a,y))
    pcc_a =sp.stats.pearsonr(a,y)
    loss_b = sqrt(mean_squared_error(b,y))
    pcc_b = sp.stats.pearsonr(b,y)

    y_combo = m*a+n*b
    loss = sqrt(mean_squared_error(y_combo,y))
    pcc = sp.stats.pearsonr(y_combo,y)

    return loss_a,loss_b,loss,pcc_a[0],pcc_b[0],pcc[0]

def opt_evaluate(path1,path2,path3,path4,y_train,y_test):

    y_1 = np.load(path1) if isinstance(path1, str) else path1
    y_2 = np.load(path2) if isinstance(path2, str) else path2
    y_3 = np.load(path3) if isinstance(path3, str) else path3 
    y_4 = np.load(path4) if isinstance(path4, str) else path4
    
    y_1_train = y_1[:len(y_train)]
    y_2_train = y_2[:len(y_train)]
    y_3_train = y_3[:len(y_train)]
    y_4_train = y_4[:len(y_train)]

    y_1_test = y_1[len(y_train):]
    y_2_test = y_2[len(y_train):]
    y_3_test = y_3[len(y_train):]
    y_4_test = y_4[len(y_train):]

    #### combo1
    a, b = compute_coef(y_1_train,y_2_train,y_train)
    print(compute_loss_pcc(a, b, y_1_train, y_2_train, y_train))
    y_1_2_train = (a*y_1_train + b*y_2_train)

    c, d = compute_coef(y_3_train,y_4_train,y_train)
    print(compute_loss_pcc(c, d, y_3_train, y_4_train, y_train))
    y_3_4_train = (c*y_3_train + d*y_4_train)

    m, n = compute_coef(y_1_2_train,y_3_4_train,y_train)
    print(compute_loss_pcc(m,n,y_1_2_train,y_3_4_train,y_train))
    y_1_2_3_4_train = (m*y_1_2_train + n*y_3_4_train)

    y_1_2 = a*y_1 + b*y_2
    y_3_4 = c*y_3 + d*y_4
    y_1_2_3_4 = m*y_4 +n*y_3_4
    
    combo_pcc_train = sp.stats.pearsonr(y_1_2_3_4[:len(y_train)],y_train)[0]
    combo_pcc_test = sp.stats.pearsonr(y_1_2_3_4[len(y_train):],y_test)[0]
    combo_rmse_train = mean_squared_error(y_1_2_3_4[:len(y_train)],y_train)
    combo_rmse_test = mean_squared_error(y_1_2_3_4[len(y_train):],y_test)
    print(combo_pcc_train, combo_rmse_train)
    print(combo_pcc_test, combo_rmse_test)
    return y_1_2_3_4, a,b,c,d,m,n
