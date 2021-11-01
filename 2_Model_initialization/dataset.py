import os

import numpy as np
import numpy_indexed as npi
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from utils import normalize

# train_2007 = [line.strip() for line in open(src+'train_2007.txt')] 
# train_2013 = [line.strip() for line in open(src+'train_2013.txt')] 
# train_2016 = [line.strip() for line in open(src+'train_2016.txt')] 
# train_all = train_2007 + train_2013 + train_2016

# test_2007 = [line.strip() for line in open(src+'core_2007.txt')] 
# test_2013 = [line.strip() for line in open(src+'core_2013.txt')] 
# test_2016 = [line.strip() for line in open(src+'core_2016.txt')] 
# test_all = test_2007 + test_2013 + test_2016

# y_train_2007 = np.load(src+'y_train_2007.npy')
# y_train_2013 = np.load(src+'y_train_2013.npy')
# y_train_2016 = np.load(src+'y_train_2016.npy')

# y_test_2007 = np.load(src+'y_test_2007.npy')
# y_test_2013 = np.load(src+'y_test_2013.npy')
# y_test_2016 = np.load(src+'y_test_2016.npy')

def gaussian_mix_spectrum_by_file(src, sigma, dis_type, adapt_range=False, exclude_zero=False, v2=False):
    '''
    Parameters
    ----------
    src : str for input filtration and eigenvalues 
    sigma : width for generating peak
    dis_type : norm or elec 
    v2 : optional
        for formatting. The default is False.

    Returns
    -------
    combo_vals : (combo_num x nf x ngrid) np.array
        tensor.
    '''
    
    # src = 'PDB_data/2007_normDist/1a0q_ES_b01_c2_pocket_eigv.txt'
    assert dis_type in ['elec','norm']
    if dis_type == 'elec':
        n_combo=50; nf=100; y_start=0; y_end=1; ngrid=100; x_start=-15; x_end=50; sigma=sigma
    if dis_type == 'norm':
        n_combo=36; nf=100; y_start=0; y_end=20; ngrid=100; x_start=-15; x_end=50; sigma=sigma

    src_x = np.loadtxt(src)
    if v2: #for norm
        src2 = src.replace('.txt', '_p2.txt')
        src_x = np.vstack([src_x, np.loadtxt(src2)])
    src_x[src_x<=1e-8]=0

    df = pd.DataFrame(src_x, columns=['combo ID','betti','ys','xs'])
    # df = df.astype(dtype={"combo ID":"int64","betti":"int64","ys":"float64","xs":"float64"})
    df = df[df['betti']==1]

    if exclude_zero:
        df = df[df['xs']!=0]

    xt=np.linspace(x_start, x_end, num=ngrid, endpoint=False)
    yt=np.linspace(y_start, y_end, num=nf, endpoint=False)

    df['y_ins'] = np.digitize(df['ys'], yt, right=True)
    df['x_ins'] = np.digitize(df['xs'], xt.T, right=True)
    # print(len(set(df.y_ins)))
    # print(len(set(df.x_ins)))

    combo_vals = np.zeros((n_combo,nf,ngrid))
    for name,mf in df.groupby('combo ID'):
        if adapt_range:
            range_dict = range_2007
            xt = np.linspace(-5, range_dict[dis_type][name], num=ngrid, endpoint=False)

        val = np.tile(xt,(len(mf),1))
        val = np.square((val-mf['xs'].values.reshape(-1,1)))/(2*sigma**2)
        val = pd.DataFrame(np.exp(-val))
        # val = val.groupby(mf['y_ins'].tolist()).sum().reindex(range(nf)).fillna(0)
        ind, val = npi.group_by(mf['y_ins']).sum(val)
        mf_val = pd.DataFrame(val,index=ind).reindex(range(100)).fillna(0).values
        # my_vals.append(mf_val)
        combo_vals[int(name-1),:,:] = mf_val #normalize or not
    # combo_vals[combo_vals<=1e-8]=0
    return combo_vals

def zero_eigen_map(src, dis_type, v2=False):
    assert dis_type in ['elec','norm']
    if dis_type == 'elec':
        n_combo=50; y_start=0; y_end=1; nf=100
    if dis_type == 'norm':
        n_combo=36; y_start=0; y_end=20; nf=100

    src_x = np.loadtxt(src)
    if v2:
        # nf=100;y_start=0;y_end=50
        src2 = src.replace('.txt', '_p2.txt')
        src_x = np.vstack([np.loadtxt(src), np.loadtxt(src2)])
    src_x[src_x<=1e-8]=0

    df = pd.DataFrame(src_x, columns=['combo ID','betti','ys','xs'])
    df = df[df['betti']==1]
    df = df[df['xs']==0]

    yt=np.linspace(y_start, y_end, num=nf, endpoint=False)
    df['y_ins'] = np.digitize(df['ys'], yt, right=True)

    mf = df.groupby(['combo ID','y_ins']).size()

    zero_map = np.zeros((n_combo,nf))
    for (i_combo, y_in), num_zero in mf.iteritems():
        zero_map[int(i_combo-1),y_in-1] = num_zero
    return zero_map

def resolve_dir(src_dir, ref_dir, dst, dis_type):
    sorted_dir=sorted(os.listdir(src_dir), key=lambda x: ref_dir.index(x.split('_')[0]))
    tensor_list = [gaussian_mix_spectrum_by_file(src_dir+'/'+src, dis_type) for src in tqdm(sorted_dir)]
    X = np.array(tensor_list)
    np.savez(dst, X)

def tensors_by_year_type(year, sigma, t, with_id=False):
    y_train = np.load(f'../PDB_data/y_train_{year}.npy').astype(np.float32).reshape(-1, 1)
    y_test = np.load(f'../PDB_data/y_test_{year}.npy').astype(np.float32).reshape(-1, 1)
    y_train_id = np.loadtxt(f'../PDB_data/train_{year}.txt',dtype=str)
    y_test_id = np.loadtxt(f'../PDB_data/core_{year}.txt',dtype=str)
    
    y = np.vstack([y_train, y_test])
    
    if t=='es':
        X_sp = sparse.load_npz(f'../PDB_data/input_tensors/{year}_normDist_sigma_{sigma}/tensor_{year}_norm.npz').astype(np.float32)
        X = X_sp.toarray().reshape(y.shape[0],36,100,100)  
    
    if t=='elec':
        X_sp = sparse.load_npz(f'../PDB_data/input_tensors/{year}_elecDist_sigma_{sigma}/tensor_{year}_elec.npz').astype(np.float32)
        X = X_sp.toarray().reshape(y.shape[0],50,100,100)
        
    if t =='es0':
        X = np.load(f'../PDB_data/input_tensors/tensor_{year}_norm_numzero.npy').astype(np.float32)[:,np.newaxis,:,:]
        
    if t=='elec0':
        X = normalize(np.load(f'../PDB_data/input_tensors/tensor_{year}_elec_numzero.npy')).astype(np.float32)[:,np.newaxis,:,:]
        
    X_train = X[:len(y_train)]
    X_test = X[len(y_train):]
    
    if with_id:
        print(f"load Mol-PSI data {year} {t} {str(sigma) if sigma else ''} with PDB IDs\n")
        return X_train, X_test, y_train.squeeze(), y_test.squeeze(), y_train_id, y_test_id 
    else:
        print(f"load Mol-PSI data {year} {t} {str(sigma) if sigma else ''}\n")
        return X_train, X_test, y_train.squeeze(), y_test.squeeze() 

