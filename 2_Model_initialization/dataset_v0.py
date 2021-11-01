import numpy as np
from scipy import sparse
from utils import normalize

def tensors_by_year_type(year, sigma, t, with_id=False):
    y_train = np.load(f'./PDB_withNTU/PDB_data/y_train_{year}.npy').astype(np.float32).reshape(-1, 1)
    y_test = np.load(f'./PDB_withNTU/PDB_data/y_test_{year}.npy').astype(np.float32).reshape(-1, 1)
    y_train_id = np.loadtxt(f'./PDB_withNTU/PDB_data/train_{year}.txt',dtype=str)
    y_test_id = np.loadtxt(f'./PDB_withNTU/PDB_data/core_{year}.txt',dtype=str)
    
    y = np.vstack([y_train, y_test])
    
    if t=='es':
        X_sp = sparse.load_npz(f'./PDB_withNTU/Temp_tensor_Dec9/{year}_normDist_sigma_{sigma}/tensor_{year}_norm.npz').astype(np.float32)
        X = X_sp.toarray().reshape(y.shape[0],36,100,100)  
    
    if t=='elec':
        X_sp = sparse.load_npz(f'./PDB_withNTU/Temp_tensor_Dec9/{year}_elecDist_sigma_{sigma}/tensor_{year}_elec.npz').astype(np.float32)
        X = X_sp.toarray().reshape(y.shape[0],50,100,100)
        
    if t =='es0':
        X = np.load(f'./PDB_withNTU/Temp_tensor_Dec9/tensor_{year}_norm_numzero.npy').astype(np.float32)[:,np.newaxis,:,:]
        
    if t=='elec0':
        X = normalize(np.load(f'./PDB_withNTU/Temp_tensor_Dec9/tensor_{year}_elec_numzero.npy')).astype(np.float32)[:,np.newaxis,:,:]
        
    X_train = X[:len(y_train)]
    X_test = X[len(y_train):]
    if with_id:
        print(f"load Mol-PSI data {year} {t} {str(sigma) if sigma else ''} with PDB IDs\n")
        return X_train, X_test, y_train.squeeze(), y_test.squeeze(), y_train_id, y_test_id 
    else:
        print(f"load Mol-PSI data {year} {t} {str(sigma) if sigma else ''}\n")
        return X_train, X_test, y_train.squeeze(), y_test.squeeze() 
