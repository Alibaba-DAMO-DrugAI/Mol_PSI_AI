
import os
from math import sqrt

import numpy as np
import scipy as sp
import skorch
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision.models as models
from einops import rearrange
from scipy import sparse
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from skorch import NeuralNetRegressor
from skorch.callbacks import (Callback, Checkpoint, EpochScoring,
                              LoadInitState, LRScheduler)
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from torch import nn
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau


# from torchsummary import summary
def tensors_by_year_type(year, sigma, t,):
    y_train = np.load(f'../PDB_data/y_train_{year}.npy').astype(np.float32).reshape(-1, 1)
    y_test = np.load(f'../PDB_data/y_test_{year}.npy').astype(np.float32).reshape(-1, 1)
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
    print(f"load Mol-PSI data {year} {t} {str(sigma) if sigma else ''}")
    return X_train, X_test, y_train.squeeze(), y_test.squeeze()

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

def reg_model_2_cls(model, method):
    if method != 'Transformer':
        dim_org = model.regression._modules['2'].state_dict()['weight'].shape[1]
        model.regression._modules['2'] = nn.Linear(dim_org, 10)
    
    if method == 'Transformer':    
        dim_org = model.nn2.state_dict()['weight'].shape[1]
        model.nn2 = nn.Linear(dim_org, 10)
    
    return model

class SimpleCNN(torch.nn.Module):
    def __init__(self, combos=1, dim=5184):
        super(SimpleCNN,self).__init__()
        #es 36*250
        self.es_norm0 = torch.nn.BatchNorm2d(combos)
        self.es_zero_conv1 = torch.nn.Conv2d(combos, 2*combos, 3, padding=1, groups=combos) #2d -> 1d
        self.es_zero_conv2 = torch.nn.Conv2d(2*combos, 4*combos, 3, padding=1, groups=combos)
        self.es_zero_pool1 = torch.nn.MaxPool2d(2)
        self.es_zero_norm1 = torch.nn.BatchNorm2d(4*combos)
        self.es_zero_drop1 = torch.nn.Dropout(p=0.5)

        self.es_zero_conv3 = torch.nn.Conv2d(4*combos, 4*combos, 3, padding=1, groups=combos)
        self.es_zero_conv4 = torch.nn.Conv2d(4*combos, 2*combos, 3, padding=1)
        self.es_zero_pool2 = torch.nn.MaxPool2d(2)
        self.es_zero_norm2 = torch.nn.BatchNorm2d(2*combos)
        self.es_zero_drop2 = torch.nn.Dropout(p=0.5)

        self.es_zero_conv5 = torch.nn.Conv2d(2*combos, combos, 3, padding=1, groups=combos)
        self.es_zero_conv6 = torch.nn.Conv2d(combos, combos, 3, padding=1,)
        self.es_zero_pool3 = torch.nn.MaxPool2d(2)
        self.es_zero_norm3 = torch.nn.BatchNorm2d(combos)
        self.es_zero_drop3 = torch.nn.Dropout(p=0.5)

        #elec count 0
        # self.elec_zero_norm0 = torch.nn.BatchNorm1d(50)
        self.dense1 = torch.nn.Linear(dim,512) # 5000 200 5184 7200
        self.norm1 = torch.nn.BatchNorm1d(512)
        self.dense2 = torch.nn.Linear(512,64)
        self.drop3 = torch.nn.Dropout(p=0.5)
        self.dense3 = torch.nn.Linear(64,1)


    def num_flat_features(self, v):
        size = v.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x4):

        x4 = F.relu(self.es_zero_conv1(x4))
        x4 = self.es_zero_pool1(F.relu(self.es_zero_conv2(x4)))
        x4 = self.es_zero_norm1(x4)
        x4 = self.es_zero_drop1(x4)

        x4 = F.relu(self.es_zero_conv3(x4))
        x4 = self.es_zero_pool2(F.relu(self.es_zero_conv4(x4)))
        x4 = self.es_zero_norm2(x4)
        x4 = self.es_zero_drop2(x4)

        x4 = F.relu(self.es_zero_conv5(x4))
        x4 = self.es_zero_pool3(F.relu(self.es_zero_conv6(x4)))
        x4 = self.es_zero_norm3(x4)
        x4 = self.es_zero_drop3(x4)

        x4 = x4.view(-1, self.num_flat_features(x4))
        x4 = F.relu(self.dense1(x4))
        x4 = self.norm1(x4)
        x4 = F.relu(self.dense2(x4))
        x4 = self.drop3(x4)
        x4 = self.dense3(x4)

        return x4

class Res_feat(torch.nn.Module):
    def __init__(self,combo=1):
        super(Res_back,self).__init__()
        resnet18 = models.resnet18(pretrained=True) #False
        modules = list(resnet18.children())[:-1]
        modules[0] = torch.nn.Conv2d(combo, 64, kernel_size=7, stride=2, padding=3, bias=False) #Conv2d_in 1 50
        #preivous 3 channels to 1
        self.backbone = torch.nn.Sequential(*modules)
        self.regression = torch.nn.Sequential(torch.nn.Linear(512,128),torch.nn.Dropout(p=0.5),torch.nn.Linear(128,1))

    def num_flat_features(self, v):
        size = v.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):

        x = self.backbone(x)
        x = x.view(-1,self.num_flat_features(x))
        x = self.regression(x)
        return x

class Res_back(torch.nn.Module):
    def __init__(self,combo=1):
        super(Res_back,self).__init__()
        resnet18 = models.resnet18(pretrained=True) #False
        modules = list(resnet18.children())[:-1]
        modules[0] = torch.nn.Conv2d(combo, 64, kernel_size=7, stride=2, padding=3, bias=False) #Conv2d_in 1 50
        #preivous 3 channels to 1
        self.backbone = torch.nn.Sequential(*modules)
        self.regression = torch.nn.Sequential(torch.nn.Linear(512,128),\
            torch.nn.Linear(128,64),torch.nn.Linear(64,1),) #,torch.nn.LeakyReLU()

    def num_flat_features(self, v):
        size = v.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):

        x = self.backbone(x)
        x = x.view(-1,self.num_flat_features(x))
        x = self.regression(x)
        
        return x

class Dense_feat(torch.nn.Module):
    def __init__(self, combo=1, flatten=1024):
        super(Dense_back,self).__init__()
        densenet121 = models.densenet121(pretrained=False) #False
        modules = list(densenet121.children())[:-1]
        modules[0][0] = torch.nn.Conv2d(combo, 64, kernel_size=7, stride=2, padding=3, bias=False) #Conv2d_in 1 50
        #preivous 3 channels to 1
        self.backbone = torch.nn.Sequential(*modules)
        self.align = torch.nn.Sequential(torch.nn.Linear(flatten,1024),) # 3072

    def num_flat_features(self, v):
        size = v.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1,self.num_flat_features(x))
        x = self.align(x)
        return x

class Dense_back(torch.nn.Module):
    def __init__(self, combo=1, flatten=9216):
        super(Dense_back,self).__init__()
        densenet121 = models.densenet121(pretrained=False) #False
        modules = list(densenet121.children())[:-1]
        modules[0][0] = torch.nn.Conv2d(combo, 64, kernel_size=7, stride=2, padding=3, bias=False) #Conv2d_in 1 50
        #preivous 3 channels to 1
        self.backbone = torch.nn.Sequential(*modules)
        self.regression = torch.nn.Sequential(torch.nn.Linear(flatten,1024),\
            torch.nn.Linear(1024,128),torch.nn.Linear(128,1)) # 3072 ,torch.nn.LeakyReLU()

    def num_flat_features(self, v):
        size = v.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1,self.num_flat_features(x))
        x = self.regression(x)
        return x

class DNN(torch.nn.Module):
    def __init__(self,dim):
        super(DNN,self).__init__()

        self.dense1 = torch.nn.Linear(dim,2048)
        self.drop1 = torch.nn.Dropout(p=0.5)
        self.dense2 = torch.nn.Linear(2048,1024)
        self.drop2 = torch.nn.Dropout(p=0.5)
        self.dense3 = torch.nn.Linear(1024,512)
        self.drop3 = torch.nn.Dropout(p=0.5)
        self.dense4 = torch.nn.Linear(512,256)
        self.drop4 = torch.nn.Dropout(p=0.5)
        self.dense5 = torch.nn.Linear(256,64)
        self.drop5 = torch.nn.Dropout(p=0.5)
        self.dense6 = torch.nn.Linear(64,1)

    def num_flat_features(self, v):
        size = v.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        num_in = self.num_flat_features(x)
        # self.dense1 = torch.nn.Linear(num_in,2048)

        x = x.view(-1, num_in)
        x = F.relu(self.dense1(x))
        x = self.drop1(x)
        x = F.relu(self.dense2(x))
        x = self.drop2(x)
        x = F.relu(self.dense3(x))
        x = self.drop3(x)
        x = F.relu(self.dense4(x))
        x = self.drop4(x)
        x = F.relu(self.dense5(x))
        x = self.drop5(x)
        x = F.relu(self.dense6(x))
        return x

class MergeNet(torch.nn.Module):
    def __init__(self):
        super(MergeNet,self).__init__()

        self.es_channel = Dense_head(36,9216)
        self.elec_channel = Dense_head(50,9216)
        self.es0_channel = Res_head(1)
        self.elec0_channel = Res_head(1)

        self.dense1 = torch.nn.Linear(512,256)
        self.drop1 = torch.nn.Dropout(p=0.5)
        self.dense2 = torch.nn.Linear(256,64)
        self.drop2 = torch.nn.Dropout(p=0.5)
        self.dense3 = torch.nn.Linear(64,1)

    def forward(self,es,elec,es0,elec0):
        es = self.es_channel(es)
        elec = self.elec_channel(elec)
        es0 = self.es0_channel(es0)
        elec0 = self.elec0_channel(elec0)

        x = torch.cat((es, elec, es0, elec0),dim=1)

        x = F.relu(self.dense1(x))
        x = self.drop1(x)
        x = F.relu(self.dense2(x))
        x = self.drop2(x)
        x = self.dense3(x)
        return x

class AdamergeNet(torch.nn.Module):
    def __init__(self):
        super(AdamergeNet,self).__init__()

        self.es_channel = Dense_head(36,9216)
        self.elec_channel = Dense_head(50,9216)
        self.es0_channel = Res_head(1)
        self.elec0_channel = Res_head(1)

    def forward(self,es,elec,es0,elec0):
        es = self.es_channel(es)
        elec = self.elec_channel(elec)
        es0 = self.es0_channel(es0)
        elec0 = self.elec0_channel(elec0)

        return es, elec, es0, elec0

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.do2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = True) # Wq,Wk,Wv for each vector, thats why *3
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)
        
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)        
        self.do1 = nn.Dropout(dropout)
        

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x) #gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1) #follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v) #product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)') #concat heads into one matrix, ready for next encoder block
        out =  self.nn1(out)
        out = self.do1(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attention, mlp in self.layers:
            x = attention(x, mask = mask) # go to attention
            x = mlp(x) #go to MLP_Block
        return x

class ImageTransformer_feat(nn.Module):
    def __init__(self, *, image_size=100, patch_size=50, num_classes=1, dim=64, depth=6, heads=8, mlp_dim=128, channels = 36, dropout = 0.5, emb_dropout = 0.1):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2  # e.g. (32/4)**2= 64
        patch_dim = channels * patch_size ** 2  # e.g. 3*8**2 = 64*3

        # self.init_bn = nn.BatchNorm2d(64) # additional bn to normalize
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_patches + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std = .02) # initialized based on the paper
        self.patch_conv= nn.Conv2d(channels, dim, patch_size, stride = patch_size) #eqivalent to x matmul E, E= embedd matrix, this is the linear patch projection
        
        #self.E = nn.Parameter(nn.init.normal_(torch.empty(BATCH_SIZE_TRAIN,patch_dim,dim)),requires_grad = True)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) #initialized based on the paper
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, mlp_dim)  # if finetuning, just use a linear layer without further hidden layers (paper)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.GELU() # use additinal hidden layers only when training on large datasets


    def forward(self, img, mask = None):
        p = self.patch_size

        x = self.patch_conv(img) # each of 64 vecotrs is linearly transformed with a FFN equiv to E matmul
        # x = self.init_bn(x) # additional bn to normalize

        #x = torch.matmul(x, self.E)
        x = rearrange(x, 'b c h w -> b (h w) c') # 64 vectors in rows representing 64 patches, each 64*3 long

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x, mask) #main game
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)
        
        return x

class ImageTransformer(nn.Module):
    def __init__(self, *, image_size=100, patch_size=10, num_classes=1, dim=64, depth=6, heads=8, mlp_dim=128, channels = 50, dropout = 0.5, emb_dropout = 0.1):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2  # e.g. (32/4)**2= 64
        patch_dim = channels * patch_size ** 2  # e.g. 3*8**2 = 64*3

        # self.init_bn = nn.BatchNorm2d(64) # additional bn to normalize
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_patches + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std = .02) # initialized based on the paper
        self.patch_conv= nn.Conv2d(channels, dim, patch_size, stride = patch_size) #eqivalent to x matmul E, E= embedd matrix, this is the linear patch projection
        
        #self.E = nn.Parameter(nn.init.normal_(torch.empty(BATCH_SIZE_TRAIN,patch_dim,dim)),requires_grad = True)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) #initialized based on the paper
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, mlp_dim)  # if finetuning, just use a linear layer without further hidden layers (paper)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.GELU() # use additinal hidden layers only when training on large datasets
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(mlp_dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias)
        self.do2 = nn.Dropout(dropout)

    def forward(self, img, mask = None):
        p = self.patch_size

        x = self.patch_conv(img) # each of 64 vecotrs is linearly transformed with a FFN equiv to E matmul
        # x = self.init_bn(x) # additional bn to normalize

        #x = torch.matmul(x, self.E)
        x = rearrange(x, 'b c h w -> b (h w) c') # 64 vectors in rows representing 64 patches, each 64*3 long

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x, mask) #main game

        x = self.to_cls_token(x[:, 0])
        
        x = self.nn1(x)
        # x = self.af1(x)
        # x = self.do1(x)
        x = self.nn2(x)
        # x = self.do2(x)
        
        return x

################# trainning

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

def train_info(net,X_train,X_test,y_train,y_test,log_dir,log_file,epoch_num=2,batch_size=512,lr=3e-4,weight_decay=1e-9,upper_dim=False,cuda=True):
    if upper_dim:
        X_train = X_train[:,np.newaxis,:,:]
        X_test = X_test[:,np.newaxis,:,:]
    max_pearson = 0
    best_model_path = 0

    loss_fn = torch.nn.MSELoss(reduction='mean')
    train_data = Data.TensorDataset(torch.from_numpy(X_train).float(),\
                                    torch.from_numpy(y_train).float())
    train_loader = Data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=True)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr, weight_decay=weight_decay)

    with open(log_dir+'/'+log_file, 'w') as f:
        for epoch in range(epoch_num):
            net.zero_grad()
            net.train()
            for step,(batch_x,batch_y) in enumerate(train_loader):
                if cuda:
                    batch_x=batch_x.cuda()
                    batch_y=batch_y.cuda()

                optimizer.zero_grad()

                batch_pred = net(batch_x)
                loss = torch.sqrt(loss_fn(batch_pred.squeeze(),batch_y))
                loss.backward()
                optimizer.step()
                print('%4d %3d %12.5e\n'%(epoch,step,loss.item()))
                f.write('%4d %3d %12.5e\n'%(epoch,step,loss.item()))

            net.eval()
            if cuda:
                y_pred=net(torch.from_numpy(X_test).float().cuda())
                y_pred=y_pred.cpu().detach().squeeze().numpy()
            else:
                y_pred=net(torch.from_numpy(X_test).float()).detach().squeeze().numpy()

            mse = mean_squared_error(y_test,y_pred)
            print('%4d test RMSE: %8.4f\n'%(epoch,sqrt(mse)))
            f.write('%4d test RMSE: %8.4f\n'%(epoch,sqrt(mse)))

            pearcorr = sp.stats.pearsonr(y_test, y_pred)
            print('%4d test pearsonR: %8.4f\n'%(epoch,pearcorr[0]))
            f.write('%4d test pearsonR: %8.4f\n'%(epoch,pearcorr[0]))

            # save best model to best_model_dir
            if max_pearson < pearcorr[0]:
                max_pearson = pearcorr[0]
                if os.path.exists(f"{log_dir}/best_models/{best_model_path}"):
                    os.remove(f"{log_dir}/best_models/{best_model_path}")
                best_model_path = log_file.split('/')[-1].replace('.txt',f'_epoch_{epoch}_pcc_{pearcorr[0]:.4f}.pth')
                print(f"save model at epoch {epoch} with loss {sqrt(mse):.4f} pearcorr {pearcorr[0]:.4f}")
                f.write(f"save model at epoch {epoch} with loss {sqrt(mse):.4f} pearcorr {pearcorr[0]:.4f}")
                torch.save(net.state_dict(), f"{log_dir}/best_models/" + best_model_path)

def train_info_merge(net,trains,tests,y_train,y_test,log_file):

    es_train, elec_train, es0_train, elec0_train = trains
    es_test, elec_test, es0_test, elec0_test = tests
    # new appendix dim
    # es0_train = es0_train[:,np.newaxis,:,:]
    # es0_test = es0_train[:,np.newaxis,:,:]
    # elec0_train = elec0_train[:,np.newaxis,:,:]
    # elec0_test = elec0_train[:,np.newaxis,:,:]

    loss_fn = torch.nn.MSELoss(reduction='mean')
    train_data = Data.TensorDataset(torch.from_numpy(es_train).float(),\
                                    torch.from_numpy(elec_train).float(),\
                                    torch.from_numpy(es0_train).float(),\
                                    torch.from_numpy(elec0_train).float(),\
                                    torch.from_numpy(y_train).float())

    train_loader = Data.DataLoader(dataset=train_data,batch_size=300,shuffle=True,num_workers=2,drop_last=True)
    optimizer = torch.optim.Adam(net.parameters(),lr=5e-3, weight_decay=1e-6)
    max_pearson = 0
    with open(log_file, 'w') as f:
        for epoch in range(2000):
            net.zero_grad()
            net.train()
            for step,(batch_es, batch_elec, batch_es0, batch_elec0, batch_y) in enumerate(train_loader):
                batch_es=batch_es.cuda()
                batch_elec=batch_elec.cuda()
                batch_es0=batch_es0.cuda()
                batch_elec0=batch_elec0.cuda()

                batch_y=batch_y.cuda()

                optimizer.zero_grad()

                batch_pred = net(batch_es, batch_elec, batch_es0, batch_elec0,)
                loss = torch.sqrt(loss_fn(batch_pred.squeeze(),batch_y))
                # vx = batch_pred.squeeze()
                # vy = batch_y
                # loss2 = torch.sum(vx * vy) * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2))
                # loss = loss1 -loss2
                # loss.backward()
                optimizer.step()

                print('%4d %3d %12.5e\n'%(epoch,step,loss.item()))
                f.write('%4d %3d %12.5e\n'%(epoch,step,loss.item()))

            net.eval()
            y_pred=net(torch.from_numpy(es_test).float().cuda(),\
                    torch.from_numpy(elec_test).float().cuda(),\
                    torch.from_numpy(es0_test).float().cuda(),\
                    torch.from_numpy(elec0_test).float().cuda())

            y_pred=y_pred.cpu().detach().squeeze().numpy()

            mse = mean_squared_error(y_test,y_pred)
            print('%4d test RMSE: %8.4f\n'%(epoch,sqrt(mse)))
            f.write('%4d test RMSE: %8.4f\n'%(epoch,sqrt(mse)))

            pearcorr = sp.stats.pearsonr(y_test, y_pred)
            print('%4d test pearsonR: %8.4f\n'%(epoch,pearcorr[0]))
            f.write('%4d test pearsonR: %8.4f\n'%(epoch,pearcorr[0]))

            # save best model to best_model_dir
            if max_pearson < pearcorr[0]:
                max_pearson = pearcorr[0]
                print(f"save model at epoch {epoch} with loss {sqrt(mse):.4f} pearcorr {pearcorr[0]:.4f}")
                f.write(f"save model at epoch {epoch} with loss {sqrt(mse):.4f} pearcorr {pearcorr[0]:.4f}")
                # torch.save(net.state_dict(),'models/'+f'es0_model.pth')

def train_info_Adamerge(net,trains,tests,y_train,y_test,log_file):
    es_train, elec_train, es0_train, elec0_train = trains
    es_test, elec_test, es0_test, elec0_test = tests

    loss_fn = torch.nn.MSELoss(reduction='mean')
    train_data = Data.TensorDataset(torch.from_numpy(es_train).float(),\
                                    torch.from_numpy(elec_train).float(),\
                                    torch.from_numpy(es0_train).float(),\
                                    torch.from_numpy(elec0_train).float(),\
                                    torch.from_numpy(y_train).float())

    train_loader = Data.DataLoader(dataset=train_data,batch_size=50,shuffle=True,num_workers=2,drop_last=True)
    optimizer = torch.optim.Adam(net.parameters(),lr=5e-3, weight_decay=1e-6)
    max_pearson = 0
    with open(log_file, 'w') as f:
        for epoch in range(2000):
            net.zero_grad()
            net.train()
            for step,(batch_es, batch_elec, batch_es0, batch_elec0, batch_y) in enumerate(train_loader):
                batch_es=batch_es.cuda()
                batch_elec=batch_elec.cuda()
                batch_es0=batch_es0.cuda()
                batch_elec0=batch_elec0.cuda()
                batch_y=batch_y.cuda()
                optimizer.zero_grad()

                y_es, y_elec, y_es0, y_elec0 = net(batch_es, batch_elec, batch_es0, batch_elec0,)
                #es & es0
                m_es,n_es = compute_coef(y_es.cpu().detach().squeeze().numpy(),\
                                            y_es0.cpu().detach().squeeze().numpy(),\
                                            batch_y.cpu().detach().numpy())
                y_combo_es = m_es*y_es + n_es*y_es0
                print("loss:\tes\tes0\tcombo\tpcc:\tes\tes0\tcombo")
                print(compute_loss_pcc(m_es,n_es,\
                                        y_es.cpu().detach().squeeze().numpy(),\
                                        y_es0.cpu().detach().squeeze().numpy(),\
                                        batch_y.cpu().detach().squeeze().numpy()))
                f.write("loss:\tes_combo\telec_combo\tfinal\tpcc:\tes_combo\telec_combo\tfnial\n")
                f.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(\
                        *compute_loss_pcc(m_es,n_es,\
                                        y_es.cpu().detach().squeeze().numpy(),\
                                        y_es0.cpu().detach().squeeze().numpy(),\
                                        batch_y.cpu().detach().squeeze().numpy())))
                #elec & elec0
                m_elec,n_elec = compute_coef(y_elec.cpu().detach().squeeze().numpy(),\
                                            y_elec0.cpu().detach().squeeze().numpy(),\
                                            batch_y.cpu().detach().numpy())
                y_combo_elec = m_elec*y_elec + n_elec*y_elec0
                print("loss:\telec\telec0\tcombo\tpcc:\telec\telec0\tcombo")
                print(compute_loss_pcc(m_elec,n_elec,\
                                        y_elec.cpu().detach().squeeze().numpy(),\
                                        y_elec0.cpu().detach().squeeze().numpy(),\
                                        batch_y.cpu().detach().squeeze().numpy()))
                f.write("loss:\tes_combo\telec_combo\tfinal\tpcc:\tes_combo\telec_combo\tfnial\n")
                f.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(\
                        *compute_loss_pcc(m_elec,n_elec,\
                                    y_elec.cpu().detach().squeeze().numpy(),\
                                    y_elec0.cpu().detach().squeeze().numpy(),\
                                    batch_y.cpu().detach().squeeze().numpy())))
                #es_combo & elec_combo
                m,n = compute_coef(y_combo_es.cpu().detach().squeeze().numpy(),\
                                    y_combo_elec.cpu().detach().squeeze().numpy(),\
                                    batch_y.cpu().detach().numpy())
                y_combo = m*y_combo_es + n*y_combo_elec
                print("loss:\tes_combo\telec_combo\tfinal\tpcc:\tes_combo\telec_combo\tfnial")
                print(compute_loss_pcc(m,n,\
                                        y_combo_es.cpu().detach().squeeze().numpy(),\
                                        y_combo_elec.cpu().detach().squeeze().numpy(),\
                                        batch_y.cpu().detach().squeeze().numpy()))
                f.write("loss:\tes_combo\telec_combo\tfinal\tpcc:\tes_combo\telec_combo\tfnial\n")
                f.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(\
                        *compute_loss_pcc(m,n,\
                                        y_combo_es.cpu().detach().squeeze().numpy(),\
                                        y_combo_elec.cpu().detach().squeeze().numpy(),\
                                        batch_y.cpu().detach().squeeze().numpy())))

                loss = torch.sqrt(loss_fn(y_combo.squeeze(),batch_y))
                loss.backward()
                optimizer.step()

                print('%d %3d %12.5e\n'%(epoch,step,loss.item()))
                f.write('%d %3d %12.5e\n'%(epoch,step,loss.item()))

            net.eval()
            y_es_pred, y_elec_pred, y_es0_pred, y_elec0_pred = \
            net(torch.from_numpy(es_test).float().cuda(),\
                torch.from_numpy(elec_test).float().cuda(),\
                torch.from_numpy(es0_test).float().cuda(),\
                torch.from_numpy(elec0_test).float().cuda())

            y_combo_es_pred = (m_es*y_es_pred + n_es*y_es0_pred).cpu().detach().squeeze().numpy()
            y_combo_elec_pred = (m_elec*y_elec_pred + n_elec*y_elec0_pred).cpu().detach().squeeze().numpy()
            y_combo_pred = (m*y_combo_es_pred + n*y_combo_elec_pred)

            mse_es = mean_squared_error(y_combo_es_pred,y_test)
            mse_elec = mean_squared_error(y_combo_elec_pred,y_test)
            mse = mean_squared_error(y_combo_pred,y_test)
            print('%d test RMSE: %8.4f %8.4f %8.4f\n'%(epoch,sqrt(mse_es),sqrt(mse_elec),sqrt(mse)))
            f.write('%d test RMSE: %8.4f %8.4f %8.4f\n'%(epoch,sqrt(mse_es),sqrt(mse_elec),sqrt(mse)))

            pearcorr_es = sp.stats.pearsonr(y_combo_es_pred, y_test)[0]
            pearcorr_elec = sp.stats.pearsonr(y_combo_elec_pred, y_test)[0]
            pearcorr = sp.stats.pearsonr(y_combo_pred, y_test)[0]

            print('%d test pearsonR: %8.4f %8.4f %8.4f\n'%(epoch,pearcorr_es,pearcorr_elec,pearcorr))
            f.write('%d test pearsonR: %8.4f %8.4f %8.4f\n'%(epoch,pearcorr_es,pearcorr_elec,pearcorr))
            # save best model to best_model_dir
            if max_pearson < pearcorr:
                max_pearson = pearcorr
                print(f"save model at epoch {epoch} with loss {sqrt(mse):.4f} pearcorr {pearcorr:.4f}")
                f.write(f"save model at epoch {epoch} with loss {sqrt(mse):.4f} pearcorr {pearcorr:.4f}")
                # torch.save(net.state_dict(),'models/'+f'es0_model.pth')

def train_info_Optmerge(net,trains,tests,y_train,y_test,log_file):
    es_train, elec_train, es0_train, elec0_train = trains
    es_test, elec_test, es0_test, elec0_test = tests

    loss_fn = torch.nn.MSELoss(reduction='mean')
    train_data = Data.TensorDataset(torch.from_numpy(es_train).float(),\
                                    torch.from_numpy(elec_train).float(),\
                                    torch.from_numpy(es0_train).float(),\
                                    torch.from_numpy(elec0_train).float(),\
                                    torch.from_numpy(y_train).float()) #also for sigma 1 2 3 4

    train_loader = Data.DataLoader(dataset=train_data,batch_size=50,shuffle=True,num_workers=2,drop_last=True)
    optimizer = torch.optim.Adam(net.parameters(),lr=5e-3, weight_decay=1e-6)
    max_pearson = 0
    with open(log_file, 'w') as f:
        for epoch in range(2000):
            net.zero_grad()
            net.train()
            for step,(batch_es, batch_elec, batch_es0, batch_elec0, batch_y) in enumerate(train_loader):
                batch_es=batch_es.cuda()
                batch_elec=batch_elec.cuda()
                batch_es0=batch_es0.cuda()
                batch_elec0=batch_elec0.cuda()
                batch_y=batch_y.cuda()
                optimizer.zero_grad()

                y_es, y_elec, y_es0, y_elec0 = net(batch_es, batch_elec, batch_es0, batch_elec0,)
                #es & es0
                m_es,n_es = opt_coef(y_es.cpu().detach().squeeze().numpy(),\
                                            y_es0.cpu().detach().squeeze().numpy(),\
                                            batch_y.cpu().detach().numpy())
                y_combo_es = m_es*y_es + n_es*y_es0
                print("loss:\tes\tes0\tcombo\tpcc:\tes\tes0\tcombo")
                print(compute_loss_pcc(m_es,n_es,\
                                        y_es.cpu().detach().squeeze().numpy(),\
                                        y_es0.cpu().detach().squeeze().numpy(),\
                                        batch_y.cpu().detach().squeeze().numpy()))
                f.write("loss:\tes_combo\telec_combo\tfinal\tpcc:\tes_combo\telec_combo\tfnial\n")
                f.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(\
                        *compute_loss_pcc(m_es,n_es,\
                                        y_es.cpu().detach().squeeze().numpy(),\
                                        y_es0.cpu().detach().squeeze().numpy(),\
                                        batch_y.cpu().detach().squeeze().numpy())))
                #elec & elec0
                m_elec,n_elec = opt_coef(y_elec.cpu().detach().squeeze().numpy(),\
                                            y_elec0.cpu().detach().squeeze().numpy(),\
                                            batch_y.cpu().detach().numpy())
                y_combo_elec = m_elec*y_elec + n_elec*y_elec0
                print("loss:\telec\telec0\tcombo\tpcc:\telec\telec0\tcombo")
                print(compute_loss_pcc(m_elec,n_elec,\
                                        y_elec.cpu().detach().squeeze().numpy(),\
                                        y_elec0.cpu().detach().squeeze().numpy(),\
                                        batch_y.cpu().detach().squeeze().numpy()))
                f.write("loss:\tes_combo\telec_combo\tfinal\tpcc:\tes_combo\telec_combo\tfnial\n")
                f.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(\
                        *compute_loss_pcc(m_elec,n_elec,\
                                    y_elec.cpu().detach().squeeze().numpy(),\
                                    y_elec0.cpu().detach().squeeze().numpy(),\
                                    batch_y.cpu().detach().squeeze().numpy())))
                #es_combo & elec_combo
                m,n = opt_coef(y_combo_es.cpu().detach().squeeze().numpy(),\
                                    y_combo_elec.cpu().detach().squeeze().numpy(),\
                                    batch_y.cpu().detach().numpy())
                y_combo = m*y_combo_es + n*y_combo_elec
                print("loss:\tes_combo\telec_combo\tfinal\tpcc:\tes_combo\telec_combo\tfnial")
                print(compute_loss_pcc(m,n,\
                                        y_combo_es.cpu().detach().squeeze().numpy(),\
                                        y_combo_elec.cpu().detach().squeeze().numpy(),\
                                        batch_y.cpu().detach().squeeze().numpy()))
                f.write("loss:\tes_combo\telec_combo\tfinal\tpcc:\tes_combo\telec_combo\tfnial\n")
                f.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(\
                        *compute_loss_pcc(m,n,\
                                        y_combo_es.cpu().detach().squeeze().numpy(),\
                                        y_combo_elec.cpu().detach().squeeze().numpy(),\
                                        batch_y.cpu().detach().squeeze().numpy())))

                loss = torch.sqrt(loss_fn(y_combo.squeeze(),batch_y))
                loss.backward()
                optimizer.step()

                print('epoch %d\tstep %3d\tloss %12.5e\n'%(epoch,step,loss.item()))
                f.write('epoch %d\tstep %3d\tloss %12.5e\n'%(epoch,step,loss.item()))

            net.eval()
            y_es_pred, y_elec_pred, y_es0_pred, y_elec0_pred = \
            net(torch.from_numpy(es_test).float().cuda(),\
                torch.from_numpy(elec_test).float().cuda(),\
                torch.from_numpy(es0_test).float().cuda(),\
                torch.from_numpy(elec0_test).float().cuda())

            y_combo_es_pred = (m_es*y_es_pred + n_es*y_es0_pred).cpu().detach().squeeze().numpy()
            y_combo_elec_pred = (m_elec*y_elec_pred + n_elec*y_elec0_pred).cpu().detach().squeeze().numpy()
            y_combo_pred = (m*y_combo_es_pred + n*y_combo_elec_pred)

            m_the, n_the = opt_coef(y_combo_es_pred,y_combo_elec_pred,y_test)
            print(f"theoretical m {m_the} n {m_the}")
            f.write(f"theoretical m {m_the} n {m_the}\n")
            print(f"optimized m {m} n {n}")
            f.write(f"optimized m {m} n {n}\n")

            mse_es = mean_squared_error(y_combo_es_pred,y_test)
            mse_elec = mean_squared_error(y_combo_elec_pred,y_test)
            mse = mean_squared_error(y_combo_pred,y_test)
            print('%d test RMSE: %8.4f %8.4f %8.4f\n'%(epoch,sqrt(mse_es),sqrt(mse_elec),sqrt(mse)))
            f.write('%d test RMSE: %8.4f %8.4f %8.4f\n'%(epoch,sqrt(mse_es),sqrt(mse_elec),sqrt(mse)))

            pearcorr_es = pearsonr(y_combo_es_pred, y_test)[0]
            pearcorr_elec = pearsonr(y_combo_elec_pred, y_test)[0]
            pearcorr = pearsonr(y_combo_pred, y_test)[0]

            print('%d test pearsonR: %8.4f %8.4f %8.4f\n'%(epoch,pearcorr_es,pearcorr_elec,pearcorr))
            f.write('%d test pearsonR: %8.4f %8.4f %8.4f\n'%(epoch,pearcorr_es,pearcorr_elec,pearcorr))
            # save best model to best_model_dir
            if max_pearson < pearcorr:
                max_pearson = pearcorr
                print(f"save model at epoch {epoch} with loss {sqrt(mse):.4f} pearcorr {pearcorr:.4f}")
                f.write(f"save model at epoch {epoch} with loss {sqrt(mse):.4f} pearcorr {pearcorr:.4f}")
                # torch.save(net.state_dict(),'models/'+f'es0_model.pth')

def valid_info(net,X_train,X_test,y_train,y_test,log_dir,log_file,epoch_num=5,batch_size=512,lr=3e-4,weight_decay=1e-9,upper_dim=False,cuda=True):
    if upper_dim:
        X_train = X_train[:,np.newaxis,:,:]
        X_test = X_test[:,np.newaxis,:,:]
    max_pearson = 0
    best_model_path = 0

    loss_fn = torch.nn.MSELoss(reduction='mean')
    train_data = Data.TensorDataset(torch.from_numpy(X_train).float(),\
                                    torch.from_numpy(y_train).float())
    train_loader = Data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=True)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr, weight_decay=weight_decay)

    for epoch in range(epoch_num):
        net.zero_grad()
        net.train()
        for step,(batch_x,batch_y) in enumerate(train_loader):
            if cuda:
                batch_x=batch_x.cuda()
                batch_y=batch_y.cuda()

            optimizer.zero_grad()

            batch_pred = net(batch_x)
            loss = torch.sqrt(loss_fn(batch_pred.squeeze(),batch_y))
            loss.backward()
            optimizer.step()
            print('validation %4d %3d %3f\n'%(epoch,step,loss.item()))

        net.eval()
        if cuda:
            y_pred=net(torch.from_numpy(X_test).float().cuda())
            y_pred=y_pred.cpu().detach().squeeze().numpy()
        else:
            y_pred=net(torch.from_numpy(X_test).float()).detach().squeeze().numpy()

        mse = mean_squared_error(y_test,y_pred)
        print('%4d validation RMSE: %8.4f\n'%(epoch,sqrt(mse)))
        # f.write('%4d validation RMSE: %8.4f\n'%(epoch,sqrt(mse)))

        pearcorr = sp.stats.pearsonr(y_test, y_pred)
        print('%4d validation pearsonR: %8.4f\n'%(epoch,pearcorr[0]))
        # f.write('%4d validation pearsonR: %8.4f\n'%(epoch,pearcorr[0]))

        # save best model to best_model_dir
        if max_pearson < pearcorr[0]:
            max_pearson = pearcorr[0]
            if os.path.exists(f"{log_dir}/valid_models/{best_model_path}"):
                os.remove(f"{log_dir}/valid_models/{best_model_path}")
            best_model_path = log_file.split('/')[-1].replace('.txt',f'_epoch_{epoch}_pcc_{pearcorr[0]:.4f}.pth')
            print(f"save valid model at epoch {epoch} with loss {sqrt(mse):.4f} pearcorr {pearcorr[0]:.4f}\n")
            # f.write(f"save valid model at epoch {epoch} with loss {sqrt(mse):.4f} pearcorr {pearcorr[0]:.4f}")
            torch.save(net.state_dict(), f"{log_dir}/valid_models/" + best_model_path)
        
    net.load_state_dict(torch.load(f"{log_dir}/valid_models/" + best_model_path))
    return net, max_pearson

def valid_info_classifier(net,X_train,X_test,y_train,y_test,log_dir,log_file,epoch_num=5,batch_size=512,lr=3e-4,weight_decay=1e-9,upper_dim=False,cuda=True):
    if upper_dim:
        X_train = X_train[:,np.newaxis,:,:]
        X_test = X_test[:,np.newaxis,:,:]
        
    y_test = torch.from_numpy(y_test)

    max_acc = 0
    best_model_path = 0

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    train_data = Data.TensorDataset(torch.from_numpy(X_train).float(),\
                                    torch.from_numpy(y_train).float())
    train_loader = Data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=True)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr, weight_decay=weight_decay)

    for epoch in range(epoch_num):
        net.zero_grad()
        net.train()
        for step,(batch_x,batch_y) in enumerate(train_loader):
            if cuda:
                batch_x=batch_x.cuda()
                batch_y=batch_y.cuda()

            optimizer.zero_grad()
            
            batch_pred = net(batch_x)
            loss = loss_fn(batch_pred.squeeze(),batch_y.long())
            loss.backward()
            optimizer.step()
            print('validation ce %4d %3d %12.5e\n'%(epoch,step,loss.item()))

        net.eval()
        if cuda:
            y_pred=net(torch.from_numpy(X_test).cuda())
            y_pred=y_pred.cpu().detach().squeeze()
        else:
            y_pred=net(torch.from_numpy(X_test)).detach().squeeze()

        ce = loss_fn(y_pred, y_test)
        print('%4d validation CE: %8.4f\n'%(epoch,ce))
        
        acc = (y_pred.argmax(dim=1) == y_test).float().sum()/y_test.shape[0] 
        print('%4d validation accuracy: %8.4f\n'%(epoch,acc))

        y_test_reg = y_test.numpy()+2.5
        y_pred_reg = y_pred.argmax(dim=1).numpy()+2.5
        
        mse = mean_squared_error(y_test_reg,y_pred_reg)
        print('%4d validation RMSE: %8.4f\n'%(epoch,sqrt(mse)))

        pearcorr = sp.stats.pearsonr(y_test_reg, y_pred_reg)[0]
        print('%4d validation pearsonR: %8.4f\n'%(epoch,pearcorr))        

        # save best model to best_model_dir
        if max_acc < acc:
            max_acc = acc
            if os.path.exists(f"{log_dir}/cls_models/{best_model_path}"):
                os.remove(f"{log_dir}/cls_models/{best_model_path}")
            best_model_path = log_file.split('/')[-1].replace('.txt',f'_epoch_{epoch}_acc_{acc:.4f}.pth')
            print(f"save valid model at epoch {epoch} with loss {ce:.4f} acc {acc:.4f}\n")
            torch.save(net.state_dict(), f"{log_dir}/cls_models/" + best_model_path)
        
    net.load_state_dict(torch.load(f"{log_dir}/cls_models/" + best_model_path))
    return net, max_acc


# train_info_merge(net_4,\
#                 (X_es_train, X_elec_train, X_es0_train, X_elec0_train),\
#                 (X_es_test, X_elec_test, X_es0_test, X_elec0_test),\
#                 y_train, y_test, 'log/Net_merge_1030.txt')

# train_info_Adamerge(net5,\
#                 (X_es_train, X_elec_train, X_es0_train, X_elec0_train),\
#                 (X_es_test, X_elec_test, X_es0_test, X_elec0_test),\
#                 y_train, y_test, 'log/Net5_1103.txt')

# train_info_Optmerge(net5,\
#                 (X_es_train, X_elec_train, X_es0_train, X_elec0_train),\
#                 (X_es_test, X_elec_test, X_es0_test, X_elec0_test),\
#                 y_train, y_test, 'log/AdaNet_0226.txt')


################## transfer 

def epoch_pcc(model, X, y):
    return pearsonr(model.predict(X).squeeze(),y.squeeze())[0]

pcc_train = EpochScoring(name='pearson_train',scoring= epoch_pcc,lower_is_better=False ,on_train=True) #
pcc_test = EpochScoring(name='pearson_test',scoring= epoch_pcc,lower_is_better=False) #,on_train=True
lrs = LRScheduler(policy=ReduceLROnPlateau, mode='max', factor=0.5, patience = 5, threshold = 1e-3)
# lrs = LRScheduler(policy=CyclicLR, base_lr=1e-3, max_lr=5e-6, step_size_up=150)
# lrs = LRScheduler(policy='WarmRestartLR', min_lr=3e-7, max_lr=8e-4, base_period=150, period_mult=2, last_epoch=1)

class ComboLoss(torch.nn.Module):
    
    def __init__(self):
        super(ComboLoss,self).__init__()
        
    def forward(self,y1,y2,lamb1=0,lamb2=1): # 0.2 0.8 0 1 0.5 0.5 0.6 0.4
        corr = pearsonr_torch(y1.squeeze(), y2.squeeze())
        loss_pearson = torch.FloatTensor(np.ones(1)).cuda() - corr
        # loss_mse = (y2 - y1).norm(2).pow(2)
        loss_rmse = torch.sqrt(nn.MSELoss()(y1, y2))

        loss = torch.sum(lamb1 * loss_pearson) + torch.sum(lamb2 * loss_rmse)
        return loss

class InputShapeSetter_Res(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        net.set_params(module__combo=X.shape[1])

class InputShapeSetter_Dense_1d(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        net.set_params(module__combo=X.shape[1])
        net.set_params(module__flatten=3072)

class InputShapeSetter_Dense_2d(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        net.set_params(module__combo=X.shape[1])
        net.set_params(module__flatten=9216)

class InputShapeSetter_Trans(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        net.set_params(module__image_size=100)
        net.set_params(module__patch_size=10)
        net.set_params(module__num_classes=1)
        net.set_params(module__channels=X.shape[1])
        net.set_params(module__dim=64)
        net.set_params(module__depth=6)
        net.set_params(module__heads=8)
        net.set_params(module__mlp_dim=128)
        # net.set_params(optimizr__initial_lr=1e-3)
