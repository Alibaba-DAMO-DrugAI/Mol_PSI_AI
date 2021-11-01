
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange
from torch import nn


def reg_model_2_cls(model, method, num_cls):
    if method != 'Transformer':
        dim_org = model.regression._modules['2'].state_dict()['weight'].shape[1]
        model.regression._modules['2'] = nn.Linear(dim_org, num_cls)
    
    if method == 'Transformer':    
        dim_org = model.nn2.state_dict()['weight'].shape[1]
        model.nn2 = nn.Linear(dim_org, num_cls)
    return model


class SimpleCNN(nn.Module):
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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Res_back(nn.Module):
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

class Res_cls_sampler(nn.Module):
    def __init__(self, num_bins, load_weights=True):
        super(Res_cls_sampler,self).__init__()
        self.cls_res = reg_model_2_cls(Res_back(50), 'Resnet', num_bins)
        self.bins = np.linspace(0, num_bins, num_bins+1, endpoint=True)
        if load_weights:
            self.cls_res.load_state_dict(torch.load(f"./Backup_model/elec_res_pred_bins{num_bins}.pt"))
    
    def forward(self, X):
        with torch.no_grad():
            pred = self.cls_res.backbone(X)
            pred = pred.view(-1,self.cls_res.num_flat_features(pred))
        pred = self.cls_res.regression(pred)
        pred_ind_np = pred.argmax(axis=1).detach().cpu().numpy()
        hist_pred = np.histogram(pred_ind_np, self.bins)

        return pred, hist_pred

class Dense_back(nn.Module):
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
        x = x.contiguous().view(-1,self.num_flat_features(x))
        x = self.regression(x)
        return x

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

