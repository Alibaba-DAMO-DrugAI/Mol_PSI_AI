
import torch
import torch.nn.functional as F
from torch import nn
from vit_pytorch.vit_pytorch import ViT
from vit_pytorch.efficient import ViT as EfficientViT

from einops import rearrange, repeat

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from einops import rearrange
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime
from glob import glob
import scipy as sp


class ourData(Dataset):
    def __init__(self,x,gt):
        self.x=x
        self.gt=gt

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return [self.x[idx],self.gt[idx]]

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
    def __init__(self, dim, hidden_dim, dropout = 0.0):
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
# helpers

def exists(val):
    return val is not None

# classes

class DistillMixin:
    def forward(self, img, distill_token = None, mask = None):
        p, distilling = self.patch_size, exists(distill_token)

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n + 1)]

        if distilling:
            distill_tokens = repeat(distill_token, '() n d -> b n d', b = b)
            x = torch.cat((x, distill_tokens), dim = 1)

        x = self._attend(x, mask)

        if distilling:
            x, distill_tokens = x[:, :-1], x[:, -1]

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        out = self.mlp_head(x)

        if distilling:
            return out, distill_tokens

        return out

class DistillableViT(DistillMixin, ViT):
    def __init__(self, *args, **kwargs):
        super(DistillableViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = ViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x, mask):
        x = self.dropout(x)
        x = self.transformer(x, mask)
        return x

class DistillableEfficientViT(DistillMixin, EfficientViT):
    def __init__(self, *args, **kwargs):
        super(DistillableEfficientViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = EfficientViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x, mask):
        return self.transformer(x)

# knowledge distillation wrapper

class DistillWrapper(nn.Module):
    def __init__(
        self,
        *,
        teacher,
        student,
        temperature = 1.,
        alpha = 0.5
    ):
        super().__init__()
        assert (isinstance(student, (DistillableViT, DistillableEfficientViT))) , 'student must be a vision transformer'

        self.teacher = teacher
        self.student = student

        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha

        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))

        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, labels, temperature = None, alpha = None, **kwargs):
        b, *_ = img.shape
        alpha = alpha if exists(alpha) else self.alpha
        T = temperature if exists(temperature) else self.temperature

        with torch.no_grad():
            teacher_logits = self.teacher(img)

        student_logits, distill_tokens = self.student(img, distill_token = self.distillation_token, **kwargs)
        distill_logits = self.distill_mlp(distill_tokens)

        loss = F.cross_entropy(student_logits, labels)

        distill_loss = F.kl_div(
            F.log_softmax(distill_logits / T, dim = -1),
            F.softmax(teacher_logits / T, dim = -1).detach(),
        reduction = 'batchmean')

        distill_loss *= T ** 2

        return loss * alpha + distill_loss * (1 - alpha)

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)

            # nn.Linear(dim, mlp_dim),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(mlp_dim, num_classes)
            
        )

    def forward(self, img, mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



##----------------------

y_train = np.load('PDB_data/y_train_2016.npy').astype('float32')
y_train = y_train.reshape(-1,1)
y_test = np.load('PDB_data/y_test_2016.npy').astype('float32')
y_test = y_test.reshape(-1,1)

y = np.append(y_train, y_test,axis=0)

##X_es = np.load('Temp_tensor_Xend50/2007_normDist_sigma_1.5/tensor_2007_norm.npy')
##X_es_train = X_es[:len(y_train),:,:]
##X_es_test = X_es[len(y_train):,:,:]


##X_es_zero = np.load('Temp_tensor_28Oct/tensor_2007_norm_numzero.npy')
##X_es0_train = X_es_zero[:len(y_train),np.newaxis,:,:]
##X_es0_test = X_es_zero[len(y_train):,np.newaxis,:,:]

######
# X_elec = np.load('Temp_tensor_Xend50/2007_elecDist_sigma_1.5/tensor_2007_elec.npy')
X_elec = np.load('Temp_tensor_Nov1/2016_elecDist_sigma_1.5/tensor_2016_elec.npy').astype('float32')
X_elec_train = X_elec[:len(y_train),:,:]
X_elec_test = X_elec[len(y_train):,:,:]

# X_norm = np.load('PDB_data/tensor_2016_norm.npy')
# X_norm = X_norm.astype('float32')
# X_norm_train = X_norm[:len(gt_train),:,:,:]
# X_norm_test = X_norm[len(gt_train):,:,:,:]
# print(torch.Tensor(X_norm_train).size(),torch.Tensor(X_norm_test).size())

##X_elec_zero = np.load('Temp_tensor_28Oct/tensor_2007_elec_numzero.npy')
##X_elec0_train = X_elec_zero[:len(y_train),np.newaxis,:,:]
##X_elec0_test = X_elec_zero[len(y_train):,np.newaxis,:,:]
##print(torch.Tensor(X_elec0_train).size())

train_set=ourData(X_elec_train, y_train)
test_set=ourData(X_elec_test, y_test)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=500, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, shuffle=False)

# def train(model, optimizer, data_loader, loss_history,device):
def train(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()
    for i, (data, target) in enumerate(data_loader):
#         data=data.to(device)
#         target=target.to(device)
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()

        output=model(data)
        pred = output.cpu().detach().numpy().reshape(-1,)
        actual = target.cpu().detach().numpy().reshape(-1,)
        loss = F.mse_loss(output,target)
        loss = torch.sqrt(loss)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                    '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())
            
# def evaluate(model, data_loader, loss_history,device,f):
def evaluate(model, data_loader, loss_history,f):
    model.eval()

    total_samples = len(data_loader.dataset)
    total_maeloss = 0
    total_loss = []

    with torch.no_grad():
        for data, target in data_loader:
            #output = F.log_softmax(model(data), dim=1)
#             data=data.to(device)
#             target=target.to(device)
            data = data.cuda()
            target = target.cuda()
            output=model(data)
            loss = F.mse_loss(output, target,reduction='mean')
            loss = torch.sqrt(loss)
            pred = output.cpu().detach().numpy().reshape(-1,)
            actual = target.cpu().detach().numpy().reshape(-1,)
            #print(pred,'\n',actual)
            #_, pred = torch.max(output, dim=1)

            total_loss.append(loss.item())
            #maeloss=F.l1_loss(output, target)
            maeloss=np.sum(np.abs(pred-actual))
            total_maeloss+=maeloss
            #correct_samples += pred.eq(target).sum()

    avg_loss=np.mean(np.array(total_loss))
    avg_maeloss = total_maeloss / total_samples

    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
            '  MAE loss:' + '{:5}'.format(avg_maeloss) +'\n')
    f.write('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
            '  MAE loss:' + '{:5}'.format(avg_maeloss) +'\n')
    pearcorr = sp.stats.pearsonr(actual, pred)
    print('Test distance pearsonR: %8.4f\n'%(pearcorr[0]))
    f.write('Test distance pearsonR: %8.4f\n'%(pearcorr[0]))
    return pearcorr[0]    

def evaluate_train(model, data_loader, loss_history,device):
    model.eval()
    total_samples = len(data_loader.dataset)
    total_maeloss = 0
    total_loss = []
    i=0
    with torch.no_grad():
        for data, target in data_loader:
            #output = F.log_softmax(model(data), dim=1)
            data=data.to(device)
            target=target.to(device)
            output=model(data)
            loss = F.mse_loss(output, target,reduction='mean')
            loss = torch.sqrt(loss)
            pred = output.cpu().detach().numpy().reshape(-1,)
            actual = target.cpu().detach().numpy().reshape(-1,)

            total_loss.append(loss.item())
            #maeloss=F.l1_loss(output, target)
            maeloss=np.sum(np.abs(pred-actual))
            total_maeloss+=maeloss
            #correct_samples += pred.eq(target).sum()
            if i==0:
                continue

N_EPOCHS = 30000

log_file = 'log/log_2D_Transformer_2021_2_24.log'
# model = ImageTransformer(image_size=100, patch_size=25, num_classes=1, channels=2,
#             dim=64, depth=6, heads=8, mlp_dim=128)

model = ViT(image_size=100, patch_size=10, num_classes=1, dim=64, depth=6, heads=8, mlp_dim=128, channels=50)
# patch_size=25  dim=64  channels=36

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for param in model.parameters():
    param.requires_grad = True
model=torch.nn.DataParallel(model)
# model=model.to(device)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) # 0.003)

bestPearson = -1
bestEpoch = 0
PATH = log_file.replace('log','models').replace('.log','.pth')
train_loss_history, test_loss_history = [], []
with open(log_file, 'w') as f:
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        f.write('Epoch: %6d\n'%(epoch))
        start_time = time.time()
#         train(model, optimizer, train_loader, train_loss_history,device)
        train(model, optimizer, train_loader, train_loss_history)
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        #evaluate_train(model,train_loader,[],device)
#         evaluate(model, test_loader, test_loss_history,device,f)
        pearsonC = evaluate(model, test_loader, test_loss_history,f)
        if pearsonC > bestPearson:
            bestPearson = pearsonC
            bestEpoch = epoch
            torch.save(model,PATH)
        print('best Pearson Correlation is %8.4f at epoch %6d'%(bestPearson,bestEpoch))
        f.write('best Pearson Correlation is %8.4f at epoch %6d'%(bestPearson,bestEpoch)) 

print('Finish.')

#PATH = "./models/gene_model_test.pt" # Use your own path
####PATH = log_file.replace('log','models').replace('.log','.pth')
#torch.save(model.state_dict(), PATH)
####torch.save(model,PATH)

# =============================================================================
# model = ViT()
# model.load_state_dict(torch.load(PATH))
# model.eval()            
# =============================================================================   

