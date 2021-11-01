
import time
from math import sqrt

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torchvision.models as models
from einops import rearrange
from sklearn.metrics import mean_squared_error
from torch import nn
from torchsummaryX import summary
# from DenseNet_elec0_v2 import Dense_back2484


# m = Dense_back2484()
# ms = summary(m, torch.zeros(2,36,100,100))

y_train = np.load('./PDB_withNTU/PDB_data/y_train_2016.npy') # normalize(
y_test = np.load('./PDB_withNTU/PDB_data/y_test_2016.npy') #normalize(

X_es = np.load('./Temp_tensor_Dec9/2016_normDist_sigma_1.5/tensor_2016_norm.npy')
X_es_train = X_es[:len(y_train),:,:]
X_es_test = X_es[len(y_train):,:,:]

X_elec = np.load('./Temp_tensor_Dec9/2016_elecDist_sigma_1.5/tensor_2016_elec.npy')
X_elec_train = X_elec[:len(y_train),:,:]
X_elec_telect = X_elec[len(y_train):,:,:]

train_data = Data.TensorDataset(torch.from_numpy(X_es_train).float(),\
                                torch.from_numpy(y_train).float())

test_data = Data.TensorDataset(torch.from_numpy(X_es_test).float(),\
                                torch.from_numpy(y_test).float())

train_loader = Data.DataLoader(dataset=train_data,batch_size=4000,shuffle=True,num_workers=4,drop_last=False)
# test_loader = Data.DataLoader(dataset=train_data,batch_size=350,shuffle=True,num_workers=4,drop_last=False)

def train(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        data = data.cuda()
        target = target.cuda()
        # output = F.log_softmax(model(data), dim=1)
        output = model(data)
        loss_fn = torch.nn.MSELoss(reduction='mean')
        loss = torch.sqrt(loss_fn(output, target))
        loss.backward()
        optimizer.step()

        pearcorr = scipy.stats.pearsonr(output.cpu().detach().squeeze().numpy(),\
                                        target.cpu().detach().squeeze().numpy())[0]
        
        print('Batch average rmse: ' + '{:.4f}'.format(loss.item()) + 
                '\tBatch pcc: ' + '{:.4f}'.format(pearcorr))

        # if i % 100 == 0:
        #     print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
        #           ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
        #           '{:6.4f}'.format(loss.item()))
        loss_history.append(loss.item())
            
def evaluate(model, X_test, y_test, loss_history):
    model.eval()

    with torch.no_grad():
        y_pred=model(torch.from_numpy(X_test).float().cuda())
        y_pred=y_pred.cpu().detach().squeeze().numpy()        
        y_ground = y_test
        
        test_rmse = sqrt(mean_squared_error(y_pred, y_ground))
        test_pearcorr = scipy.stats.pearsonr(y_pred, y_ground)[0]
        
        print('test rmse: ' + '{:.4f}'.format(test_rmse) + 
            '\ttest pcc: ' + '{:.4f}'.format(test_pearcorr) + '\n')
        loss_history.append(test_rmse)
        

N_EPOCHS = 3000

model = ImageTransformer(image_size=100, patch_size=50, num_classes=1, channels=36,
            dim=64, depth=6, heads=8, mlp_dim=128).cuda()

ms2 = summary(model, torch.zeros(2,36,100,100).cuda())


# s = summary(model, torch.zeros(2,36,100,100).cuda())
optimizer = torch.optim.Adam(model.parameters(), lr=0.000007)


train_loss_history, test_loss_history = [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    start_time = time.time()
    train(model, optimizer, train_loader, train_loss_history)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds\n')
    evaluate(model, X_es_test, y_test, test_loss_history)




# print('Execution time')
# PATH = ".\ViTnet_Cifar10_4x4_aug_1.pt" # Use your own path
# torch.save(model.state_dict(), PATH)


# =============================================================================
# model = ViT()
# model.load_state_dict(torch.load(PATH))
# model.eval()            
# =============================================================================
