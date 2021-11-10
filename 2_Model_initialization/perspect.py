from math import sqrt

import numpy as np
import scipy as sp
import torch
import torch.utils.data as Data
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from args import *
from dataset import tensors_by_year_type
from models import *
from utils import sample_by_int_bins


def pred_bins_model(num_bins, log_dir):
    ## ref bins
    # bins_30 = np.linspace(2, 12, 30+1, endpoint=True)
    # bins_50 = np.linspace(2, 12, 50+1, endpoint=True)
    # bins_75 = np.linspace(2, 12, 75+1, endpoint=True)
    # bins_90 = np.linspace(2, 12, 90+1, endpoint=True)
    # bins_125 = np.linspace(2, 12, 125+1, endpoint=True)
    # bins_150 = np.linspace(2, 12, 150+1, endpoint=True)

    ## ref interpolations
    # y_interpo_30 = np.array([0.5*(bins_30[i]+bins_30[i+1]) for i in range(len(bins_30)-1)])
    # y_interpo_50 = np.array([0.5*(bins_50[i]+bins_50[i+1]) for i in range(len(bins_50)-1)])
    # y_interpo_75 = np.array([0.5*(bins_75[i]+bins_75[i+1]) for i in range(len(bins_75)-1)])
    # y_interpo_90 = np.array([0.5*(bins_90[i]+bins_90[i+1]) for i in range(len(bins_90)-1)])
    # y_interpo_125 = np.array([0.5*(bins_125[i]+bins_125[i+1]) for i in range(len(bins_125)-1)])
    # y_interpo_150 = np.array([0.5*(bins_150[x]+bins_150[x+1]) for x in range(len(bins_150)-1)])

    arg_bins = np.linspace(2, 12, num_bins+1, endpoint=True)

    # load 2016 for predicting bins
    X_train_, X_test, y_train_, y_test, y_train_id, y_test_id = tensors_by_year_type('2016', 1.5, 'elec', with_id = True)
    y_ind_train_ =  np.digitize(y_train_, arg_bins)-1 #-1
    y_ind_test = np.digitize(y_test, arg_bins)-1 #-1

    # load 2013 for checking 
    *_, y_test_id_2013 = tensors_by_year_type('2013', None, 'es0', with_id = True)
    print('exclude intersection between train_2016 and core_2013')
    y_train_minus_test_2013 = np.array([x for x,x_id in zip(y_train_,y_train_id) if x_id not in y_test_id_2013])
    y_train_minus_test_2013_id = np.array([x_id for x,x_id in zip(y_train_,y_train_id) if x_id not in y_test_id_2013])
    print('exclude',y_train_id.shape[0]-y_train_minus_test_2013_id.shape[0],'IDs\n')

    # model data opt fine tune
    data = Data.TensorDataset(torch.from_numpy(X_train_).float(), torch.from_numpy(y_ind_train_))
    loader = Data.DataLoader(dataset=data,batch_size=256,shuffle=False,num_workers=0,drop_last=False)


    cls_model = Res_cls_sampler(num_bins,1).to(device)
    tmp_epoches = 5

    optimizer = torch.optim.Adam(cls_model.parameters(),lr=cls_model.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(tmp_epoches):
        cls_model.zero_grad()
        cls_model.train()
        print(f'Fine tune bins prediction epoch:{epoch+1}/5')
        for step,(batch_x,batch_y) in tqdm(enumerate(loader)):
            optimizer.zero_grad()
            batch_x=batch_x.to(device)
            batch_y=batch_y.to(device)
            batch_pred = cls_model(batch_x)[0] #,(batch_hist,_)
            loss = loss_fn(batch_pred, batch_y)
            loss.backward()
            optimizer.step()
            
    print('Finished fune tune\n')

    # predict bins
    cls_model.eval()
    pred_tensor, (hist_pred,_) = cls_model(torch.from_numpy(X_test).float().to(device))
    pred_ind = pred_tensor.argmax(1).detach().cpu().numpy()
    hist_pred, _ = np.histogram(pred_ind, np.linspace(0, num_bins, num_bins+1, endpoint=True))

    # samples based on predicted bins 
    np.random.seed(args.seed)
    n_splits = int(1/args.valid_size)
    valid_test_ratio = y_train_.shape[0]/y_test.shape[0]/n_splits

    n_samples_by_bins = np.around(valid_test_ratio*hist_pred,0).astype(int)
    print(f'sampling from {n_samples_by_bins.shape[0]} bins (Should be {num_bins})')
    assert n_samples_by_bins.shape[0] == num_bins

    valid_index = np.array([sample_by_int_bins(y_train_minus_test_2013,arg_bins,x,n) for x,n in zip(range(num_bins),n_samples_by_bins)],dtype=object)
    valid_y = np.array([y_train_minus_test_2013[x] for x in valid_index],dtype=object)
    valid_id = np.hstack([y_train_minus_test_2013_id[x] for x in valid_index])

    print('checking intersection between 2016 train and 2013 test', np.intersect1d(valid_id,y_test_id_2013))
    if log_dir:
        torch.save(cls_model.state_dict(), f'./{log_dir}/cls_models/Resnet_elec_cls_bins{num_bins}.pt')
        np.save(f'./{log_dir}/cls_models/y_2016_split_by_pred_test_bins{num_bins}_checked.npy',valid_index)
    return valid_index

def transfer_model(method, t, load_tranfer=True):
    if t == 'es' and method == 'Densenet':
        model = Dense_back(36)
        if load_tranfer:
            model.load_state_dict(torch.load("./Backup_model/es_dense.pt"))
    elif t == 'es' and method == 'Resnet':
        model = Res_back(36)
        if load_tranfer:       
            model.load_state_dict(torch.load("./Backup_model/es_res.pt"))        
    elif t == 'es' and method == 'Transformer':
        model = ImageTransformer(channels=36)
        if load_tranfer:
            model.load_state_dict(torch.load("./Backup_model/es_trans.pt")) 
        
    elif t == 'elec' and method == 'Densenet':
        model = Dense_back(50)
        if load_tranfer:
            model.load_state_dict(torch.load("./Backup_model/elec_dense.pt"))  
    elif t == 'elec' and method == 'Resnet':
        model = Res_back(50)
        if load_tranfer:
            model.load_state_dict(torch.load("./Backup_model/elec_res.pt"))  
    elif t == 'elec' and method == 'Transformer':
        model = ImageTransformer(channels=50)
        if load_tranfer:
            model.load_state_dict(torch.load("./Backup_model/elec_trans.pt"))  
        
    elif t == 'es0' and method == 'Densenet':
        model = Dense_back(1,3072)
        if load_tranfer:
            model.load_state_dict(torch.load("./Backup_model/es0_dense.pt"))  
    elif t == 'es0' and method == 'Resnet':
        model = Res_back(1)
        if load_tranfer:
            model.load_state_dict(torch.load("./Backup_model/es0_res.pt"))
    
    elif t == 'elec0' and method == 'Densenet':
        model = Dense_back(1,3072)
        if load_tranfer:
            model.load_state_dict(torch.load("./Backup_model/elec0_dense.pt"))  
    elif t == 'elec0' and method == 'Resnet':
        model = Res_back(1)
        if load_tranfer:
            model.load_state_dict(torch.load("./Backup_model/elec0_res.pt")) 
    else:
        raise Exception("No such a model")
    
    model = model.cuda() if args.cuda else model
    print(f"load model {method} {t} load_tranfer_paras is {load_tranfer}")
    return model

def valid_model(method, t, sigma, log_dir):
    if t == 'es' and method == 'Densenet':
        model = Dense_back(36)

    elif t == 'es' and method == 'Resnet':
        model = Res_back(36)
    
    elif t == 'es' and method == 'Transformer':
        model = ImageTransformer(channels=36)

    elif t == 'elec' and method == 'Densenet':
        model = Dense_back(50)

    elif t == 'elec' and method == 'Resnet':
        model = Res_back(50)

    elif t == 'elec' and method == 'Transformer':
        model = ImageTransformer(channels=50)
        
    elif t == 'es0' and method == 'Densenet':
        model = Dense_back(1,3072)

    elif t == 'es0' and method == 'Resnet':
        model = Res_back(1)

    elif t == 'elec0' and method == 'Densenet':
        model = Dense_back(1,3072)

    elif t == 'elec0' and method == 'Resnet':
        model = Res_back(1)

    else:
        raise Exception("No such a model")
    
    model = model.to(device)
    print(f"load initial model {method} {t}\n")

    print(f"Transfer on (train_ = train + validation) {2016} {t} {str(sigma) if sigma else ''}:")
    X_train_, X_test, y_train_, y_test = tensors_by_year_type('2016', sigma, t)
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=args.valid_size, random_state=args.valid_seed)

    n_splits = int(1/args.valid_size)
    num_bins = args.num_bins
    arg_bins = np.linspace(2, 12, num_bins+1, endpoint=True)

    if args.valid_method == 'exist_model':
        model.load_state_dict(torch.load(f'{args.valid_path}'))
        print(f"load exist model {args.valid_path} successfully!")
        return model
        
    if args.valid_method == 'bins' or args.valid_method == 'random':
        if args.valid_method == 'bins':
            y_train_ind = np.digitize(y_train_, arg_bins)
            split_folder = StratifiedKFold(n_splits=n_splits, random_state=args.valid_seed, shuffle=True)
            enumerater = enumerate(split_folder.split(X_train_, y_train_ind))

        if args.valid_method == 'random':
            split_folder = KFold(n_splits=n_splits, random_state=args.valid_seed, shuffle=True)
            enumerater = enumerate(split_folder.split(X_train_, y_train_))

        valid_pearson = 0
        best_model = 0
        for j, (train_index, valid_index) in enumerater: # y_bins y_train_ y_train_ind
            X_valid = X_train_[valid_index]
            X_train = X_train_[train_index]
            y_valid = y_train_[valid_index]
            y_train = y_train_[train_index]

            hist_train, _ = np.histogram(y_train, bins=arg_bins)
            hist_valid, _ = np.histogram(y_valid, bins=arg_bins) 
            hist_ratio = np.around(hist_train/hist_valid,3)

            print(f'bins for train and valid: {arg_bins}, i.e. 2≤?<3')
            print('hist_train: number of samples in bins',hist_train)
            print('hist_valid: number of samples in bins',hist_valid)
            print(f'hist_ratio: sample size ratio in bins (train/valid) n_split = {n_splits}\n{hist_ratio}')
            valid_model, max_pearson = valid_info(model, X_train, X_valid, y_train, y_valid, log_dir,\
            f"{method}_{t}{'_'+str(sigma) if sigma else ''}_{2016}_valid_fold_{j}.txt",\
            epoch_num=args.valid_num_epoch, batch_size=args.valid_batch_size, lr=args.valid_lr,)

            if max_pearson > valid_pearson:
                valid_pearson = max_pearson
                best_model = valid_model

        print(f"{method} {t} {str(sigma) if sigma else ''} best validation model with pcc {valid_pearson}\n")
        return best_model

    if args.valid_method == 'bins_pred':
        try:
            print(f'load pred_bins{num_bins} cache')
            valid_inds = np.load(f"./Pred_bins/y_2016_split_by_pred_test_bins{num_bins}_checked.npy",allow_pickle=True)
        except Exception as e:
            print(e)
            print(f'errors occur when loading cache! now predict bins{num_bins}\n')
            valid_inds = pred_bins_model(num_bins, log_dir)
        
        valid_ind = np.hstack(valid_inds)
        train_ind = np.array([x for x in range(len(y_train_)) if x not in valid_ind])
        
        y_train = y_train_[train_ind]
        y_valid = y_train_[valid_ind]
        
        X_train = X_train_[train_ind]
        X_valid = X_train_[valid_ind]
        
        hist_valid, _ = np.histogram(y_valid, bins=arg_bins)
        print(f'{len(hist_valid)} bins for train and valid from test_pred bins, i.e. 2≤?<3')
        print([f'{_[i]:.2f}≤x<{_[i+1]:.2f}' for i in range(len(_)-1)])
        print('hist_valid: number of samples in bins',hist_valid)    
        print(f'sample size ratio of (train/valid) {y_train.shape[0]}:{y_valid.shape[0]}')

        valid_model, max_pearson = valid_info(model, X_train, X_valid, y_train, y_valid, log_dir,\
        f"{method}_{t}{'_'+str(sigma) if sigma else ''}_{2016}_valid_test_pred.txt",\
        epoch_num=args.valid_num_epoch, batch_size=args.valid_batch_size, lr=args.valid_lr,)

        print(f"{method} {t} {str(sigma) if sigma else ''} best validation model with pcc {max_pearson}\n")
        return valid_model

def train_info(net,X_train,X_test,y_train,y_test,log_dir,log_file,epoch_num=2,batch_size=512,lr=3e-4,weight_decay=args.weight_decay,upper_dim=False,cuda=True):
    if upper_dim:
        X_train = X_train[:,np.newaxis,:,:]
        X_test = X_test[:,np.newaxis,:,:]
    max_pearson = -1
    best_model_path = '0'

    loss_fn = torch.nn.MSELoss(reduction='mean')
    train_data = Data.TensorDataset(torch.from_numpy(X_train).float(),\
                                    torch.from_numpy(y_train).float())
    train_loader = Data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
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

    train_loader = Data.DataLoader(dataset=train_data,batch_size=300,shuffle=True,num_workers=0,drop_last=True)
    optimizer = torch.optim.Adam(net.parameters(),lr=5e-3, weight_decay=args.weight_decay)
    max_pearson = -1
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

    train_loader = Data.DataLoader(dataset=train_data,batch_size=50,shuffle=True,num_workers=0,drop_last=True)
    optimizer = torch.optim.Adam(net.parameters(),lr=5e-3, weight_decay=args.weight_decay)
    max_pearson = -1
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

    train_loader = Data.DataLoader(dataset=train_data,batch_size=50,shuffle=True,num_workers=0,drop_last=True)
    optimizer = torch.optim.Adam(net.parameters(),lr=5e-3, weight_decay=args.weight_decay)
    max_pearson = -1
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

def valid_info(net,X_train,X_test,y_train,y_test,log_dir,log_file,epoch_num=5,batch_size=512,lr=3e-4,weight_decay=args.weight_decay,upper_dim=False,cuda=True):
    if upper_dim:
        X_train = X_train[:,np.newaxis,:,:]
        X_test = X_test[:,np.newaxis,:,:]
    max_pearson = -1
    best_model_path = '0'

    loss_fn = torch.nn.MSELoss(reduction='mean')
    train_data = Data.TensorDataset(torch.from_numpy(X_train).float(),\
                                    torch.from_numpy(y_train).float())
    train_loader = Data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
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

def valid_info_classifier(net,X_train,X_test,y_train,y_test,log_dir,log_file,epoch_num=5,batch_size=512,lr=3e-4,weight_decay=args.weight_decay,upper_dim=False,cuda=True):
    if upper_dim:
        X_train = X_train[:,np.newaxis,:,:]
        X_test = X_test[:,np.newaxis,:,:]
        
    y_test = torch.from_numpy(y_test)

    max_acc = 0
    best_model_path = '0'

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    train_data = Data.TensorDataset(torch.from_numpy(X_train).float(),\
                                    torch.from_numpy(y_train).float())
    train_loader = Data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
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


