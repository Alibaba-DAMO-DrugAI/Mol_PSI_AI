import copy
import datetime
import os

from args import * 
from dataset import *
from perspect import *
from utils import *

#%% logs
now = datetime.datetime.now()
date = now.strftime("%B%d")
log_dir = f'log_{date}'
if not os.path.exists(f"log_{date}/best_models/"):
    os.makedirs(f"log_{date}/best_models/")
if not os.path.exists(f"log_{date}/valid_models/"):
    os.makedirs(f"log_{date}/valid_models/")
if not os.path.exists(f"log_{date}/cls_models/"):
    os.makedirs(f"log_{date}/cls_models/")

#%% args 
# args.year = '2007'
# args.model = 'Densenet'
# args.datatype = 'elec0'
# args.sigma = 2
# args.num_repeat = 2
# args.num_epoch = 2
# args.valid_num_epoch = 2
# args.valid_method = 'bins_pred'
# args.valid_path = './Backup_model/elec0_dense.pt'
# args.num_bins = 150
print(args,'\n')

#%% main
X_train, X_test, y_train, y_test = tensors_by_year_type(args.year,args.sigma,args.datatype)

model_valid = valid_model(args.model,args.datatype,args.sigma,log_dir)

for i in range(1,args.num_repeat+1): #number of repeats
    print(f"repeat {i+1}:")
    # model = transfer_model(args.model,args.datatype,args.transfer_model)
    model = copy.deepcopy(model_valid)
    train_info(model,X_train,X_test,y_train,y_test, log_dir,\
    f"try_{i}_{args.model}_{args.datatype}{'_'+str(args.sigma) if args.sigma else ''}_{args.year}.txt",\
    epoch_num=args.num_epoch, batch_size=args.batch_size, lr=args.lr)

print("work done!")
# %%
