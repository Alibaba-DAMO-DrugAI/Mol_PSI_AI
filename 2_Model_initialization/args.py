from argparse import ArgumentParser

import numpy as np
import torch
import os

def make_args():
    parser = ArgumentParser()
    # dataset
    parser.add_argument('--year', type=str, default='2007', help='PDBbinding dataset v(2007, 2013, 2016)')
    parser.add_argument('--model', type=str, default='Resnet', help='Deep backbones to extract features from Mol-PSIs.\
                        Densenet, Resnet for 2D/1D Mol-PSIs, (Image) Transformer for 1D Mol-PSIs')
    parser.add_argument('--sigma', type=float, default=1.5, help='Dimensionality of scale parameter sigma (0.0001, 0.75, 1.5, 2.25), for es0 and elec0, sigma=None')
    parser.add_argument('--datatype', type=str, default='elec', help='ES-IDM 1D (es0), ES-IEM 1D (elec0), ES-IDM 2D (es) and ES-IEM 2D (elec)')

    # train
    parser.add_argument('--num_epoch', type=int, default=1001, help='number of training epoch')
    parser.add_argument('--batch_size', type=int, default=1024, help='number of data points for each itration')
    parser.add_argument('--gpu', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--cuda', dest='cuda', default='0', type=str)
    parser.add_argument('--seed', type=int, default=11, help='Random seed for all random situation.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--transfer_model', type=bool, default=False, help='Training from the transfer model')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--num_repeat', type=int, default=1, help='Number of tries for the same experiments')

    # valid
    parser.add_argument('--valid_model', type=bool, default=True, help='Training from the transfer model')
    parser.add_argument('--valid_method', type=str, default='bins_pred', help='Stratified the train and valid subsets by random/bins/bins_pred/exist_model')
    parser.add_argument('--valid_bins_num',dest='num_bins', type=int, default=30, help='The number of bins used for (--valid_method bins/bins_pred 30,50,75,90,125,150)')
    parser.add_argument('--valid_path', type=str, default='', help='load valid model state dict when transfering on exist model (--valid_method exist_model)')
    parser.add_argument('--valid_size', type=int, default=0.2, help='Number of training dataset used for transfer')
    parser.add_argument('--valid_seed', type=int, default=11, help='Random seed for validation dataset selection')
    parser.add_argument('--valid_num_epoch', type=int, default=51, help='number of transfer epoch')
    parser.add_argument('--valid_batch_size', type=int, default=256, help='number of data points for each transfering itration')
    parser.add_argument('--valid_lr', type=float, default=5e-4, help='Initial transfer learning rate.')

    try:
        args = parser.parse_args() #call from command line
    except:
        args = parser.parse_args(args=[]) #call from notebook
    return args

args = make_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    torch.cuda.manual_seed(args.seed)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
else:
    print('Using CPU')
device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')

if args.model == 'trans' and (args.datatype == 'es0' or args.datatype == 'elec0'):
    raise ValueError('Image Transformer is not applicable for 2D Mol-PSIs')
if args.sigma and (args.datatype == 'es0' or args.datatype == 'elec0'):
    raise ValueError('1D Mol-PSIs is not applicable with sigmas')
if not args.sigma and (args.datatype != 'es0' and args.datatype != 'elec0'):
    raise ValueError('2D Mol-PSIs requires a specified sigma')
