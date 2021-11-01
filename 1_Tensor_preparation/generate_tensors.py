
import os
import numpy as np
import pandas as pd
import numpy_indexed as npi
import matplotlib.pyplot as plt

from tqdm import tqdm
import cv2 as cv


# range_2007 = {'norm':{int(line.split()[0]):float(line.split()[1]) for line in open('../PDB_data/maxmin_2007.txt')},\
#             'elec':{int(line.split()[0]):float(line.split()[1]) for line in open('../PDB_data/maxmin_2007_elec.txt')},}

train_2007 = [line.strip() for line in open('../PDB_data/train_2007.txt')]
train_2013 = [line.strip() for line in open('../PDB_data/train_2013.txt')]
train_2016 = [line.strip() for line in open('../PDB_data/train_2016.txt')]
train_all = train_2007 + train_2013 + train_2016

test_2007 = [line.strip() for line in open('../PDB_data/core_2007.txt')]
test_2013 = [line.strip() for line in open('../PDB_data/core_2013.txt')]
test_2016 = [line.strip() for line in open('../PDB_data/core_2016.txt')]
test_all = test_2007 + test_2013 + test_2016

y_train_2007 = np.load('../PDB_data/y_train_2007.npy')
y_train_2013 = np.load('../PDB_data/y_train_2013.npy')
y_train_2016 = np.load('../PDB_data/y_train_2016.npy')

y_test_2007 = np.load('../PDB_data/y_test_2007.npy')
y_test_2013 = np.load('../PDB_data/y_test_2013.npy')
y_test_2016 = np.load('../PDB_data/y_test_2016.npy')

y_2007 = np.append(y_train_2007,y_test_2007)
y_2013 = np.append(y_train_2013,y_test_2013)
y_2016 = np.append(y_train_2016,y_test_2016)

# def generate_img_by_group(name,mf):

def normalize(x,ma,mi,ran=255):
    x[x!=0]=(x[x!=0]-mi)/(ma-mi)*ran
    return x

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
    # src = '../PDB_data/2007_normDist/1a0q_ES_b01_c2_pocket_eigv.txt'

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

def checking_img(img,combo_num,names,y_trues):
    plt.figure()
    # assert combo_num in [36,50]
    if combo_num == 36:
        m,n = 4,9
    if combo_num == 50:
        m,n = 5,10
    for j in range(combo_num):
        plt.subplot(m,n,j+1)
        plt.imshow(img[j,:,:])
    plt.show()
    print('\n',names[i],y_trues[i],'\n')

################## demo with v2 on 2007 Norm 100x100
sigma = 1.5 # 0.0001, 0.75, 1.5, 2.25 # process each sigma in parallel can save lots of time, same to normDist and elecDist.
n_combo = 36
root = '../PDB_data/2007_normDist'
tardir = root.replace('PDB_data', 'PDB_data/Temp_tensor_Dec_12')+f'_sigma_{sigma}' #100x100
if not os.path.exists(tardir):
    os.makedirs(tardir)

files = [x for x in os.listdir(root) if x.endswith('eigv.txt')]
files_sorted = sorted(files,key=lambda x:(train_2007+test_2007).index(x.split('_')[0]))

tensor_2007_norm = np.zeros((len(files_sorted),n_combo,100,100))
for i,file in enumerate(tqdm(files_sorted)):
    img = gaussian_mix_spectrum_by_file(root+'/'+file, sigma, 'norm',\
        adapt_range=False, exclude_zero=True, v2=True)
#    checking_img(img,n_combo,train_2007+test_2007,y_2007) #to check or not
    tensor_2007_norm[i,:,:,:] = img
np.savez(tardir+'/'+'tensor_2007_norm.npy',tensor_2007_norm)

## norm_numzero
tensor_2007_norm_numzero = np.zeros((len(files_sorted),n_combo,100))
for i,file in enumerate(tqdm(files_sorted)):
    zero_img = zero_eigen_map(root+'/'+file, 'norm', v2=True)
#    plt.figure()
#    plt.imshow(zero_img)
#    plt.show()
    tensor_2007_norm_numzero[i,:,:] =zero_img
np.save(os.path.dirname(tardir)+'/'+'tensor_2007_norm_numzero.npy',tensor_2007_norm_numzero)


################## demo with v2 on 2007 elec 100x100 and 2007 elec0 100x100
sigma = 1.5 # 0.0001, 0.75, 1.5, 2.25 # process each sigma in parallel can save lots of time, same to normDist and elecDist.
n_combo = 50
root = '../PDB_data/2007_elecDist'
tardir = root.replace('PDB_data', 'PDB_data/Temp_tensor_Dec_12')+f'_sigma_{sigma}'
if not os.path.exists(tardir):
    os.makedirs(tardir)

files = [x for x in os.listdir(root) if x.endswith('eigv_2.txt')]
files_sorted = sorted(files,key=lambda x:(train_2007+test_2007).index(x.split('_')[0]))

tensor_2007_elec = np.zeros((len(files_sorted),n_combo,100,100))
for i,file in enumerate(tqdm(files_sorted)):
    img = gaussian_mix_spectrum_by_file(root+'/'+file, sigma, 'elec',\
        adapt_range=False, exclude_zero=True, v2=False)
#    checking_img(img,n_combo,train_2007+test_2007,y_2007) #to check or not
    tensor_2007_elec[i,:,:,:] =img
np.savez(tardir+'/'+'tensor_2007_elec.npy',tensor_2007_elec)

## elec_numzero
tensor_2007_elec_numzero = np.zeros((len(files_sorted),n_combo,100))
for i,file in enumerate(tqdm(files_sorted)):
    zero_img = zero_eigen_map(root+'/'+file, 'elec', v2=False)
#    plt.figure()
#    plt.imshow(zero_img)
#    plt.show()
    tensor_2007_elec_numzero[i,:,:] =zero_img
np.save(os.path.dirname(tardir)+'/'+'tensor_2007_elec_numzero.npy',tensor_2007_elec_numzero)


################## demo with v2 on 2013 Norm 100x100
# sigma = 1 # 0.0001, 0.75, 1.5, 2.25 # process each sigma in parallel can save lots of time
# n_combo = 36
# root = '../PDB_data/2013_norm'
# tardir = root.replace('PDB_data', 'PDB_data/Temp_tensor_Dec_12')+f'_sigma_{sigma}'
# if not os.path.exists(tardir):
#     os.makedirs(tardir)

# files = [x for x in os.listdir(root)]
# files_sorted = sorted(files,key=lambda x:(train_2013+test_2013).index(x.split('_')[0]))

# tensor_2013_norm = np.zeros((len(files_sorted),n_combo,100,100))
# for i,file in enumerate(tqdm(files_sorted)):
#     img = gaussian_mix_spectrum_by_file(root+'/'+file, sigma, 'norm',\
#         adapt_range=False, exclude_zero=True)
#     checking_img(img,n_combo,train_2013+test_2013,y_2013) #to check or not
#     tensor_2013_norm[i,:,:,:] = img
# np.savez(tardir+'/'+'tensor_2013_norm.npy',tensor_2013_norm)

## norm_numzero
# tensor_2013_norm_numzero = np.zeros((len(files_sorted),n_combo,100))
# for i,file in enumerate(tqdm(files_sorted)):
#     zero_img = zero_eigen_map(root+'/'+file, 'norm',)
#     plt.figure()
#     plt.imshow(zero_img)
#     plt.show()
#     tensor_2013_norm_numzero[i,:,:] =zero_img
# np.save(os.path.dirname(tardir)+'/'+'tensor_2013_norm_numzero.npy',tensor_2013_norm_numzero)

################## demo with v2 on 2013 elec 100x100
# sigma = 1 # 0.0001, 0.75, 1.5, 2.25 # process each sigma in parallel can save lots of time
# n_combo = 50
# root = '../PDB_data/2013_elec'
# tardir = root.replace('PDB_data', 'PDB_data/Temp_tensor_Dec_12')+f'_sigma_{sigma}'
# if not os.path.exists(tardir):
#     os.makedirs(tardir)

# files = [x for x in os.listdir(root) if x.endswith('eigv_2.txt')]
# files_sorted = sorted(files,key=lambda x:(train_2013+test_2013).index(x.split('_')[0]))

# tensor_2013_elec = np.zeros((len(files_sorted),n_combo,100,100))
# for i,file in enumerate(tqdm(files_sorted)):
#     img = gaussian_mix_spectrum_by_file(root+'/'+file, sigma, 'elec',\
#         adapt_range=False, exclude_zero=True)
#     checking_img(img,n_combo,train_2013+test_2013,y_2013) #to check or not
#     tensor_2013_elec[i,:,:,:] = img
# np.savez(tardir+'/'+'tensor_2013_elec.npy',tensor_2013_elec)

## elec_numzero
# tensor_2013_elec_numzero = np.zeros((len(files_sorted),n_combo,100))
# for i,file in enumerate(tqdm(files_sorted)):
#     zero_img = zero_eigen_map(root+'/'+file, 'elec',)
#     plt.figure()
#     plt.imshow(zero_img)
#     plt.show()
#     tensor_2013_elec_numzero[i,:,:] =zero_img
# np.save(os.path.dirname(tardir)+'/'+'tensor_2013_elec_numzero.npy',tensor_2013_elec_numzero)


################## demo with v2 on 2016 Norm 100x100
# sigma = 1 # 0.0001, 0.75, 1.5, 2.25 # process each sigma in parallel can save lots of time
# n_combo = 36
# root = '../PDB_data/2016_norm'
# tardir = root.replace('PDB_data', 'PDB_data/Temp_tensor_Dec_12')+f'_sigma_{sigma}'
# if not os.path.exists(tardir):
#     os.makedirs(tardir)

# files = [x for x in os.listdir(root)]
# files_sorted = sorted(files,key=lambda x:(train_2016+test_2016).index(x.split('_')[0]))

# tensor_2016_norm = np.zeros((len(files_sorted),n_combo,100,100))
# for i,file in enumerate(tqdm(files_sorted)):
#     img = gaussian_mix_spectrum_by_file(root+'/'+file, sigma, 'norm',\
#         adapt_range=False, exclude_zero=True)
#     checking_img(img,n_combo,train_2016+test_2016,y_2016) #to check or not
#     tensor_2016_norm[i,:,:,:] = img
# np.savez(tardir+'/'+'tensor_2016_norm.npy',tensor_2016_norm)

## norm_numzero
# tensor_2016_norm_numzero = np.zeros((len(files_sorted),n_combo,100))
# for i,file in enumerate(tqdm(files_sorted)):
#     zero_img = zero_eigen_map(root+'/'+file, 'norm',)
#     plt.figure()
#     plt.imshow(zero_img)
#     plt.show()
#     tensor_2016_norm_numzero[i,:,:] =zero_img
# np.save(os.path.dirname(tardir)+'/'+'tensor_2016_norm_numzero.npy',tensor_2016_norm_numzero)

################## demo with v2 on 2016 elec 100x100
# sigma = 1 # 0.0001, 0.75, 1.5, 2.25 # process each sigma in parallel can save lots of time
# n_combo = 50
# root = '../PDB_data/2016_elec'
# tardir = root.replace('PDB_data', 'PDB_data/Temp_tensor_Dec_12')+f'_sigma_{sigma}'
# if not os.path.exists(tardir):
#     os.makedirs(tardir)

# files = [x for x in os.listdir(root) if x.endswith('eigv_2.txt')]
# files_sorted = sorted(files,key=lambda x:(train_2016+test_2016).index(x.split('_')[0]))

# tensor_2016_elec = np.zeros((len(files_sorted),n_combo,100,100))
# for i,file in enumerate(tqdm(files_sorted)):
#     img = gaussian_mix_spectrum_by_file(root+'/'+file, sigma, 'elec',\
#         adapt_range=False, exclude_zero=True)
#     checking_img(img,n_combo,train_2016+test_2016,y_2016) #to check or not
#     tensor_2016_elec[i,:,:,:] = img
# np.savez(tardir+'/'+'tensor_2016_elec.npy',tensor_2016_elec)

## elec_numzero
# tensor_2016_elec_numzero = np.zeros((len(files_sorted),n_combo,100))
# for i,file in enumerate(tqdm(files_sorted)):
#     zero_img = zero_eigen_map(root+'/'+file, 'elec',)
#     plt.figure()
#     plt.imshow(zero_img)
#     plt.show()
#     tensor_2016_elec_numzero[i,:,:] =zero_img
# np.save(os.path.dirname(tardir)+'/'+'tensor_2016_elec_numzero.npy',tensor_2016_elec_numzero)


################## check img with subplots and output
# tensor_2007_norm = np.load('Temp_tensor_100x100/2007_normDist_sigma_2/tensor_2007_norm.npy')
# tensor_2007_elec = np.load('Temp_tensor_100x100/2007_elecDist_sigma_2/tensor_2007_elec.npy')

# img_dir ='Temp_img_100x100/2007_normDist_sigma_2'
# if not os.path.exists(img_dir):
#     os.makedirs(img_dir)

# for i in tqdm(range(1300)):
#     fig = plt.figure(figsize=(20,12))
#     for j in range(36):
#         plt.subplot(4,9,j+1,)
#         plt.imshow(tensor_2007_norm[i,j,:,:])
#         # plt.xlabel('Filtration size')
#         # plt.ylabel('Spectral values')
#     fig.text(0.5,0.1,'Spectral values',ha ='center',size=20)
#     fig.text(0.09,0.5,'Filtration size',va ='center',rotation ='vertical',size=20)
#     plt.show()
    # plt.savefig(img_dir+'/'+f'{(train_2007+test_2007)[i]}.png')



