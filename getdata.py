import numpy as np 
import scipy.io 
from scipy.ndimage.filters import laplace
import matplotlib.pyplot as plt 


data_path = "./data256_4000/raw/"
train_data_folders = ['1','2','3','4']
test_data_folders = ['6hour']


def filter():
     filter = np.ones((64,64))
     for i in range(64):
         for j in range(64):
             filter[i,j]=np.exp(-(1.2*(i-32)/32)**10)*np.exp(-(1.2*(j-32)/32)**10)
     return filter

def get_snapshots(field, folders, filtering=False, slicing=True):
    print("Getting snapshots...%s:" %field)
    #obtains series of large domain field snapshots from folder (4000x4000km) from all folders 
    for folder in folders:
        print("   folder %g/%g" %(folders.index(folder)+1,len(folders)))
        inpath = data_path + folder + "/" + field + ".mat"
        if folders == train_data_folders:
            chop_init = 250
        elif folders == test_data_folders:
            chop_init = 1000
        data = scipy.io.loadmat(inpath)['in'][chop_init:]
        if folders.index(folder) == 0: 
            Data = data
        else: 
            Data = np.append(Data,data,0)
    ld = len(Data)
    
    # slices data into 16 images (1000x1000km)
    if slicing==True: 
        print("   slicing...")
        Data_sliced = np.empty((16*ld,64,64))
        for j in range(4): 
            for k in range(4):
                Data_sliced[(4*j+k)*ld:(4*j+k+1)*ld] = Data[:,j*64:j*64+64,k*64:k*64+64]
                print('   (%g,%g)' %(j,k))
        if filtering == True: 
            print('Filtering...')
            Data_sliced = Data_sliced*filter()
        return Data_sliced
    else:
        if filtering == True:
            print('Filtering...')
            Data = Data*filter()
        return Data
    

def get_flux(field, folders, filtering=False, slicing=True):
    #finds PV2 and V1 and calculates the flux from them
    print("Getting field...:")
    f = get_snapshots(field,folders, filtering, slicing)
    print('...V1:')
    V1 = get_snapshots('V1',folders, filtering, slicing)
    print('Averaging...') 
    flux = np.reshape(np.mean((f*V1),(1,2)),(-1,1))
    return flux


def save_field(field='PSI1',name='PSI1',filtering=False,slicing=True):
    train = get_snapshots(field,train_data_folders,filtering,slicing)
    std, mean = np.std(train), np.mean(train)
    train = 10*(train-mean)/std
    test = get_snapshots(field,test_data_folders,filtering,slicing)
    test = 10*(test-mean)/std
    np.savez('./data256_4000/fields/'+name, train, test, mean, std); del train, test, mean, std  
    return 

def save_flux(flux,name,filtering=False,slicing=True):
    train = get_flux(flux,train_data_folders,filtering,slicing)
    std, mean = np.std(train), np.mean(train)
    train = (train-mean)/(3*std)
    test = get_flux(flux,test_data_folders,filtering,slicing)
    test = (test-mean)/(3*std)
    np.savez('./data256_4000/fluxes/'+name, train, test, mean, std); del train, test, mean, std  
    return 



#Save data to file 
    
save_field(field='PSI1',name='PSI1',filtering=False,slicing=True)
save_field(field='PSI1',name='PSI1_f',filtering=True,slicing=True)

save_flux(flux='PV1',name='PV1',filtering=False,slicing=True)
save_flux(flux='PSI2',name='PSI2',filtering=False,slicing=True)
save_flux(flux='PV1',name='PV1_f',filtering=True,slicing=True)
save_flux(flux='PSI2',name='PSI2_f',filtering=True,slicing=True)



