#QGmain.py Author: Tom George
#To use, set user input variables, check global variable, 

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import svm
from time import time
import sys 
sys.path.append('./networks/')



#USER INPUT VARIABLES
flux = "PSI2"  #flux to learn, probably PSI2 (unfiltered) or PSI2_f (filtered)
field = "PSI1"  #field to learn flux, probably PSI1 or PSI1_f (filtered)



#SOME GLOBAL VARIABLES
eps = 1e-3 #adamoptimizer learning rate
K = 100 #learning batch size
reload_data = True #if data is already loaded, save time by setting False
testfreq = 100 #how often testing is done 
drop_prob = 0.7 #this is the keep-probability
data_path = './data256_4000/'




#LOAD AND MANIPULATE DATA
#load data
if reload_data == True:
    
    print('Loading data...')
    #this is all the available saved data, it is normalised by dividing by 2x the standard deviation of the test data
    in_data = np.load(data_path + 'fields/' + field + ".npz").items()
    in_train = in_data[0][1]
    in_test = in_data[1][1]; del in_data
    out_data = np.load(data_path + 'fluxes/' + flux + ".npz").items()
    out_train = np.reshape(out_data[0][1],(-1,1))
    out_test = np.reshape(out_data[1][1],(-1,1)); del out_data


    #reassign the names, these are your data before any manipulation done below
    trainimages_ = in_train; del in_train
    testimages_ = in_test; del in_test
    trainoutput_ = out_train; del out_train
    testoutput_ = out_test ; del out_test
    print('Done.')
    

#manipulate data
##1) No manipulation
X = np.reshape(trainimages_,(-1,64*64))
testimages = np.reshape(testimages_,(-1,64*64))
Y = np.reshape(trainoutput_,(-1))
testoutput = testoutput_
    

##2) Train on only the first s images in your total training data
# s = 33000
# trainimages = trainimages_[0:s,:,:]
# trainoutput = trainoutput_[0:s]
# testimages = testimages_
# testoutput = testoutput_



#SOME FUNCTIONS AND ARRAY INITIALISATION
#functions
def accuracy(yp,yt):
    return stats.mstats.linregress(yp,yt)[2]

def skill(yp,yt):
    return 1 - np.sqrt(((np.dot((yt-yp).T,(yt-yp)))/(len(yt))))/np.std(yt)


clf = svm.SVR()
clf.fit(np.reshape(X,(-1,64*64))[:2000], Y[:2000])  

x = np.reshape(X,(-1,64*64))

clf.predict(testimages[:1000])

