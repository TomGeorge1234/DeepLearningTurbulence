import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from skimage.measure import block_reduce
import time
import sys 




#USER INPUT VARIABLES
flux = "PSI2"  #flux to learn, probably PSI2 (unfiltered) or PSI2_f (filtered)
field = "PSI1"  #field to learn flux, probably PSI1 or PSI1_f (filtered)
reload_data = True #if data is already loaded, save time by setting False
data_path = './data256_4000/'


def norm(X):
    return (X - np.mean(X))/(3*np.std(X))



# IGNORE THIS 


# #LOAD DATA
# #load data
# if reload_data == True:
    
#     print('Loading data...')
#     #this is all the available saved data, it is normalised by dividing by 2x the standard deviation of the test data
#     in_data = np.load(data_path + 'fields/' + field + ".npz").items()
#     in_train = in_data[0][1]
#     in_test = in_data[1][1]; del in_data
#     out_data = np.load(data_path + 'fluxes/' + flux + ".npz").items()
#     out_train = np.reshape(out_data[0][1],(-1,1))
#     out_test = np.reshape(out_data[1][1],(-1,1)); del out_data

#     #reassign the names, these are your data before any manipulation done below
#     X = np.reshape(block_reduce(in_train,(1,2,2),np.sum),(-1,32*32)); del in_train
#     X_ = np.reshape(block_reduce(in_test,(1,2,2),np.sum),(-1,32*32)); del in_test
#     Y = np.reshape(out_train,(-1)); del out_train
#     Y_ = np.reshape(out_test,(-1)); del out_test
#     X, X_, Y, Y_ = norm(X), norm(X_), norm(Y), norm(Y_)
#     print('Done.')
    
# np.savez('./data256_4000/other/sklearndata', X, X_, Y, Y_ ); 




# X and X_ is the train and test input 
# Y and Y_ are the train and test outputs 
data = np.load('./data256_4000/other/sklearndata.npz').items()
X, X_, Y, Y_ = data[0][1],data[1][1],data[2][1],data[3][1]





#SOME FUNCTIONS
def skill(yp,yt):
    return 1 - np.sqrt(((np.dot((yt-yp).T,(yt-yp)))/(len(yt))))/np.std(yt)

def lin_regress(yp,yt): #returns the coefficients of the linear fit yt = alpha + beta.yp
    beta = stats.mstats.linregress(yp,yt)[2]*(np.std(yt)/np.std(yp))
    alpha = np.mean(yt) - beta*np.mean(yp)
    return alpha, beta
    
def linearly_regressed(yp,yt):
    a = lin_regress(yp,yt)
    return a[0] + a[1]*yp

def R_squared(yp,yt):
    return stats.mstats.linregress(yp,yt)[2]**2

def time_series(prediction):
    flux_nontriv = Y_
    flux_nontriv_recon = linearly_regressed(prediction,flux_nontriv)
    fig, ax = plt.subplots(figsize=(15, 4))
    plt.plot(np.arange(200,600,0.25),flux_nontriv[0:1600],color='#ff7f0e',alpha=0.5, label = r'Truth')
    plt.plot(np.arange(200,600,0.25),flux_nontriv_recon[0:1600],color='#ff7f0e',alpha=1, label = r'Prediction, Skill: %.4f, $R^{2}$: %.2f' %(skill(flux_nontriv_recon,flux_nontriv),R_squared(flux_nontriv_recon,flux_nontriv)))
    plt.xlabel(r'Days', fontsize=18)
    plt.ylabel(r'Eddy Thickness Flux', fontsize=18)
    plt.xticks([200,250,300,350,400,450,500,550,600],fontsize=14)
    plt.yticks([-1,-0,1],fontsize=14)
    plt.yticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14,frameon=False,loc=8)
    plt.margins(x=0)
    plt.show()









####This runs the sklearn regression on small amounts of the data set to find an estimate for the computing time scaling (assuming t = C*N^k)
####It returns a log log graph, and an estimate for how long the regression would take on the WHOLE data set (112000)
    
#### chose only 1
    
# reg = tree.DecisionTreeRegressor()
# reg = KernelRidge(alpha=1)
# reg = linear_model.BayesianRidge()
# reg = svm.SVR()
# reg = MLPRegressor(hidden_layer_sizes=(100,10,))
# reg = RandomForestRegressor(n_estimators = 100)  
# T = []
# N = [100,300,500]
# for n in N:
#     t = time.time()
#     reg.fit(X[:n], Y[:n])  
#     T.append(time.time() - t)
# plt.scatter(np.log(N),np.log(T))
# alpha, beta = lin_regress(np.log(N),np.log(T))
# print('Scaling coeff: %.2f' %beta)
# predicted_time = np.exp(beta*np.log(112000) + alpha)
# print('Full set time prediction, %.1f mins' %(predicted_time/60))





####This runs the regression on the WHOLE data set then fits it to the training data set,
#### then plots the estimated flux against the true flux and calculates the skill

# reg = svm.SVR()
# reg = KernelRidge(alpha=1.0)
# reg = tree.DecisionTreeRegressor()
reg = RandomForestRegressor(n_estimators = 100)
# reg = MLPRegressor(hidden_layer_sizes=(100,10,))
# reg = linear_model.BayesianRidge()
reg.fit(X[:112000],Y[:112000]) 
prediction = reg.predict(X_)
prediction = linearly_regressed(prediction,Y_)
print('Skill: %.4f' %(skill(prediction,Y_)))
print('R_squared = %.4f' %(R_squared(prediction,Y_)))
time_series(prediction)


sk = 0
for i in range(10):
    reg = MLPRegressor(hidden_layer_sizes=(100,10,))
    reg.fit(X[:112000], Y[:112000]) 
    prediction = reg.predict(X_)
    skll = skill(linearly_regressed(prediction,Y_),Y_)
    print('Skill: %.4f' %skll)
    sk += skll
print(sk/10)

    
