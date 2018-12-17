import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.cm as cm
from eofs.standard import Eof
import time






savekey = 'live' #explain network and make the save recognisable
reload_data = True #if data is already loaded, save time by setting False
data_path = './data256_4000/'




def skill(yp,yt):
    skill = 1 - np.sqrt(((np.dot((yt-yp).T,(yt-yp)))/(len(yt))))/np.std(yt)
    return skill

def lin_regress(yp,yt): #returns the coefficients of the linear fit yt = alpha + beta.yp
    beta = stats.mstats.linregress(yp,yt)[2]*(np.std(yt)/np.std(yp))
    alpha = np.mean(yt) - beta*np.mean(yp)
    return alpha, beta

def linearly_regressed(yp,yt):
    a = lin_regress(yp,yt)
    return a[0] + a[1]*yp

def R_squared(yp,yt):
    return stats.mstats.linregress(yp,yt)[2]




if reload_data == True: 
    PSI1 = np.load(data_path + 'fields/' + "PSI1.npz").items()
    PSI2 = np.load(data_path + 'fields/' + "PSI2.npz").items()
    V1 = np.load(data_path + 'fields/' + "V1.npz").items()
    
    PSI1_train, PSI1_test = PSI1[0][1], PSI1[1][1]; del PSI1
    PSI2_train, PSI2_test = PSI2[0][1], PSI2[1][1]; del PSI2
    V1_train, V1_test = V1[0][1], V1[1][1]; del V1

test_flux = np.load(data_path + 'fluxes/' + "PSI2.npz").items()[1][1]


# streamfunctions_train = np.concatenate((PSI1_train,PSI2_train),1)
# streamfunctions_test = np.concatenate((PSI1_test,PSI2_test),1)
# solver = Eof(streamfunctions_train)
# steamfunction_eofs = solver.eofs()
# np.savez(data_path + 'EOFs/steamfunction_eofs', steamfunction_eofs)


steamfunction_eofs = np.load(data_path + 'EOFs/steamfunction_eofs.npz').items()[0][1]

#by reconstructing PV1 
eofs_top = np.reshape(steamfunction_eofs[:,:64,:],(-1,4096)).T
eofs_bottom = np.reshape(steamfunction_eofs[:,64:,:],(-1,4096)).T

test_coeffs = np.dot(np.reshape(PSI1_test,(16000,-1)),eofs_top)
a = np.reshape(np.dot(test_coeffs[0],eofs_top.T),(64,64))
plt.imshow(a)
plt.imshow(PSI1_test[0])


n = [34,35,36,37,38,39,40,41]
for N in n:
    PSI2_recon = np.dot(test_coeffs[:,0:N],eofs_bottom[:,0:N].T)
    guessflux = np.reshape(np.mean(np.reshape((np.reshape(PSI2_recon,(-1,64,64))*V1_test),(-1,4096)),1),(-1,1))
    error = skill(linearly_regressed(guessflux,test_flux),test_flux)[0][0]
    print(N,error)
#max is for N = 37
   
    
flux_nontriv = test_flux
PSI2_recon = np.dot(test_coeffs[:,0:37],eofs_bottom[:,0:37].T)
flux_nontriv_recon = np.reshape(np.mean(np.reshape((np.reshape(PSI2_recon,(-1,64,64))*V1_test),(-1,4096)),1),(-1,1))
flux_nontriv_recon = linearly_regressed(flux_nontriv_recon,flux_nontriv)


#PSI2 = k*PSI1+c
PSI2_recon = PSI1_test
guessflux = np.reshape(np.mean(np.reshape((np.reshape(PSI2_recon,(-1,64,64))*V1_test),(-1,4096)),1),(-1,1))
skill = skill(linearly_regressed(guessflux,test_flux),test_flux)[0][0]
print(skill)


fig, ax = plt.subplots(figsize=(15, 4))
plt.plot(np.arange(200,600,0.25),flux_nontriv[0:1600],color='#ff7f0e',alpha=0.5, label = r'Truth')
plt.plot(np.arange(200,600,0.25),flux_nontriv_recon[0:1600],color='#ff7f0e',alpha=1, label = r'Prediction, Skill: %.3f, $R^{2}$: %.2f' %(skill(linearly_regressed(flux_nontriv_recon,flux_nontriv),flux_nontriv),R_squared(linearly_regressed(flux_nontriv_recon,flux_nontriv),flux_nontriv)))
plt.xlabel(r'Days', fontsize=18)
plt.ylabel(r'Eddy Thickness Flux', fontsize=18)
plt.xticks([200,250,300,350,400,450,500,550,600],fontsize=14)
plt.yticks([-1,-0,1],fontsize=14)
plt.yticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14,frameon=False,loc=8)
plt.margins(x=0)
plt.show()
# plt.savefig("./figures3/-.png", dpi=300, bbox_inches = 'tight',transparent=True)
