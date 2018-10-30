import matplotlib.pyplot as plt
import matplotlib
import numpy as np 
import matplotlib.cm as cm
import scipy.signal
from eofs.standard import Eof
from scipy import stats
from time import time 

from matplotlib import rc
plt.rc('text',usetex=True)


def RMSerror(yp,yt):
    RMSerror = np.sqrt(((np.dot((yt-yp).T,(yt-yp)))/(len(yt))))/np.std(yt)
    return RMSerror

def skill(yp,yt):
    return 1 - np.sqrt(((np.dot((yt-yp).T,(yt-yp)))/(len(yt))))/np.std(yt)

def lin_regress(yp,yt): #returns the coefficients of the linear fit yt = alpha + beta.yp
    beta = stats.mstats.linregress(yp,yt)[2]*(np.std(yt)/np.std(yp))
    alpha = np.mean(yt) - beta*np.mean(yp)
    return alpha, beta
    
def linearly_regressed(yp,yt):
    a = lin_regress(yp,yt)
    return a[0] + a[1]*yp



#single print of field

# PSI1 = np.load("./data256_4000/" + "PSI1_test" + ".npz").items()[0][1] * 0.1 * np.load("./data256_4000/" + "PSI1_test" + ".npz").items()[2][1] + np.load("./data256_4000/" + "PSI1_test" + ".npz").items()[1][1]
# PV1 = np.load("./data256_4000/" + "PV1_test" + ".npz").items()[0][1] * 0.1 * np.load("./data256_4000/" + "PV1_test" + ".npz").items()[2][1] + np.load("./data256_4000/" + "PV1_test" + ".npz").items()[1][1]
# PSI2 = np.load("./data256_4000/" + "PSI2_test" + ".npz").items()[0][1] * 0.1 * np.load("./data256_4000/" + "PSI2_test" + ".npz").items()[2][1] + np.load("./data256_4000/" + "PSI2_test" + ".npz").items()[1][1]
# PV2 = np.load("./data256_4000/" + "PV2_test" + ".npz").items()[0][1] * 0.1 * np.load("./data256_4000/" + "PV2_test" + ".npz").items()[2][1] + np.load("./data256_4000/" + "PV2_test" + ".npz").items()[1][1]

# flux_PV1 = np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[0][1]*3*np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[2][1] + np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[1][1]# flux_PV1_full = np.load("./data256_4000/" + "flux_PV1full_test" + ".npz").items()[0][1]# flux_full = np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[0][1]*3*np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[2][1]+np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[1][1]

# plt.plot(flux_full[0:1000])
# plt.plot((flux_triv+flux_nontriv)[0:1000])
# print(stats.mstats.linregress(flux_full,flux_triv+flux_nontriv)[2])

# K = 500
# Z = PSI2[K]
# ax = plt.subplots()
# plt.imshow(Z, extent=[0, 1000, 0, 1000])
# plt.yticks([0,200,400,600,800,1000],size=14)
# plt.xticks([0,200,400,600,800,1000],size=14)
# cbar = plt.colorbar()
# ax = plt.axis()
# for font_objects in cbar.ax.yaxis.get_ticklabels():
#     font_objects.set_size(16)
# # plt.show()
# plt.savefig("./figures3/PSI2.png", dpi=300, bbox_inches = 'tight', transparent=True)



# plt.figure(1,(11,4))
# plt.plot(np.arange(200,500,0.25),flux_PV1[0:1200],label=r'PV Flux: $\overline{v_{1}q_{1}}$')
# plt.plot(np.arange(200,500,0.25),flux_PSI2f[0:1200],label=r'Filtered $\psi_{2}$ Flux: $\overline{P \times v_{1}\psi_{2}}$')
# plt.plot(np.arange(200,500,0.25),flux_PSI2[0:1200],label=r'$\psi_{2}$ Flux: $\overline{v_{1}\psi_{2}}$')
# # plt.plot(np.arange(200,500,0.25),flux_q1_weird[0:1200],'red',label=r'Red subdomain')
# # plt.plot(np.arange(200,500,0.25),flux_q1_weird[1200:2400],'orange',label=r'Orange subdomain')
# plt.xlabel(r'Days', size=20)
# plt.ylabel(r'Flux', size=20)
# plt.legend()
# # plt.show()
# plt.savefig("./figures3/fluxes.png", dpi=300, bbox_inches = 'tight', transparent=True)

# PSI2f_correlation = np.zeros(300)
# for t in range (300):
#     PSI2f_correlation[t] = scipy.stats.mstats.linregress(EKEf[t:1200],-flux_PSI2f[0:1200-t])[2]
# PV1_correlation = np.zeros(300)
# for t in range (300):
#     PV1_correlation[t] = scipy.stats.mstats.linregress(EKE[t:1200],-flux_PV1[0:1200-t])[2]
# PSI2_correlation = np.zeros(300)
# for t in range (300):
#     PSI2_correlation[t] = scipy.stats.mstats.linregress(EKE[t:1200],-flux_PSI2[0:1200-t])[2]
# PV1full_correlation = np.zeros(300)
# for t in range (300):
#     PV1full_correlation[t] = scipy.stats.mstats.linregress(EKEfull[t:1200],flux_PV1_full[0:1200-t])[2]

# plt.figure(2,(13,4))
# plt.plot(np.arange(0,75,0.25),PV1full_correlation,label=r'Correlation[$\overline{v_{1}q_{1}},EKE$], over the full domain')
# plt.plot(np.arange(0,75,0.25),PSI2f_correlation,label=r'Correlation[$\overline{P \times v_{1}\psi_{2}},EKE$], over a subdomain')
# plt.plot(np.arange(0,75,0.25),PV1_correlation,label=r'Correlation[$\overline{v_{1}q_{1}}*EKE$], over a subdomain')
# plt.plot(np.arange(0,75,0.25),PSI2_correlation,label=r'')
# plt.xlabel(r'Lag Days', size=20)
# plt.ylabel(r'Correlation', size=20)
# plt.legend()
# plt.show()
# plt.savefig("./figures3/fluxEKEcorrelations.png", dpi=300, bbox_inches = 'tight', transparent=True)

    

# plt.plot(flux_PSI2[0:2000])
# plt.plot(flux_PSI2f[0:2000])
# scipy.stats.mstats.linregress(flux_PSI2[0:365*4],flux_PSI2f[0:365*4])[2]
# np.corrcoef(flux_PSI2.T,flux_PSI2f.T)








#phase space plot 

# def moving_average(data_set, periods=3):
#     weights = np.ones(periods) / periods
#     return np.convolve(data_set, weights, mode='valid')


# ax = plt.figure().add_subplot(111)

# R1 = np.max(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages330.npz').items()[7][1])
# R2 = np.max(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages1000.npz').items()[7][1])
# R3 = np.max(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages3300.npz').items()[7][1])
# R4 = np.max(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages10000.npz').items()[7][1])
# R5 = np.max(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages18000.npz').items()[7][1])
# R6 = np.max(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages33000.npz').items()[7][1])
# R7 = np.max(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages56000.npz').items()[7][1])
# R8 = np.max(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages115200.npz').items()[7][1])

# R1_ = np.max(moving_average(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages330.npz').items()[6][1],192))
# R2_ = np.max(moving_average(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages1000.npz').items()[6][1],192))
# R3_ = np.max(moving_average(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages3300.npz').items()[6][1],192))
# R4_ = np.max(moving_average(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages10000.npz').items()[6][1],192))
# R5_ = np.max(moving_average(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages18000.npz').items()[6][1],192))
# R6_ = np.max(moving_average(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages33000.npz').items()[6][1],192))
# R7_ = np.max(moving_average(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages56000.npz').items()[6][1],192))
# R8_ = np.max(moving_average(np.load('/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages115200.npz').items()[6][1],192))

# y = (R1,R2,R3,R4,R5,R6,R7,R8)
# y_ = (R1_,R2_,R3_,R4_,R5_,R6_,R7_,R8_)
# x = (330,1000,3300,10000,18000,33000,56000,115200)
# plt.scatter(x,y_,marker='.',label="Training")
# plt.plot(x,y_,alpha=0.7)
# plt.scatter(x,y,marker='.',label="Testing")
# plt.plot(x,y,alpha=0.7)
# plt.ylim(-0.05,1.05)
# ax.set_xscale('log')
# plt.xlabel(r'No. training images', fontsize=22)
# plt.ylabel(r'R', fontsize = 22)
# plt.legend(fontsize = 16)
# # plt.show()
# plt.savefig("./figures3/phase_space.png", dpi=300, bbox_inches = 'tight',transparent=True)

      
  
# skill chart for deeper nets 
def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


ax = plt.figure().add_subplot(111)

flux_std = np.std(np.load("./data256_4000/" + "fluxes/PSI2" + ".npz").items()[0][1]) 

path = '/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfile'

#calculations of the skills from the costs 
S1 = 1-((np.min(np.load(path+'1c2f.npz').items()[5][1])**0.5)/flux_std)
S2 = 1-((np.min(np.load(path+'2c2f.npz').items()[5][1])**0.5)/flux_std)
S3 = 1-((np.min(np.load(path+'3c2f.npz').items()[5][1])**0.5)/flux_std)
S4 = 1-((np.min(np.load(path+'4c3f.npz').items()[5][1])**0.5)/flux_std)
S5 = 1-((np.min(np.load(path+'5c4f.npz').items()[5][1])**0.5)/flux_std)
S6 = 1-((np.min(np.load(path+'6c4f.npz').items()[5][1])**0.5)/flux_std)

S1_ = 1-((np.min(moving_average(np.load(path+'1c2f.npz').items()[4][1],192))**0.5)/flux_std)
S2_ = 1-((np.min(moving_average(np.load(path+'2c2f.npz').items()[4][1],192))**0.5)/flux_std)
S3_ = 1-((np.min(moving_average(np.load(path+'3c2f.npz').items()[4][1],192))**0.5)/flux_std)
S4_ = 1-((np.min(moving_average(np.load(path+'4c3f.npz').items()[4][1],192))**0.5)/flux_std)
S5_ = 1-((np.min(moving_average(np.load(path+'5c4f.npz').items()[4][1],192))**0.5)/flux_std)
S6_ = 1-((np.min(moving_average(np.load(path+'6c4f.npz').items()[4][1],192))**0.5)/flux_std)

y = (S1,S2,S3,S4,S5)
y_ = (S1_,S2_,S3_,S4_,S5_)
x = (26161,40901,87525,245282,538989)
plt.scatter(x,y_,marker='.',label="Training")
plt.plot(x,y_,alpha=0.7)
plt.scatter(x,y,marker='.',label="Testing")
plt.plot(x,y,alpha=0.7)
plt.ylim(-0.05,1.05)
ax.set_xscale('log')
plt.xlabel(r'Complexity (No. trainable parameters)', fontsize=22)
plt.ylabel(r'Skill', fontsize = 22)
plt.legend(fontsize = 16)
plt.show()
# plt.savefig("./figures3/depth_skill.png", dpi=300, bbox_inches = 'tight',transparent=True)




# flux time series'
  
flux_nontriv = (np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[2][1])/(1.2*40*40)
flux_nontriv_f = (np.load("./data256_4000/fluxes/" + "PSI2_f" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PSI2_f" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2_f" + ".npz").items()[2][1])/(1.2*40*40)

flux_nontriv_recon = (np.load("./arrays/outfilenontriv.npz").items()[2][1]*3*np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[2][1])/(1.2*40*40)
flux_nontriv_f_recon = (np.load("./arrays/outfilefiltered_3layer.npz").items()[2][1]*3*np.load("./data256_4000/fluxes/" + "PSI2_f" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2_f" + ".npz").items()[2][1])/(1.2*40*40)


fig, ax = plt.subplots(figsize=(8, 4))
plt.plot(np.arange(200,400,0.25),flux_nontriv[0:800],color='#ff7f0e',alpha=0.5,label=r'Truth')
plt.plot(np.arange(200,400,0.25),flux_nontriv_recon[0:800],color='#ff7f0e',label=r'Prediction, S = %.3f' %skill(linearly_regressed(flux_nontriv_recon,flux_nontriv),flux_nontriv)[0][0])
plt.plot(np.arange(200,400,0.25),flux_nontriv_f[0:800],color='#d62728',alpha=0.5,label=r'Truth (filitered)')
plt.plot(np.arange(200,400,0.25),flux_nontriv_f_recon[0:800],color='#d62728',label=r'Prediction (filtered), S = %.3f' %skill(linearly_regressed(flux_nontriv_f_recon,flux_nontriv_f),flux_nontriv_f)[0][0])
plt.xlabel(r'Days', fontsize=16)
plt.ylabel(r'Flux', fontsize=16)
plt.legend(fontsize=12)
# plt.show()
plt.savefig("./figures3/filtervsnonfilter.png", dpi=300, bbox_inches = 'tight',transparent=True)



#scatter
# plt.figure(6)
# plt.scatter(flux_nontriv[::5],linearly_regressed(flux_nontriv_recon,flux_nontriv)[::5],marker = '.', s = 1, color='#ff7f0e')
# plt.plot([-8,2],[-8,2],":",color="black")
# plt.axis([-1,1,-1,1],'equal')
# plt.xlabel(r"True Flux", fontsize=26); plt.ylabel(r"Predicted Flux", fontsize=26)
# plt.ylim([-6,1])
# plt.xlim([-6,1])
# plt.yticks([-6,-4,-2,0],fontsize=20)
# plt.xticks([-6,-4,-2,0],fontsize=20)
# plt.gca().set_aspect('equal','box')
# plt.savefig("./figures3/fluxscatter.png", dpi=300, bbox_inches = 'tight', transparent=True)


