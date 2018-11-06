import matplotlib.pyplot as plt
import matplotlib
import numpy as np 
import matplotlib.cm as cm
import scipy.signal
from eofs.standard import Eof
from scipy import stats
from time import time 
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# PSI1 = np.load("./data256_4000/fields/" + "PSI1" + ".npz").items()
# PSI1 = PSI1[1][1]*0.1*PSI1[3][1]+PSI1[2][1]
# PV1 = np.load("./data256_4000/fields/" + "PV1" + ".npz").items()
# PV1 = PV1[1][1]*0.1*PV1[3][1]+PV1[2][1]
# PSI2 = np.load("./data256_4000/fields/" + "PSI2" + ".npz").items()
# PSI2 = PSI2[1][1]*0.1*PSI2[3][1]+PSI2[2][1]
# PV2 = np.load("./data256_4000/fields/" + "PV2" + ".npz").items()
# PV2 = PV2[1][1]*0.1*PV2[3][1]+PV2[2][1]

# flux_PV1 = np.load("./data256_4000/fluxes/" + "PV1_unchopped" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PV1_unchopped" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PV1_unchopped" + ".npz").items()[2][1]# flux_PV1_full = np.load("./data256_4000/" + "flux_PV1full_test" + ".npz").items()[0][1]# flux_full = np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[0][1]*3*np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[2][1]+np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[1][1]

# K = 300


# Z = PV1[K]
# fig, ax = plt.subplots(figsize=(4, 4))
# im = plt.imshow(Z, extent=[0, 1000, 0, 1000])
# plt.yticks([0,250,500,750,1000],size=16)
# plt.xticks([0,250,500,750,1000],size=16)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(im, cax=cax)
# ax = plt.axis()
# for font_objects in cbar.ax.yaxis.get_ticklabels():
#     font_objects.set_size(16)
# # plt.show()
# plt.savefig("./figures3/PV1.png", dpi=300, bbox_inches = 'tight', transparent=True)

# Z = PSI1[K]
# fig, ax = plt.subplots(figsize=(4, 4))
# im = plt.imshow(Z, extent=[0, 1000, 0, 1000])
# plt.yticks([0,250,500,750,1000],size=16)
# plt.xticks([0,250,500,750,1000],size=16)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(im, cax=cax)
# ax = plt.axis()
# for font_objects in cbar.ax.yaxis.get_ticklabels():
#     font_objects.set_size(16)
# # plt.show()
# plt.savefig("./figures3/PSI1.png", dpi=300, bbox_inches = 'tight', transparent=True)

# Z = PV2[K]
# fig, ax = plt.subplots(figsize=(4, 4))
# im = plt.imshow(Z, extent=[0, 1000, 0, 1000])
# plt.yticks([0,250,500,750,1000],size=16)
# plt.xticks([0,250,500,750,1000],size=16)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(im, cax=cax)
# ax = plt.axis()
# for font_objects in cbar.ax.yaxis.get_ticklabels():
#     font_objects.set_size(16)
# # plt.show()
# plt.savefig("./figures3/PV2.png", dpi=300, bbox_inches = 'tight', transparent=True)

# Z = PSI2[K]
# fig, ax = plt.subplots(figsize=(4, 4))
# im = plt.imshow(Z, extent=[0, 1000, 0, 1000])
# plt.yticks([0,250,500,750,1000],size=16)
# plt.xticks([0,250,500,750,1000],size=16)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(im, cax=cax)
# ax = plt.axis()
# for font_objects in cbar.ax.yaxis.get_ticklabels():
#     font_objects.set_size(16)
# # plt.show()
# plt.savefig("./figures3/PSI2.png", dpi=300, bbox_inches = 'tight', transparent=True)



# fig, ax = plt.subplots(figsize=(12*0.8, 4*0.8))
# plt.plot(np.arange(0,400,0.25),flux_PV1[0:1600],color='#1f77b4',alpha=1,label=r'Truth')
# # plt.plot(np.arange(200,400,0.25),flux_nontriv_recon[0:800],color='#ff7f0e',label=r'Prediction, S = %.3f' %skill(linearly_regressed(flux_nontriv_recon,flux_nontriv),flux_nontriv)[0][0])
# plt.xlabel(r'Days', fontsize=18)
# plt.ylabel(r'Flux: $\overline{v_{1}\prime q_{1}\prime}$', fontsize=18)
# plt.xticks([0,100,200,300,400],fontsize=16)
# plt.yticks(fontsize=16)
# # plt.legend(fontsize=14)
# # plt.show()
# plt.savefig("./figures3/full_unchopped.png", dpi=300, bbox_inches = 'tight',transparent=True)



















#phase space plot 

# def moving_average(data_set, periods=3):
#     weights = np.ones(periods) / periods
#     return np.convolve(data_set, weights, mode='valid')

# flux_std = np.std(np.load("./data256_4000/" + "fluxes/PSI2" + ".npz").items()[0][1]) 

# path = '/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfilenoimages'

# R1 = np.mean(np.load(path + '330.npz').items()[5][1][-1:])
# R2 = np.mean(np.load(path + '1000.npz').items()[5][1][-1:])
# R3 = np.mean(np.load(path + '3300.npz').items()[5][1][-1:])
# R4 = np.mean(np.load(path + '10000.npz').items()[5][1][-3:])
# R5 = np.mean(np.load(path + '18000.npz').items()[5][1][-5:])
# R6 = np.mean(np.load(path + '33000.npz').items()[5][1][-10:])
# R7 = np.mean(np.load(path + '56000.npz').items()[5][1][-17:])
# R8 = np.mean(np.load(path +'112000.npz').items()[5][1][-34:])
# R9 = np.mean(np.load(path + '224000.npz').items()[5][1][-67:])

# R1_ = np.mean(moving_average(np.load(path + '330.npz').items()[4][1],160)[-10:])
# R2_ = np.mean(moving_average(np.load(path + '1000.npz').items()[4][1],160)[-30:])
# R3_ = np.mean(moving_average(np.load(path + '3300.npz').items()[4][1],160)[-99:])
# R4_ = np.mean(moving_average(np.load(path + '10000.npz').items()[4][1],160)[-300:])
# R5_ = np.mean(moving_average(np.load(path + '18000.npz').items()[4][1],160)[-540:])
# R6_ = np.mean(moving_average(np.load(path + '33000.npz').items()[4][1],160)[-990:])
# R7_ = np.mean(moving_average(np.load(path + '56000.npz').items()[4][1],160)[-1680:])
# R8_ = np.mean(moving_average(np.load(path + '112000.npz').items()[4][1],160)[-3456:])
# R9_ = np.mean(moving_average(np.load(path + '224000.npz').items()[4][1],160)[-6720:])

# ax = plt.figure().add_subplot(111)
# y = (R1,R2,R3,R4,R5,R6,R7,R8,R9)
# y_ = (R1_,R2_,R3_,R4_,R5_,R6_,R7_,R8_,R9_)
# x = (330,1000,3300,10000,18000,33000,56000,112000,224000)
# plt.scatter(x,y_,marker='.',label="Training")
# plt.plot(x,y_,alpha=0.7)
# plt.scatter(x,y,marker='.',label="Testing")
# plt.plot(x,y,alpha=0.7)
# plt.ylim(-0.05,0.95)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# ax.set_xscale('log')
# plt.xlabel(r'No. training images', fontsize=18)
# plt.ylabel(r'Skill', fontsize = 18)
# plt.legend(fontsize = 16)
# # plt.show()
# plt.savefig("./figures3/phase_space.png", dpi=300, bbox_inches = 'tight',transparent=True)

    












  
  
# skill chart for deep nets 
def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')



flux_std = np.std(np.load("./data256_4000/" + "fluxes/PSI2" + ".npz").items()[0][1]) 

path = '/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfile'

#calculations of the skills from the costs 
S1 = np.mean(np.load(path+'1c2f_superlight.npz').items()[5][1][-34:])
S2 = np.mean(np.load(path+'1c2f_light.npz').items()[5][1][-34:])
S3 = np.mean(np.load(path+'1c2f.npz').items()[5][1][-34:])
S4 = np.mean(np.load(path+'1c2f_heavy.npz').items()[5][1][-34:])
S5 = np.mean(np.load(path+'2c2f_light.npz').items()[5][1][-34:])
S6 = np.mean(np.load(path+'2c2f.npz').items()[5][1][-34:])
S7 = np.mean(np.load(path+'3c2f_light.npz').items()[5][1][-34:])
S8 = np.mean(np.load(path+'3c2f.npz').items()[5][1][-34:])
S9 = np.mean(np.load(path+'3c2f_heavy.npz').items()[5][1][-34:])
S10 = np.mean(np.load(path+'4c3f.npz').items()[5][1][-34:])
S11 = np.mean(np.load(path+'4c3f_heavy.npz').items()[5][1][-34:])
S12 = np.mean(np.load(path+'5c4f.npz').items()[5][1][-34:])
S13 = np.mean(np.load(path+'6c4f.npz').items()[5][1][-34:])

S1_ = np.mean(moving_average(np.load(path+'1c2f_superlight.npz').items()[4][1],160)[-3400:])
S2_ = np.mean(moving_average(np.load(path+'1c2f_light.npz').items()[4][1],160)[-3400:])
S3_ = np.mean(moving_average(np.load(path+'1c2f.npz').items()[4][1],160)[-3400:])
S4_ = np.mean(moving_average(np.load(path+'1c2f_heavy.npz').items()[4][1],160)[-3400:])
S5_ = np.mean(moving_average(np.load(path+'2c2f.npz').items()[4][1],160)[-3400:])
S6_ = np.mean(moving_average(np.load(path+'2c2f.npz').items()[4][1],160)[-3400:])
S7_ = np.mean(moving_average(np.load(path+'3c2f_light.npz').items()[4][1],160)[-3400:])
S8_ = np.mean(moving_average(np.load(path+'3c2f.npz').items()[4][1],160)[-3400:])
S9_ = np.mean(moving_average(np.load(path+'3c2f_heavy.npz').items()[4][1],160)[-3400:])
S10_ = np.mean(moving_average(np.load(path+'4c3f.npz').items()[4][1],160)[-3400:])
S11_ = np.mean(moving_average(np.load(path+'4c3f_heavy.npz').items()[4][1],160)[-3400:])
S12_ = np.mean(moving_average(np.load(path+'5c4f.npz').items()[4][1],160)[-3400:])
S13_ = np.mean(moving_average(np.load(path+'6c4f.npz').items()[4][1],160)[-3400:])


ax = plt.figure().add_subplot(111)
y = (S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13)
y_ = (S1,S2_,S3_,S4_,S5_,S6_,S7_,S8_,S9_,S10_,S11_,S12_,S13_)
x = (9000,20937,26161,30183,33161,40901,56866,87525,179167,245282,397823,538989,1000355)
plt.scatter(x,y_,marker='.',label="Training")
plt.plot(x,y_,alpha=0.7)
plt.scatter(x,y,marker='.',label="Testing")
plt.plot(x,y,alpha=0.7)
plt.ylim(0.15,0.4)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_xscale('log')
plt.xlabel(r'Complexity (No. trainable parameters)', fontsize=18)
plt.ylabel(r'Skill', fontsize = 18)
plt.legend(fontsize = 16)
plt.show()
# plt.savefig("./figures3/depth_skill.png", dpi=300, bbox_inches = 'tight',transparent=True)














# flux time series'
  
# flux_nontriv = (np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[2][1])/(1.2*40*40)
# flux_full = np.load("./data256_4000/fluxes/" + "PV1" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PV1" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PV1" + ".npz").items()[2][1]
# flux_triv = flux_full - flux_nontriv

# flux_nontriv_recon = (np.load("./arrays/outfile3c2f.npz").items()[2][1]*3*np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[2][1])/(1.2*40*40)

# # 1f77b4
# # 2ca02c
# # ff7f0e orange
# fig, ax = plt.subplots(figsize=(8, 4))
# plt.plot(np.arange(200,400,0.25),flux_full[0:800],color='#1f77b4',alpha=1,label=r'Truth')
# # plt.plot(np.arange(200,400,0.25),flux_nontriv_recon[0:800],color='#ff7f0e',label=r'Prediction, S = %.3f' %skill(linearly_regressed(flux_nontriv_recon,flux_nontriv),flux_nontriv)[0][0])
# plt.xlabel(r'Days', fontsize=18)
# plt.ylabel(r'Flux', fontsize=18)
# plt.xticks([200,250,300,350,400],fontsize=14)
# plt.yticks(fontsize=14)
# # plt.legend(fontsize=14)
# # plt.show()
# plt.savefig("./figures3/full.png", dpi=300, bbox_inches = 'tight',transparent=True)



#scatter
# fig, ax = plt.subplots(figsize=(8, 4))
# plt.scatter(flux_nontriv[::5],linearly_regressed(flux_nontriv_recon,flux_nontriv)[::5],marker = '.', s = 1, color='#ff7f0e')
# plt.plot([-8,2],[-8,2],":",color="black")
# plt.axis([-1,1,-1,1],'equal')
# plt.xlabel(r"True Flux", fontsize=18); plt.ylabel(r"Predicted Flux", fontsize=18)
# plt.ylim([-6,1])
# plt.xlim([-6,1])
# plt.yticks([-6,-4,-2,0],fontsize=14)
# plt.xticks([-6,-4,-2,0],fontsize=14)
# plt.gca().set_aspect('equal','box')
# # plt.show()
# plt.savefig("./figures3/fluxscatter.png", dpi=300, bbox_inches = 'tight', transparent=True)


