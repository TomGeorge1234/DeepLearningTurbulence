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

def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')













#single print of field

# PSI1 = np.load("./data256_4000/fields/" + "PSI1" + ".npz").items()
# PSI1 = PSI1[1][1]*0.1*PSI1[3][1]+PSI1[2][1]
# PV1 = np.load("./data256_4000/fields/" + "PV1" + ".npz").items()
# PV1 = PV1[1][1]*0.1*PV1[3][1]+PV1[2][1]
# PSI2 = np.load("./data256_4000/fields/" + "PSI2" + ".npz").items()
# PSI2 = PSI2[1][1]*0.1*PSI2[3][1]+PSI2[2][1]
# PV2 = np.load("./data256_4000/fields/" + "PV2" + ".npz").items()
# PV2 = PV2[1][1]*0.1*PV2[3][1]+PV2[2][1]
# # PV1_full = np.load("./data256_4000/fields/" + "PV1_fulldomain" + ".npz").items()
# # PV1_full = PV1_full[1][1]*0.1*PV1_full[3][1]+PV1_full[2][1]

# # flux_PV1 = np.load("./data256_4000/fluxes/" + "PV1_unchopped" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PV1_unchopped" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PV1_unchopped" + ".npz").items()[2][1]# flux_PV1_full = np.load("./data256_4000/" + "flux_PV1full_test" + ".npz").items()[0][1]# flux_full = np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[0][1]*3*np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[2][1]+np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[1][1]

# K = 300


# Z = PV1[K]
# fig, ax = plt.subplots(figsize=(4, 4))
# im = plt.imshow(Z, extent=[0, 1000, 0, 1000])
# plt.yticks([0,250,500,750,1000],size=16)
# plt.xticks([0,250,500,750,1000],size=16)
# plt.axis('off')
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
# plt.axis('off')
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
# plt.axis('off')
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
# plt.axis('off')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(im, cax=cax)
# ax = plt.axis()
# for font_objects in cbar.ax.yaxis.get_ticklabels():
#     font_objects.set_size(16)
# # plt.show()
# plt.savefig("./figures3/PSI2.png", dpi=300, bbox_inches = 'tight', transparent=True)


# Z = PV1_full[K]
# fig, ax = plt.subplots(figsize=(9, 9))
# im = plt.imshow(Z, extent=[0, 4000, 0, 4000])
# plt.yticks([0,1000,2000,3000,4000],size=16)
# plt.xticks([0,1000,2000,3000,4000],size=16)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(im, cax=cax)
# ax = plt.axis()
# for font_objects in cbar.ax.yaxis.get_ticklabels():
#     font_objects.set_size(16)
# # plt.show()
# plt.savefig("./figures3/PSI1_fulldomain.png", dpi=300, bbox_inches = 'tight', transparent=True)



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
# fig = plt.gcf()
# fig.set_size_inches(6,4)
# y = (R1,R2,R3,R4,R5,R6,R7,R8,R9)
# y_ = (R1_,R2_,R3_,R4_,R5_,R6_,R7_,R8_,R9_)
# x = (330,1000,3300,10000,18000,33000,56000,112000,224000)
# plt.scatter(x,y_,marker='.',label="Training")
# plt.plot(x,y_,alpha=0.7)
# plt.scatter(x,y,marker='.',label="Testing")
# plt.plot(x,y,alpha=0.7)
# plt.ylim(-0.05,0.95)
# plt.xlim(150,500000)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# ax.set_xscale('log')
# plt.xlabel(r'No. training images', fontsize=18)
# plt.ylabel(r'Skill', fontsize = 18)
# plt.legend(fontsize = 16)
# # plt.show()
# plt.savefig("./figures3/phase_space.png", dpi=300, bbox_inches = 'tight',transparent=True)

    












  
  
# skill chart for deep nets 


# flux_std = np.std(np.load("./data256_4000/" + "fluxes/PSI2" + ".npz").items()[0][1]) 

# path = '/Users/tomgeorge/Documents/Summer2018/CaltechSURF/QG/arrays/outfile'

# #calculations of the skills from the costs 
# S1 = np.mean(np.load(path+'1c2f_superlight.npz').items()[5][1][-34:])
# S2 = np.mean(np.load(path+'1c2f_light.npz').items()[5][1][-34:])
# S3 = np.mean(np.load(path+'1c2f.npz').items()[5][1][-34:])
# S4 = np.mean(np.load(path+'1c2f_heavy.npz').items()[5][1][-34:])
# S5 = np.mean(np.load(path+'2c2f_light.npz').items()[5][1][-34:])
# S6 = np.mean(np.load(path+'2c2f.npz').items()[5][1][-34:])
# S7 = np.mean(np.load(path+'3c2f_light.npz').items()[5][1][-34:])
# S8 = np.mean(np.load(path+'3c2f.npz').items()[5][1][-34:])
# S9 = np.mean(np.load(path+'3c2f_heavy.npz').items()[5][1][-34:])
# S10 = np.mean(np.load(path+'4c3f.npz').items()[5][1][-34:])
# S11 = np.mean(np.load(path+'4c3f_heavy.npz').items()[5][1][-34:])
# S12 = np.mean(np.load(path+'5c4f.npz').items()[5][1][-34:])
# S13 = np.mean(np.load(path+'6c4f.npz').items()[5][1][-34:])

# S1_ = np.mean(moving_average(np.load(path+'1c2f_superlight.npz').items()[4][1],160)[-3400:])
# S2_ = np.mean(moving_average(np.load(path+'1c2f_light.npz').items()[4][1],160)[-3400:])
# S3_ = np.mean(moving_average(np.load(path+'1c2f.npz').items()[4][1],160)[-3400:])
# S4_ = np.mean(moving_average(np.load(path+'1c2f_heavy.npz').items()[4][1],160)[-3400:])
# S5_ = np.mean(moving_average(np.load(path+'2c2f.npz').items()[4][1],160)[-3400:])
# S6_ = np.mean(moving_average(np.load(path+'2c2f.npz').items()[4][1],160)[-3400:])
# S7_ = np.mean(moving_average(np.load(path+'3c2f_light.npz').items()[4][1],160)[-3400:])
# S8_ = np.mean(moving_average(np.load(path+'3c2f.npz').items()[4][1],160)[-3400:])
# S9_ = np.mean(moving_average(np.load(path+'3c2f_heavy.npz').items()[4][1],160)[-3400:])
# S10_ = np.mean(moving_average(np.load(path+'4c3f.npz').items()[4][1],160)[-3400:])
# S11_ = np.mean(moving_average(np.load(path+'4c3f_heavy.npz').items()[4][1],160)[-3400:])
# S12_ = np.mean(moving_average(np.load(path+'5c4f.npz').items()[4][1],160)[-3400:])
# S13_ = np.mean(moving_average(np.load(path+'6c4f.npz').items()[4][1],160)[-3400:])


# ax = plt.figure().add_subplot(111)
# fig = plt.gcf()
# fig.set_size_inches(6,4)
# # y = (S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13)
# # y_ = (S1_,S2_,S3_,S4_,S5_,S6_,S7_,S8_,S9_,S10_,S11_,S12_,S13_)
# y = (S1,S2,S3,S6,S7,S8,S9,S11)
# y_ = (S1_,S2_,S3_,S6_,S7_,S8_,S9_,S11_)
# # x = (5349,20937,26161,30183,29291,40901,56866,87525,179167,245282,397823,538989,1000355)
# x = (5349,20937,26161,40901,56866,87525,179167,397823)
# plt.scatter(x,y_,marker='.',label="Training")
# plt.plot(x,y_,alpha=0.7)
# plt.scatter(x,y,marker='.',label="Testing")
# plt.plot(x,y,alpha=0.7)
# plt.ylim(-0.05,0.45)
# plt.xlim(3000,2000000)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# ax.set_xscale('log')
# plt.xlabel(r'Complexity (No. trainable parameters)', fontsize=18)
# plt.ylabel(r'Skill', fontsize = 18)
# plt.legend(fontsize = 16)
# # plt.show()
# plt.savefig("./figures3/depth_skill_bad.png", dpi=300, bbox_inches = 'tight',transparent=True)














# flux time series'
  
# flux_nontriv = (np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[2][1])/(1.2*40*40)
# flux_full = np.load("./data256_4000/fluxes/" + "PV1" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PV1" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PV1" + ".npz").items()[2][1]
# flux_triv = flux_full - flux_nontriv

# flux_nontriv_recon = (np.load("./arrays/outfile3c2f.npz").items()[2][1]*3*np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[2][1])/(1.2*40*40)

# #1f77b4
# # 2ca02c
# # ff7f0e orange
# fig, ax = plt.subplots(figsize=(8, 5.33))
# plt.plot(np.arange(200,400,0.25),flux_nontriv[0:800],color='#ff7f0e',alpha=1,label=r'Truth')
# plt.plot(np.arange(200,400,0.25),flux_full[0:800],color='#1f77b4',alpha=1,label=r'Truth')
# plt.plot(np.arange(200,400,0.25),flux_triv[0:800],color='#2ca02c',label=r'Prediction, S = %.3f' %skill(linearly_regressed(flux_nontriv_recon,flux_nontriv),flux_nontriv)[0][0])
# plt.xlabel(r'Days', fontsize=18)
# plt.ylabel(r'Flux', fontsize=18)
# plt.xticks([200,250,300,350,400],fontsize=14)
# plt.yticks([-8,-6,-4,-2,0,2,4,6,8,10],fontsize=14)
# plt.yticks(fontsize=14)
# # plt.legend(fontsize=14)
# # plt.show()
# plt.savefig("./figures3/fluxtimeseries.png", dpi=300, bbox_inches = 'tight',transparent=True)



# scatter
# fig, ax = plt.subplots(figsize=(8, 5.3))
# plt.scatter(flux_nontriv[::5],linearly_regressed(flux_nontriv_recon,flux_nontriv)[::5],marker = '.', s = 1, color='#ff7f0e')
# plt.plot([-8,2],[-8,2],":",color="black")
# plt.axis([-1,1,-1,1],'equal')
# plt.xlabel(r"True Flux", fontsize=24); plt.ylabel(r"Predicted Flux", fontsize=24)
# plt.ylim([-6,1])
# plt.xlim([-6,1])
# plt.yticks([-6,-4,-2,0],fontsize=18)
# plt.xticks([-6,-4,-2,0],fontsize=18)

# fig  = plt.gcf()
# fwidth = fig.get_figwidth()
# fheight = fig.get_figheight()
# bb = ax.get_position()
# axwidth = fwidth * (bb.x1 - bb.x0)
# axheight = fheight * (bb.y1 - bb.y0)
# if axwidth > axheight:
#     narrow_by = (axwidth - axheight) / fwidth
#     bb.x0 += narrow_by / 2
#     bb.x1 -= narrow_by / 2
# elif axheight > axwidth:
#     shrink_by = (axheight - axwidth) / fheight
#     bb.y0 += shrink_by / 2
#     bb.y1 -= shrink_by / 2
# ax.set_position(bb)

# # plt.show()
# plt.savefig("./figures3/fluxscatter.png", dpi=300, bbox_inches = 'tight', transparent=True)












#Correlations 

# V1 = np.load("./data256_4000/fields/" + "V1" + ".npz").items()
# V1 = V1[1][1]*3*V1[3][1] + V1[2][1]
# V2 = np.load("./data256_4000/fields/" + "V2" + ".npz").items()
# V2 = V2[1][1]*3*V2[3][1] + V2[2][1]
# U1 = np.load("./data256_4000/fields/" + "U1" + ".npz").items()
# U1 = U1[1][1]*3*U1[3][1] + U1[2][1]
# U2 = np.load("./data256_4000/fields/" + "U2" + ".npz").items()
# U2 = U2[1][1]*3*U2[3][1] + U2[2][1]

# EKE = np.average(np.multiply(V1,V1) + np.multiply(U1,U1) + 5*np.multiply(V2,V2) + 5*np.multiply(U2,U2),(1,2))
# EKE = (EKE/np.std(EKE))[0:1000]
# flux_full = np.reshape(np.load("./data256_4000/fluxes/" + "PV1" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PV1" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PV1" + ".npz").items()[2][1],(-1))[0:1000]

# def lag_correlate(x,y,number):
#     R = np.zeros(number)
#     for i in range(number):
#         x1 = x[:-(i+1)]
#         y1 = y[(i+1):]
#         R[i] = stats.mstats.linregress(x1,y1)[2]
#     return R

# autocorrelation = lag_correlate(flux_full,flux_full,400)
# EKEcorrelation =  lag_correlate(flux_full,EKE,400)

# fig, ax = plt.subplots(figsize=(8, 5))
# plt.plot(np.arange(0,60,0.25),autocorrelation[:240],color='#1f77b4',alpha=1,label=r'PV Flux Autocorrelation')
# plt.plot(np.arange(0,60,0.25),EKEcorrelation[:240],color='#d62728',alpha=1,label=r'PV vs EKE Lag correlation')
# plt.xlabel(r'Lag days', fontsize=18)
# plt.ylabel(r'Corrrelation coefficient', fontsize=18)
# plt.xticks([0,10,20,30,40,50,60],fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=14)
# # plt.show()
# plt.savefig("./figures3/correlations.png", dpi=300, bbox_inches = 'tight',transparent=True)









#Spectra 


# flux_nontriv = np.reshape((np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[2][1])/(1.2*40*40),(-1))
# flux_nontriv_recon = np.reshape((np.load("./arrays/outfile3c2f.npz").items()[2][1]*3*np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[2][1])/(1.2*40*40),(-1))
# flux_nontriv = flux_nontriv/np.std(flux_nontriv)
# flux_nontriv_recon = flux_nontriv_recon/np.std(flux_nontriv_recon)

# P = scipy.signal.welch(flux_nontriv,fs=4)
# P_recon = scipy.signal.welch(flux_nontriv_recon,fs=4)
# maxi = np.max(P[1])
# Power = P[1]/maxi
# Power_recon = P_recon[1]/maxi


# fig, ax = plt.subplots(figsize=(8, 5.33))
# plt.plot(P_recon[0][:40],Power_recon[:40],color='#ff7f0e',alpha=1,label=r'Prediction')
# plt.plot(P[0][:40],Power[:40],color='#ff7f0e',alpha=0.5,label=r'Truth')
# plt.xlabel(r'Frequency (Days$^{-1}$)', fontsize=24)
# plt.ylabel(r'Relative power', fontsize=24)
# # plt.xticks([200,250,300,350,400],fontsize=14)
# # plt.yticks([-8,-6,-4,-2,0,2,4,6,8,10],fontsize=14)
# plt.yticks(fontsize=18)
# plt.xticks([10**-2,10**-1,10**0],fontsize=18)
# plt.legend(fontsize=18)

# fig  = plt.gcf()
# fwidth = fig.get_figwidth()
# fheight = fig.get_figheight()
# bb = ax.get_position()
# axwidth = fwidth * (bb.x1 - bb.x0)
# axheight = fheight * (bb.y1 - bb.y0)
# if axwidth > axheight:
#     narrow_by = (axwidth - axheight) / fwidth
#     bb.x0 += narrow_by / 2
#     bb.x1 -= narrow_by / 2
# elif axheight > axwidth:
#     shrink_by = (axheight - axwidth) / fheight
#     bb.y0 += shrink_by / 2
#     bb.y1 -= shrink_by / 2
# ax.set_position(bb)

# ax.set_xscale('log')
# # plt.show()
# plt.savefig("./figures3/spectra.png", dpi=300, bbox_inches = 'tight',transparent=True)
















#Histogram
# flux_nontriv = np.reshape((np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[2][1])/(1.2*40*40),(-1))
# flux_nontriv_recon = np.reshape((np.load("./arrays/outfile3c2f.npz").items()[2][1]*3*np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[2][1])/(1.2*40*40),(-1))

# fig, ax = plt.subplots(figsize=(8, 5.33))


# # Generate some random data
# hist1,bins1 = np.histogram(flux_nontriv_recon,bins=np.arange(-6,1,0.5))
# hist2,bins2 = np.histogram(flux_nontriv,bins=np.arange(-6,1,0.5))

# ax.step(bins1[:-1],hist1/16000,'#ff7f0e',linestyle='-',linewidth=1,where='mid')
# ax.bar(bins1[:-1],hist1/16000,width=0.5,linewidth=0,facecolor='#ff7f0e',label='Prediction')
# ax.bar(bins2[:-1],hist2/16000,width=0.5,linewidth=0,facecolor=[255/256,191/256,135/256],alpha = 0.85, label='Truth')
         
# plt.xlabel(r'Flux ', fontsize=24)
# plt.ylabel(r'Probability', fontsize=24)

# fig  = plt.gcf()
# fwidth = fig.get_figwidth()
# fheight = fig.get_figheight()
# bb = ax.get_position()
# axwidth = fwidth * (bb.x1 - bb.x0)
# axheight = fheight * (bb.y1 - bb.y0)
# if axwidth > axheight:
#     narrow_by = (axwidth - axheight) / fwidth
#     bb.x0 += narrow_by / 2
#     bb.x1 -= narrow_by / 2
# elif axheight > axwidth:
#     shrink_by = (axheight - axwidth) / fheight
#     bb.y0 += shrink_by / 2
#     bb.y1 -= shrink_by / 2
# ax.set_position(bb)

# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.legend(fontsize=18)
# # plt.show()
# plt.savefig("./figures3/histogram.png", dpi=300, bbox_inches = 'tight',transparent=True)








#Training
    
N_av = 10
N_crop1 = 897
N_crop2 = 897
training_skill = moving_average(np.load("./arrays/outfile3c2f.npz").items()[4][1],N_av)[N_crop1:-N_crop2][::10]
testing_skill = moving_average(np.load("./arrays/outfile3c2f.npz").items()[5][1][1:],N_av)
training_epochs = np.linspace(0 + (N_av - 1 + N_crop1)*(1/1120),16.925 - (N_av - 1 + N_crop2)*(1/1120),len(training_skill))
testing_epochs = np.linspace(0 + (N_av - 1)*(5/56),16.925 - (N_av - 1)*(5/56),len(testing_skill))

#1f77b4
# 2ca02c
# ff7f0e orange
fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(training_epochs,training_skill,color='#1f77b4',alpha=1,label=r'Training data', linewidth = 0.7 )
plt.plot(testing_epochs,testing_skill,color='#ff7f0e',alpha=1,label=r'Testing data', linewidth = 2 )
plt.xlabel(r'Epochs', fontsize=22)
plt.ylabel(r'Skill', fontsize=22)
plt.xticks([0,2,4,6,8,10,12,14,16],fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
# plt.show()
plt.savefig("./figures3/training.png", dpi=300, bbox_inches = 'tight',transparent=True)
