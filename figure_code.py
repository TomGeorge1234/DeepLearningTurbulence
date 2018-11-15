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


def R_squared(yp,yt):
    return stats.mstats.linregress(yp,yt)[2]

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
# PSI1 = PSI1[1][1]*0.1
# PV1 = np.load("./data256_4000/fields/" + "PV1" + ".npz").items()
# PV1 = PV1[1][1]*0.1
# PSI2 = np.load("./data256_4000/fields/" + "PSI2" + ".npz").items()
# PSI2 = PSI2[1][1]*0.1
# PV2 = np.load("./data256_4000/fields/" + "PV2" + ".npz").items()
# PV2 = PV2[1][1]*0.1
# PV1_full = np.load("./data256_4000/fields/" + "PV1_fulldomain" + ".npz").items()
# PV1_full = PV1_full[1][1]*0.1

# # flux_PV1 = np.load("./data256_4000/fluxes/" + "PV1_unchopped" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PV1_unchopped" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PV1_unchopped" + ".npz").items()[2][1]# flux_PV1_full = np.load("./data256_4000/" + "flux_PV1full_test" + ".npz").items()[0][1]# flux_full = np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[0][1]*3*np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[2][1]+np.load("./data256_4000/" + "flux_PV1_test" + ".npz").items()[1][1]

# # K = 300+9*1000
# K1 = 300

# Z = PV1[K]
# fig, ax = plt.subplots(figsize=(4, 4))
# im = plt.imshow(Z, extent=[0, 1000, 0, 1000])
# plt.yticks([0,250,500,750,1000],size=16)
# plt.xticks([0,250,500,750,1000],size=16)
# plt.axis('off')
# divider = make_axes_locatable(ax)
# # cax = divider.append_axes("right", size="5%", pad=0.05)
# # cbar = plt.colorbar(im, cax=cax)
# # ax = plt.axis()
# # for font_objects in cbar.ax.yaxis.get_ticklabels():
#     # font_objects.set_size(16)
# # plt.show()
# plt.savefig("./figures3/PV1.png", dpi=300, bbox_inches = 'tight', transparent=True)

# Z = PSI1[K]
# fig, ax = plt.subplots(figsize=(4, 4))
# im = plt.imshow(Z, extent=[0, 1000, 0, 1000])
# plt.yticks([0,250,500,750,1000],size=16)
# plt.xticks([0,250,500,750,1000],size=16)
# plt.axis('off')
# divider = make_axes_locatable(ax)
# # cax = divider.append_axes("right", size="5%", pad=0.05)
# # cbar = plt.colorbar(im, cax=cax)
# # ax = plt.axis()
# # for font_objects in cbar.ax.yaxis.get_ticklabels():
#     # font_objects.set_size(16)
# # plt.show()
# plt.savefig("./figures3/PSI1.png", dpi=300, bbox_inches = 'tight', transparent=True)

# Z = PV2[K]
# fig, ax = plt.subplots(figsize=(4, 4))
# im = plt.imshow(Z, extent=[0, 1000, 0, 1000])
# plt.yticks([0,250,500,750,1000],size=16)
# plt.xticks([0,250,500,750,1000],size=16)
# plt.axis('off')
# divider = make_axes_locatable(ax)
# # cax = divider.append_axes("right", size="5%", pad=0.05)
# # cbar = plt.colorbar(im, cax=cax)
# # ax = plt.axis()
# # for font_objects in cbar.ax.yaxis.get_ticklabels():
#     # font_objects.set_size(16)
# # plt.show()
# plt.savefig("./figures3/PV2.png", dpi=300, bbox_inches = 'tight', transparent=True)

# Z = PSI2[K]
# fig, ax = plt.subplots(figsize=(4, 4))
# im = plt.imshow(Z, extent=[0, 1000, 0, 1000])
# plt.yticks([0,250,500,750,1000],size=16)
# plt.xticks([0,250,500,750,1000],size=16)
# plt.axis('off')
# divider = make_axes_locatable(ax)
# # cax = divider.append_axes("right", size="5%", pad=0.05)
# # cbar = plt.colorbar(im, cax=cax)
# # ax = plt.axis()
# # for font_objects in cbar.ax.yaxis.get_ticklabels():
# #     font_objects.set_size(16)
# # plt.show()
# plt.savefig("./figures3/PSI2.png", dpi=300, bbox_inches = 'tight', transparent=True)


# Z = PV1_full[K1]
# fig, ax = plt.subplots(figsize=(9, 9))
# im = plt.imshow(Z, extent=[0, 4000, 0, 4000])
# plt.yticks([0,1000,2000,3000,4000],size=16)
# plt.xticks([0,1000,2000,3000,4000],size=16)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="10%", pad=0.5)
# cbar = plt.colorbar(im, cax=cax)
# ax = plt.axis()
# for font_objects in cbar.ax.yaxis.get_ticklabels():
#     font_objects.set_size(30)
# # plt.show()
# plt.savefig("./figures3/PV1_fulldomain.png", dpi=300, bbox_inches = 'tight', transparent=True)
# 


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
# plt.ylim(-0.15,1.05)
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
# plt.xlim(3000,1000000)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# ax.set_xscale('log')
# plt.xlabel(r'CNN Complexity (No. trainable parameters)', fontsize=18)
# plt.ylabel(r'Skill', fontsize = 18)
# plt.legend(fontsize = 16)
# # plt.show()
# plt.savefig("./figures3/depth_skill.png", dpi=300, bbox_inches = 'tight',transparent=True)














# flux time series'
  
flux_nontriv = (np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[2][1])/(1.2*40*40)
flux_full = np.load("./data256_4000/fluxes/" + "PV1" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PV1" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PV1" + ".npz").items()[2][1]
flux_triv = flux_full - flux_nontriv
flux_nontriv_recon = (np.load("./arrays/outfile3c2f.npz").items()[2][1]*3*np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[2][1])/(1.2*40*40)
EKE = np.load("./data256_4000/fluxes/" + "EKE" + ".npz").items()[0][1]

flux_nontriv = (flux_nontriv-np.mean(flux_nontriv))/np.std(flux_nontriv)
flux_nontriv_recon = (flux_nontriv_recon-np.mean(flux_nontriv_recon))/np.std(flux_nontriv_recon)
EKE = (EKE - np.mean(EKE))/np.std(EKE)

# # #1f77b4 blue
# # # 2ca02c green
# # # d62728 red
# # # ff7f0e orange

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(np.arange(200,600,0.25),flux_nontriv[0:1600],color='#ff7f0e',alpha=1)
ax1.set_xlabel(r'Days', fontsize=18)
ax1.set_ylabel(r'Coupled Flux', fontsize=18, color = '#ff7f0e')
ax1.tick_params('y',colors='#ff7f0e',labelsize=16)
ax1.tick_params('x',labelsize=16)
ax1.set_ylim([-3,3])
ax2 = ax1.twinx()
ax2.plot(np.arange(200,600,0.25),EKE[0:1600],color='#d62728',alpha=1)
ax2.set_ylabel(r'Eddy Kinetic Energy', fontsize=18, color='#d62728')
ax2.tick_params('y',colors='#d62728',labelsize=16)
ax2.set_ylim([-2.5,2.5])
plt.xticks([200,300,400,500,600])
# plt.show()
plt.savefig("./figures3/fluxtimeseries.png", dpi=300, bbox_inches = 'tight',transparent=True)


# fig, ax = plt.subplots(figsize=(8, 5))
# plt.plot(np.arange(200,400,0.25),EKE[0:800]/10**9,color='#d62728',alpha=1)
# plt.xlabel(r'Days', fontsize=18)
# plt.ylabel(r'EKE (\smallfont{$\times 10^{-9}$})', fontsize=18)
# plt.xticks([200,250,300,350,400],fontsize=14)
# plt.yticks(fontsize=14)
# plt.yticks(fontsize=14)
# # plt.legend(fontsize=14,loc=2)
# # plt.show()
# plt.savefig("./figures3/EKE.png", dpi=300, bbox_inches = 'tight',transparent=True)


# fig, ax = plt.subplots(figsize=(15, 4))
# plt.plot(np.arange(200,600,0.25),flux_nontriv[0:1600],color='#ff7f0e',alpha=0.5, label = r'Truth')
# plt.plot(np.arange(200,600,0.25),flux_nontriv_recon[0:1600],color='#ff7f0e',alpha=1, label = r'Prediction, Skill: %.3f, $R^{2}$: %.2f' %(skill(linearly_regressed(flux_nontriv_recon,flux_nontriv),flux_nontriv),R_squared(linearly_regressed(flux_nontriv_recon,flux_nontriv),flux_nontriv)))
# plt.xlabel(r'Days', fontsize=18)
# plt.ylabel(r'Coupled Flux', fontsize=18)
# plt.xticks([200,250,300,350,400,450,500,550,600],fontsize=14)
# plt.yticks([-2,-1,-0,1,2],fontsize=14)
# plt.yticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=14,loc=2)
# # plt.show()
# plt.savefig("./figures3/nontrivrecon.png", dpi=300, bbox_inches = 'tight',transparent=True)



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

# ax.text(0.05, 0.95, r'$R^{2} = %.2f$' %R_squared(linearly_regressed(flux_nontriv_recon,flux_nontriv),flux_nontriv), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, fontsize=18)

# # plt.show()
# plt.savefig("./figures3/fluxscatter.png", dpi=300, bbox_inches = 'tight', transparent=True)












#Correlations 

# EKE = np.load("./data256_4000/fluxes/" + "EKE" + ".npz").items()[0][1][:1000]
# flux_full = np.reshape(np.load("./data256_4000/fluxes/" + "PV1" + ".npz").items()[1][1],(-1))[:1000]
# flux_nontriv = np.reshape(np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[1][1],(-1))[:1000]
# PSI2 = np.load("./data256_4000/fields/" + "PSI2" + ".npz").items()[1][1][:1000]


# #this shift function rolls the data around the x axis by 1 i.e. shift([a,b,c,d]) = [d,a,b,c] therefore d_arrray/dx = (array - shift(array))/dx
# def shift(x):
#     return np.roll(x,1,2)


# dx = (1000/64)
# dt = 1/4
# U = 20 

# #load fields 
# U1 = np.load("./data256_4000/fields/" + "U1" + ".npz").items()
# U2 = np.load("./data256_4000/fields/" + "U2" + ".npz").items()
# V1 = np.load("./data256_4000/fields/" + "V1" + ".npz").items()
# V2 = np.load("./data256_4000/fields/" + "V2" + ".npz").items()

# #unnormalise them
# U1 = U1[1][1]*0.1*U1[3][1] + U1[2][1]
# U2 = U2[1][1]*0.1*U2[3][1] + U2[2][1]
# V1 = V1[1][1]*0.1*V1[3][1] + V1[2][1]
# V2 = V2[1][1]*0.1*V2[3][1] + V2[2][1]

# #define EKE(x,y,t)
# def EKE(U1,U2,V1,V2):
#     return U1**2+5*U2**2+V1**2+5*V2**2

# # calculate UdEKEdx then average over the box (ie average over dimensions (1,2) leaving the time dimension[0])
# UdEKEdx = np.average( U*(EKE(U1,U2,V1,V2) - shift(EKE(U1,U2,V1,V2)))[:,:,1:-1]/dx ,(1,2))[1:]
# # calculate EKE(t) by average over dimensions (1,2) then find the time derivative
# dEKEdt = (np.average(EKE(U1,U2,V1,V2),(1,2))[1:] - np.average(EKE(U1,U2,V1,V2),(1,2))[:-1])/dt

# DEKEDt = (dEKEdt + UdEKEdx)[:999] 

# def lag_correlate(x,y,number):
#     R = np.zeros(number)
#     for i in range(number):
#         x1 = x[:-(i+1)]
#         y1 = y[(i+1):]
#         R[i] = stats.mstats.linregress(x1,y1)[2]
#     return R



# fullautocorrelation = lag_correlate(flux_full,flux_full,400)
# coupledautocorrelation = lag_correlate(flux_nontriv,flux_nontriv,400)
# PSI2autocorrelation = lag_correlate(PSI2,PSI2,400)
# EKEfullcorrelation =  lag_correlate(flux_full,EKE,400)
# EKEcoupledcorrelation =  lag_correlate(flux_nontriv,EKE,400)
# # DEKEDtcoupledcorrelation = lag_correlate(flux_nontriv[1:],DEKEDt,400)


# # # #1f77b4 blue
# # # # 2ca02c green
# # # # d62728 red
# # # # ff7f0e orange

# fig, ax = plt.subplots(figsize=(8, 4))
# plt.plot(np.arange(0,60,0.25),coupledautocorrelation[:240],color='#ff7f0e',alpha=1,label=r'Coupled PV Flux $\bigotimes$ Coupled PV Flux')
# plt.plot(np.arange(0,60,0.25),PSI2autocorrelation[:240],color='#2ca02c',alpha=1,label=r'PSI2 snapshots $\bigotimes$ PSI2 snapshots')
# plt.plot(np.arange(0,60,0.25),EKEfullcorrelation[:240],color='#d62728',alpha=1,label=r'Full PV $\bigotimes$ EKE')
# plt.xlabel(r'Lag days', fontsize=18)
# plt.ylabel(r'Corrrelation coefficient', fontsize=18)
# plt.xticks([0,10,20,30,40,50,60],fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=12)
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
# plt.ylabel(r'Relative Spectral Power Density', fontsize=24)
# # plt.xticks([200,250,300,350,400],fontsize=14)
# # plt.yticks([-8,-6,-4,-2,0,2,4,6,8,10],fontsize=14)
# plt.yticks(fontsize=18)
# plt.xticks([10**-2,10**-1,10**0],fontsize=18)
# plt.legend(fontsize=18)
# plt.xlim([10**-2,0.6])

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
# ax.set_yscale('log')
# # plt.show()
# plt.savefig("./figures3/spectra.png", dpi=300, bbox_inches = 'tight',transparent=True)
















#Histogram
# flux_nontriv = np.reshape((np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[1][1]*3*np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[2][1])/(1.2*40*40),(-1))
# flux_nontriv_recon = np.reshape((np.load("./arrays/outfile3c2f.npz").items()[2][1]*3*np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[3][1] + np.load("./data256_4000/fluxes/" + "PSI2" + ".npz").items()[2][1])/(1.2*40*40),(-1))

# fig, ax = plt.subplots(figsize=(8, 5.33))


# # Generate some random data
# hist1,bins1 = np.histogram(flux_nontriv_recon,bins=np.arange(-6,1,0.05))
# hist2,bins2 = np.histogram(flux_nontriv,bins=np.arange(-6,1,0.05))

# ax.step(bins1[:-1],hist1/16000,'#ff7f0e',linestyle='-',linewidth=1,where='mid')
# ax.bar(bins1[:-1],hist1/16000,width=0.05
#         ,linewidth=0,facecolor='#ff7f0e',label='Prediction')
# ax.bar(bins2[:-1],hist2/16000,width=0.05,linewidth=0,facecolor=[255/256,191/256,135/256],alpha = 0.85, label='Truth')
         
# plt.xlabel(r'Flux ', fontsize=24)
# plt.ylabel(r'Probability Density', fontsize=24)

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
    
# N_av = 10
# N_crop1 = 897
# N_crop2 = 897
# N_epochs = 27.518
# training_skill = moving_average(np.load("./arrays/outfile3c2f_overfit.npz").items()[4][1],N_av)[N_crop1:-N_crop2][::10]
# testing_skill = moving_average(np.load("./arrays/outfile3c2f_overfit.npz").items()[5][1][1:],N_av)
# training_epochs = np.linspace(0 + (N_av - 1 + N_crop1)*(1/1120),N_epochs - (N_av - 1 + N_crop2)*(1/1120),len(training_skill))
# testing_epochs = np.linspace(0 + (N_av - 1)*(5/56),N_epochs - (N_av - 1)*(5/56),len(testing_skill))

# #1f77b4
# # 2ca02c
# # ff7f0e orange
# fig, ax = plt.subplots(figsize=(7, 5))
# plt.plot(training_epochs,training_skill,color='#1f77b4',alpha=1,label=r'Training data', linewidth = 0.7 )
# plt.plot(testing_epochs,testing_skill,color='#ff7f0e',alpha=1,label=r'Testing data', linewidth = 2 )
# plt.xlabel(r'Epochs', fontsize=22)
# plt.ylabel(r'Skill', fontsize=22)
# plt.xticks([0,4,8,12,16,20,24],fontsize=18)
# plt.yticks(fontsize=18)
# plt.legend(fontsize=18)
# # plt.show()
# plt.savefig("./figures3/training.png", dpi=300, bbox_inches = 'tight',transparent=True)





