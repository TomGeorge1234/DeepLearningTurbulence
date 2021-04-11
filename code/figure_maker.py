import numpy as np 
import matplotlib as m 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from eofs.standard import Eof
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean

cmp = 'Spectral'
# cmap = cmocean.cm.zeta
# cmap = zeta

from matplotlib import rc
plt.rc('text',usetex=True)
        


#RESULTS GRAPHS FOR CNN VS EOFS
#
#flux_true = np.load('./data256_4000/flux_test_6hour_cropped.npz').items()[0][1][0:1200]
##flux_guess_eof = np.load('./arrays/eof_bestflux.npz').items()[0][1][0:1200]
##flux_guess_cnn = np.load('./arrays/outfilemax_attempt.npz').items()[2][1][0:1200]
##
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.plot(np.arange(200,500,0.25),flux_true,label=r"Truth")
#plt.plot(np.arange(200,500,0.25),flux_guess_cnn, label =r"Prediction: CNN")
##plt.plot(np.arange(200,500,0.25),flux_guess_eof,'C2', label =r"Prediction: PCA", alpha=0.4)
#plt.legend(fontsize="xx-small")
#ax.set_ylim(-1,1)
#ax.set_xlim(200,500)
#plt.xlabel(r'Time (days)')
#plt.ylabel(r'$q_{1}$ Flux')
#ax.set_aspect(50)
#plt.show()
##plt.savefig("./figures3/results_trueeofcnn.pdf", bbox_inches = 'tight', dpi=400, transparent=True)


# cmap = plt.get_cmap('Spectral')
cmap = 'viridis'
foldername = 'norm'

def save_image(data, filename, weight = False):
    sizes = np.shape(data)     
    fig = plt.figure(figsize=(1,1))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if weight == True:
        im = ax.imshow(norm(data), cmap=plt.get_cmap('plasma'), vmin=-1, vmax=1)
    else:
        im = ax.imshow(norm(data), cmap=cmap, vmin=-1, vmax=1)
    plt.savefig('./figures3/snapshots/' + foldername + '/' + filename + '.pdf', dpi = 300) 
    plt.close()

a = np.array([[-1,1]])
plt.figure(figsize=(9,2))
img = plt.imshow(a, cmap=cmap)
plt.gca().set_visible(False)
cax = plt.axes([0.12, 0.45, 0.8, 0.4])
cbar = plt.colorbar(cax=cax, orientation='horizontal', ticks=[-1,0,1])
cbar.ax.tick_params(labelsize=60) 
plt.savefig('./figures3/snapshots/' + foldername + '/' + 'main_colourbar' + '.pdf', dpi = 300, transparent=True)

a = np.array([[-1,1]])
plt.figure(figsize=(4,2))
img = plt.imshow(a, cmap='plasma')
plt.gca().set_visible(False)
cax = plt.axes([0.12, 0.45, 0.8, 0.4])
cbar = plt.colorbar(cax=cax, orientation='horizontal', ticks=[-1,0,1])
cbar.ax.tick_params(labelsize=53) 
plt.savefig('./figures3/snapshots/' + foldername + '/' + 'weight_colourbar' + '.pdf', dpi = 300, transparent=True)


#PLOTS A SINGLE SNAPSHOT NO AXES

def norm(x):
    if np.abs(np.min(x)) > np.abs(np.max(x)):
        return(x/np.abs(np.min(x)))
    else:
        return(x/np.max(x))

im64x64 = np.load('./arrays/imagesoutput_new.npz').items()[0][1][0]
im32x32 = np.load('./arrays/imagesoutput_new.npz').items()[1][1][0,:,:,0]

im32x32_conv1_1 = np.load('./arrays/imagesoutput_new.npz').items()[2][1][0,:,:,0]
im32x32_conv1_2 = np.load('./arrays/imagesoutput_new.npz').items()[2][1][0,:,:,1]
im32x32_conv1_3 = np.load('./arrays/imagesoutput_new.npz').items()[2][1][0,:,:,2]
im32x32_conv1_4 = np.load('./arrays/imagesoutput_new.npz').items()[2][1][0,:,:,3]
im32x32_conv1_5 = np.load('./arrays/imagesoutput_new.npz').items()[2][1][0,:,:,4]
im32x32_conv1_6 = np.load('./arrays/imagesoutput_new.npz').items()[2][1][0,:,:,5]
im32x32_conv1_7 = np.load('./arrays/imagesoutput_new.npz').items()[2][1][0,:,:,6]
im32x32_conv1_8 = np.load('./arrays/imagesoutput_new.npz').items()[2][1][0,:,:,7]

im_conv2_1 = np.load('./arrays/imagesoutput_new.npz').items()[4][1][0,:,:,0]
im_conv2_2 = np.load('./arrays/imagesoutput_new.npz').items()[4][1][0,:,:,2]
im_conv2_3 = np.load('./arrays/imagesoutput_new.npz').items()[4][1][0,:,:,4]
im_conv2_4 = np.load('./arrays/imagesoutput_new.npz').items()[4][1][0,:,:,6]
im_conv2_5 = np.load('./arrays/imagesoutput_new.npz').items()[4][1][0,:,:,8]
im_conv2_6 = np.load('./arrays/imagesoutput_new.npz').items()[4][1][0,:,:,10]
im_conv2_7 = np.load('./arrays/imagesoutput_new.npz').items()[4][1][0,:,:,12]
im_conv2_8 = np.load('./arrays/imagesoutput_new.npz').items()[4][1][0,:,:,14]

im_conv3_1 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,0]
im_conv3_2 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,1]
im_conv3_3 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,2]
im_conv3_4 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,3]
im_conv3_5 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,4]
im_conv3_6 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,5]
im_conv3_7 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,6]
im_conv3_8 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,7]
im_conv3_9 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,8]
im_conv3_10 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,9]
im_conv3_11 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,10]
im_conv3_12 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,11]
im_conv3_13 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,12]
im_conv3_14 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,13]
im_conv3_15 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,14]
im_conv3_16 = np.load('./arrays/imagesoutput_new.npz').items()[6][1][0,:,:,15]

weight_matrix_1 = np.load('./arrays/imagesoutput_new.npz').items()[8][1][:,:,0,0]
weight_matrix_2 = np.load('./arrays/imagesoutput_new.npz').items()[8][1][:,:,0,1]
weight_matrix_3 = np.load('./arrays/imagesoutput_new.npz').items()[8][1][:,:,0,2]
weight_matrix_4 = np.load('./arrays/imagesoutput_new.npz').items()[8][1][:,:,0,3]
weight_matrix_5 = np.load('./arrays/imagesoutput_new.npz').items()[8][1][:,:,0,4]
weight_matrix_6 = np.load('./arrays/imagesoutput_new.npz').items()[8][1][:,:,0,5]
weight_matrix_7 = np.load('./arrays/imagesoutput_new.npz').items()[8][1][:,:,0,6]
weight_matrix_8 = np.load('./arrays/imagesoutput_new.npz').items()[8][1][:,:,0,7]


save_image(im64x64,'im64x64_test')
# save_image(im32x32,'im32x32')

save_image(im32x32_conv1_1,'im32x32_conv1_1')
save_image(im32x32_conv1_4,'im32x32_conv1_4')
save_image(im32x32_conv1_7,'im32x32_conv1_7')
save_image(im32x32_conv1_8,'im32x32_conv1_8')

save_image(im_conv2_1,'im_conv2_1')
save_image(im_conv2_2,'im_conv2_2')
save_image(im_conv2_3,'im_conv2_3')
save_image(im_conv2_4,'im_conv2_4')
save_image(im_conv2_5,'im_conv2_5')
save_image(im_conv2_6,'im_conv2_6')
save_image(im_conv2_7,'im_conv2_7')
save_image(im_conv2_8,'im_conv2_8')

save_image(im_conv3_1,'im_conv3_1')
save_image(im_conv3_2,'im_conv3_2')
save_image(im_conv3_3,'im_conv3_3')
save_image(im_conv3_4,'im_conv3_4')
save_image(im_conv3_5,'im_conv3_5')
save_image(im_conv3_6,'im_conv3_6')
save_image(im_conv3_7,'im_conv3_7')
save_image(im_conv3_8,'im_conv3_8')
save_image(im_conv3_9,'im_conv3_9')
save_image(im_conv3_10,'im_conv3_10')
save_image(im_conv3_11,'im_conv3_11')
save_image(im_conv3_12,'im_conv3_12')
save_image(im_conv3_13,'im_conv3_13')
save_image(im_conv3_14,'im_conv3_14')
save_image(im_conv3_15,'im_conv3_15')
save_image(im_conv3_16,'im_conv3_16')

save_image(weight_matrix_1,'weight_matrix_1',weight=True)
save_image(weight_matrix_4,'weight_matrix_4',weight=True)
save_image(weight_matrix_7,'weight_matrix_7',weight=True)
save_image(weight_matrix_8,'weight_matrix_8',weight=True)

save_image(norm(im32x32_conv1_1)-norm(im32x32_conv1_4),'filterdifference1')
save_image(norm(im32x32_conv1_7)-norm(im32x32_conv1_8),'filterdifference2')


# SINGLE PRINT OF FIELD WITH AXES
# a1 = np.load('./data256_4000/PSI1_train.npz').items()[0][1][100]
# a2 = np.load('./data256_4000/PSI1_train.npz').items()[0][1][101]
# a3 = np.load('./data256_4000/PSI1_train.npz').items()[0][1][102]
# a4 = np.load('./data256_4000/PSI1_train.npz').items()[0][1][103]
# a5 = np.load('./data256_4000/PSI1_train.npz').items()[0][1][104]

# plt.figure()
# plt.imshow(a5, extent=[0, 1000, 0, 1000])
# plt.yticks([0,200,400,600,800,1000])

# plt.savefig("./figures3/single_axes5.png", dpi=300, bbox_inches = 'tight', transparent=True)


#noise accuracy
# plt.figure(2,(10,4))
# plt.scatter([0,0.2,0.5,1,3,4,5],[0.7757,0.7650,0.7641,0.7282,0.5893,0.4984,0.0582],marker='x')
# plt.xlabel('|Noise|/|Signal|', fontsize = 20)
# plt.ylabel('Maximum R', fontsize = 20)
# plt.savefig("./figures/noisescatter.png", dpi=dpi, bbox_inches = 'tight', transparent=True)




#EOFS OF TRAINING DATA 

#PSI1 = np.load('./data256_4000/PSI1_train.npz').items()[0][1]
#solver = Eof(PSI1)
#eigenvecs = solver.eofs()
#np.savez('./arrays/PSI1eofs_', eigenvecs)
#
#coeffs = np.dot(np.reshape(PSI1[400],(1,4096)),np.reshape(eigenvecs,(-1,4096)).T)
#
#recon = np.dot(coeffs,np.reshape(eigenvecs,(-1,4096)))  
#
#def save_image(data, filename):
#  sizes = np.shape(data)     
#  fig = plt.figure(figsize=(1,1))
#  ax = plt.Axes(fig, [0., 0., 1., 1.])
#  ax.set_axis_off()
#  fig.add_axes(ax)
#  ax.imshow(data)
#  plt.savefig('./figures3/' + filename + '.png', dpi = 300) 
#  plt.close()
#
#save_image(eigenvecs[0],'eigenvec0')
#save_image(eigenvecs[1],'eigenvec1')
#save_image(eigenvecs[2],'eigenvec2')
#save_image(eigenvecs[100],'eigenvec100')
#save_image(PSI1[400],'PSI1[400]')







##PHASE SPACE PLOT 
#training_max = []
#testing_max = []
#for j in ['100','500','1000','3000','5000','10000','20000','30000','50000','100000']:
#    train_accuracy = np.load('./arrays/outfilesamples:'+j+'.npz').items()[6][1]
#    train_max = np.max(np.mean(train_accuracy[np.remainder(len(train_accuracy),200):].reshape(-1, 200), axis=1))
#    test_accuracy = np.load('./arrays/outfilesamples:'+j+'.npz').items()[7][1]
#    test_max = np.max(test_accuracy)
#    training_max.append(train_max)
#    testing_max.append(test_max)
#    
#x = [100,500,1000,3000,5000,10000,20000,30000,50000,100000]
#plt.plot(x,training_max,'x',label='Training')
#plt.plot(x,training_max,'k-',linewidth=0.6,label='__nolegend__',alpha=0.5)
#plt.plot(x,testing_max,'x',label='Testing')
#plt.plot(x,testing_max,'k-',linewidth=0.6,label='__nolegend__',alpha=0.5)
#plt.xlabel('Number of training images', size=15)
#plt.ylabel('Accuracy', size=15)
#plt.legend()
#plt.axes().set_aspect(60000)
##plt.show()
#plt.savefig("./figures3/phasespace.png", dpi=300, bbox_inches = 'tight', transparent=True)
#
#

#PHASE SPACE PLOT 
# training_max = []
# testing_max = []
# for j in ['100','500','1000','3000','5000','10000','20000','30000','50000','100000']:
#     train_accuracy = np.load('./arrays/outfilesamples:'+j+'.npz').items()[6][1]
#     train_max = np.max(np.mean(train_accuracy[np.remainder(len(train_accuracy),200):].reshape(-1, 200), axis=1))
#     test_accuracy = np.load('./arrays/outfilesamples:'+j+'.npz').items()[7][1]
#     test_max = np.max(test_accuracy)
#     training_max.append(train_max)
#     testing_max.append(test_max)
    
# x = [100,500,1000,3000,5000,10000,20000,30000,50000,100000]
# plt.figure(1,(9,3))
# plt.plot(x,training_max,'.',label='Training')
# plt.plot(x,training_max,'b-',linewidth=0.6,label='__nolegend__',alpha=0.5)
# plt.plot(x,testing_max,'.',label='Testing')
# plt.plot(x,testing_max,'r-',linewidth=0.6,label='__nolegend__',alpha=0.5)
# plt.xscale('log')
# plt.xlabel('Number of training images', size=15)
# plt.ylabel('Accuracy', size=15)
# plt.legend()
# #plt.show()
# plt.savefig("./figures3/phasespace.png", dpi=300, bbox_inches = 'tight', transparent=True)
# #
# #