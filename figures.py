import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.cm as cm
import scipy.signal
from eofs.standard import Eof
from time import time 

dpi = 300 
imidx = 460
arrays = np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileinfotesthalfPV1.npz').items()
ratioPV1PV2 = 9.9 #these values approximately normalise the input data 
ratioPV1PSI1 = 7.3e-4
ratioPV1PSI2 = 3e-3


yp_train = arrays[0][1]
yt_train = arrays[1][1]
yp_test = arrays[2][1]
yt_test = arrays[3][1]
cost = arrays[4][1]
cost_test = arrays[5][1]
accuracy = arrays[6][1]
accuracy_test = arrays[7][1]

def neuralnetwork(x):
    y1 = nonlin(np.dot(x,W1) + B1) #y1 has dimension [K,M1] (B is automatically reshaped) 
    y2 = nonlin2(np.dot(y1,W2) + B2) #l2 has dimension [K,1]
    return y1, y2

train_data_folders = ["data_all1","data_all2","data_all3","data_all4","data_all5"] #the folders (as a lit of strings, each within QGModelMATLAB) where the data is. If multiple folders, data will be concatenated.
test_data_folders = ["data_all_daily"]
input_channels = ["PV1"]
from get_data import calldata

#data import line
#1 day 
data = calldata("test")
input = data[0]
output = data[1]
outputEKE = output[0]
outputPV = output[1]



#PLOTS


#time series of EKE vs PV
data2 = output[1][0:1000]
t = np.arange(0,1000,1)

fig, ax1 = plt.subplots(figsize=(9, 4))

ax1.set_xlabel('Days')
ax1.set_ylabel('Potential Vorticity Flux')
ax1.plot(t, data2)
ax1.tick_params(axis='y')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("./figures/PVFtimeseries.png", dpi=dpi, bbox_inches='tight',transparent=True)







#PV mean + anomylous = PV 
Z_full = np.reshape(input[imidx],(32,32))
Z_mean = np.reshape(np.mean(input,0),(32,32))
Z_anom = np.reshape(inputrm[imidx],(32,32))

plt.figure(2)
plt.subplot(131)
plt.imshow(Z_full, interpolation='bilinear', cmap=cm.viridis,
                origin='lower', extent=[0, 1000, 0, 1000],
                vmax=abs(Z_full).max(), vmin=-abs(Z_full).max())
plt.yticks([0,500,1000])
plt.title('(a) Total', fontsize=8)
plt.axis('off')
    
    
plt.subplot(132)
plt.imshow(Z_mean, interpolation='bilinear', cmap=cm.viridis,
                origin='lower', extent=[0, 1000, 0, 1000],
                vmax=abs(Z_mean).max(), vmin=-abs(Z_mean).max())
plt.yticks([0,500,1000])
plt.title('(b) Mean', fontsize=8)
plt.axis('off')

plt.subplot(133)
plt.imshow(Z_anom, interpolation='bilinear', cmap=cm.viridis,
                origin='lower', extent=[0, 1000, 0, 1000],
                vmax=abs(Z_anom).max(), vmin=-abs(Z_anom).max())
plt.yticks([0,500,1000])
plt.title('(c) Anomaly', fontsize=8)
plt.axis('off')
plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=1.0)
    
plt.savefig("./figures/anom.png", dpi=dpi, bbox_inches='tight')








#PV plot sequences 







#correlation plots 
plt.figure(4,(5,7))   

inputcorrelation = np.array([])
for i in range(50):
    inputcorrelation = np.append(inputcorrelation,np.corrcoef(input[500].T,input[500+i].T)[1,0])
 
PVcorrelation = np.array([])
PVcorrelation = np.append(PVcorrelation,1)
for i in range(50-1):
    PVcorrelation = np.append(PVcorrelation,np.corrcoef(outputPV[500:-(i+1)].T,outputPV[500+(i+1):].T)[1,0])
    
plt.plot(PVcorrelation,label = 'PV flux')
plt.plot(inputcorrelation, label = 'Input')
plt.xlabel('Lag (days)')
plt.ylabel('Correlation')
plt.axhline(y=0, color='k', linewidth = 0.8 )
plt.legend()
plt.savefig("./figures/PV&incorrelation.png", dpi=dpi, bbox_inches = 'tight',transparent = True )







#cost plots 
plt.figure(5)
plt.plot(np.arange(len(cost)/1000)*1000,cost[0::1000],label = 'Training data')
plt.plot(np.arange(int(len(cost)/10000))*10000,cost_test[0::100],label = 'Test data')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig("./figures/loss.png", dpi=dpi, bbox_inches = 'tight')









#time series of predicted output

Ya = outputEKEn
Xa = inputrm
y1a, y2a = neuralnetwork(Xa)

plt.figure(7,(10,5))
k = int(0.4*len(y2a))
plt.plot(y2a[k:k+500],label='Prediction')
plt.plot(Ya[k:k+500],label='Truth')
plt.xlabel('Days')
plt.ylabel('EKE flux')
plt.legend()
plt.grid()

plt.savefig("./figures/predictedtimeseriesnewEKE.png", dpi=dpi, bbox_inches = 'tight')

#this is the fourier transform - just ignore 
plt.subplot(212)
ytf = scipy.signal.welch(np.reshape(Ya,(1,len(Ya))),nperseg=int(len(Ya)/5),scaling='spectrum')[1]
ypf = scipy.signal.welch(np.reshape(yc,(1,len(yc))),nperseg=int(len(yc)/5),scaling='spectrum')[1]
plt.loglog(ytf.T,label='Truth')
plt.loglog(ypf.T,label='Prediction')
plt.xlabel('Frequency (Days^-1)')
plt.ylabel('Power Spectrum(PV flux)')
plt.grid()
plt.legend()

plt.savefig("./figures/predictedtimeseries.png", dpi=dpi, bbox_inches = 'tight')







#scatter 
plt.figure(6)
plt.scatter(Y_test,y2_test,marker = '.', s = 1)
plt.plot([-1,1],[-1,1],":",color="green")
plt.axis([-1,1,-1,1],'equal')
plt.xlabel("Truth EKE"); plt.ylabel("Guess EKE")
plt.yticks([-1,-0.5,0,0.5,1])
plt.gca().set_aspect('equal','box')
plt.savefig("./figures/scatternewEKE.png", dpi=dpi, bbox_inches = 'tight')



#scatter 

plt.figure(6)
plt.scatter(Ya[100:1000],yc[100:1000],marker = '.', s = 1)
plt.plot([-1,1],[-1,1],":",color="green")
plt.axis([-1,1,-1,1],'equal')
plt.xlabel("Truth"); plt.ylabel("Guess")
plt.yticks([-1,-0.5,0,0.5,1])
plt.gca().set_aspect('equal','box')
plt.savefig("./figures/scatternew.png", dpi=dpi, bbox_inches = 'tight')







#single print of field
K = 500

PV1 = np.load("./data256_4000/" + "PV1_full_test" + ".npz").items()[0][1][K]
PV2 = np.load("./data256_4000/" + "PV2_full_test" + ".npz").items()[0][1][K]
PSI1 = np.load("./data256_4000/" + "PSI1_full_test" + ".npz").items()[0][1][K]
PSI2 = np.load("./data256_4000/" + "PSI2_full_test" + ".npz").items()[0][1][K]

from matplotlib import rc
plt.rc('text',usetex=True)
Z = PSI1
plt.figure(8)
plt.imshow(Z, extent=[0, 4000, 0, 4000])
plt.yticks([0,1000,2000,3000,4000])

plt.savefig("./figures3/PSI1full.png", dpi=300, bbox_inches = 'tight', transparent=True)


#noise accuracy
plt.figure(2,(10,4))
plt.scatter([0,0.2,0.5,1,3,4,5],[0.7757,0.7650,0.7641,0.7282,0.5893,0.4984,0.0582],marker='x')
plt.xlabel('|Noise|/|Signal|', fontsize = 20)
plt.ylabel('Maximum R', fontsize = 20)
plt.savefig("./figures/noisescatter.png", dpi=dpi, bbox_inches = 'tight', transparent=True)



#phase space plot 
input1 = calldata(train_data_folders , input_channels)[0]
input2 = calldata(datafolder , input_channels)[0]

np.random.shuffle(input1)
np.random.shuffle(input2)

N = 100

min_d = np.array([])

for i in range(N):
    G=int(np.exp(i/N)*len(input1))
    t0 = time()
    min_d = np.append(min_d,np.min(np.linalg.norm(input2[i]-input1[0:G],axis=1)))
    print(i, time()-t0)


plt.plot(min_d)




#scatters for results presentation
array = np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileoptimummodes200.npz').items()
yp_test, yt_test = array[2][1], array[3][1]
plt.scatter(yt_test,yp_test,s=0.1)
plt.plot([-1,1],[-1,1],":",color="green")
plt.axis([-1,1,-1,1],'equal')
plt.xlabel("Truth",fontsize = 20); plt.ylabel("Guess",fontsize = 20)
plt.gca().set_aspect('equal','box')
plt.yticks([-1,-0.5,0,0.5,1])
plt.xticks([-1,-0.5,0,0.5,1])
plt.savefig("./figures/scatter_best.png", dpi=dpi, bbox_inches = 'tight', transparent = True)



#time series for results presentation
array = np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileoptimummodes200.npz').items()
yp_test, yt_test = array[2][1], array[3][1]
ax = plt.figure(3,(1.5*6,6)).add_subplot(111)
series_start = 550
x_axis = np.arange(series_start,series_start+700,1)
plt.plot(x_axis,yp_test[series_start:series_start+700],label='Predicted')
plt.plot(x_axis,yt_test[series_start:series_start+700],label='Truth')
plt.ylim((-1,1))
plt.xlabel("Days",fontsize = 20); plt.ylabel("Output",fontsize = 20)
ax.set_aspect(170)
plt.legend()

plt.savefig("./figures/timeseries_best.png", dpi=dpi, bbox_inches = 'tight', transparent = True )


#R plots 
ax = plt.figure().add_subplot(111)
R1 = np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileoptimummodes200.npz').items()[7][1]
R1_ = np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileoptimummodes200.npz').items()[6][1][::100]
plt.plot(R1,color='blue')
plt.plot(R1_,alpha=0.2,color='blue')
plt.xlabel("Epoch",fontsize = 20); plt.ylabel("R",fontsize = 20)
plt.xticks([0,50,99,149,198])
ax.set_xticklabels(('0','5','10','15','20'))
plt.savefig("./figures/R_plot_best.png", dpi=dpi, bbox_inches = 'tight',transparent=True)


#info content plt
ax = plt.figure(11,(7,3)).add_subplot(111)
R1 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileinfotestsixteenthPV1.npz').items()[7][1])
R2 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileinfotesteighthPV1.npz').items()[7][1])
R3 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileinfotestquarterPV1.npz').items()[7][1])
R4 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileinfotesthalfPV1.npz').items()[7][1])
R5 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileinfotestPV1.npz').items()[7][1])
R6 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileinfotestPV1halfPV2.npz').items()[7][1])
R7 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileinfotestPV1PV2.npz').items()[7][1])
y = (0,R1,R2,R3,R4,R5,R6,R7)
x = (0,1/16,1/8,1/4,1/2,1,3/2,2)
plt.scatter(x,y,marker='x',color='blue')
x_tick = [0,1/4,1/2,3/4,1,5/4,3/2,7/4,2]
labels = ['No data','1/4 PV1','1/2 PV1','','PV1','','PV1 + 1/2 PV2','','PV1 + PV2 \n (Full information)']
plt.xticks(x_tick,labels,rotation = 0,fontsize=8)
plt.ylim(-0.05,1.05)
plt.xlabel('Information')
plt.ylabel('R')
plt.savefig("./figures/info_content.png", dpi=dpi, bbox_inches = 'tight',transparent=True)

#info content plt for EOF
ax = plt.figure(11,(7,3)).add_subplot(111)
R1 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileEOF1PV1.npz').items()[7][1])
R2 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileEOF5PV1.npz').items()[7][1])
R3 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileEOF10PV1.npz').items()[7][1])
R4 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileEOF20PV1.npz').items()[7][1])
R5 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileEOF50PV1.npz').items()[7][1])
R6 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileEOF100PV1.npz').items()[7][1])
R7 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileEOF200PV1.npz').items()[7][1])
R8 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileEOF500PV1.npz').items()[7][1])
R9 = np.max(np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileEOF1024PV1.npz').items()[7][1])
y = (0,R1,R2,R3,R4,R5,R6,R7,R8,R9)
x = (0,1,5,10,20,50,100,200,500,1024)
y_ = (0.1970,0.3587,0.4354,0.7376,0.7728,0.7787,0.7775,0.7757)
x_ = (1,4,10,50,140,200,500,1024)
plt.scatter(x,y,marker='x',color='blue',label = "2 layer FC ANN")
plt.scatter(x_,y_,marker='x',color='green', label = "3 layer CNN")
plt.ylim(-0.05,1.15)
plt.xlabel('EOF Modes kept')
plt.xticks([0,100,200,300,400,500,600,700,800,900,1000])
plt.ylabel('R')
plt.legend()
plt.savefig("./figures/EOFinfo.png", dpi=300, bbox_inches = 'tight',transparent=True)


#eof info cnn
plt.figure(4,(9,4))
x = [1,4,10,50,140,200,500,1024]
y = [0.1970,0.3587,0.4354,0.7376,0.7728,0.7787,0.7775,0.7757]
plt.plot(x,y,'x',color = 'blue')
plt.ylim([0,1])
plt.xlabel('EOF Modes kept')
plt.ylabel('R')
plt.set_aspect_ratio(3)
plt.legend()
plt.savefig("./figures/EOFinfoCNN.png", dpi=dpi, bbox_inches = 'tight',transparent=True)


#conv vs fc comparison 
ax = plt.figure().add_subplot(111)
R_fc = np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfilefceof64do50.npz').items()[7][1]
R_fc_train = np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfilefceof64do50.npz').items()[6][1][::100]
R_conv = np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileconv2eof64nodrop.npz').items()[7][1]
R_conv_train = np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileconv2eof64nodrop.npz').items()[6][1][::100]
plt.plot(R_fc,label='Fully connected',color='orange')
plt.plot(R_conv,label='Deep CNN',color='blue')
plt.plot(R_fc_train,alpha=0.2,color='orange')
plt.plot(R_conv_train,alpha=0.2,color='blue')
plt.xlabel("Epoch"); plt.ylabel("R")
plt.xticks([0,99,198,297])
ax.set_xticklabels(('0','10','20','30'))
plt.savefig("./figures/fcvsconvR.png", dpi=dpi, bbox_inches = 'tight',transparent=True)

ax = plt.figure().add_subplot(111)
L_fc = np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfilefceof64do50.npz').items()[5][1]
L_fc_train = np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfilefceof64do50.npz').items()[4][1][::100]
L_conv = np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileconv2eof64nodrop.npz').items()[5][1]
L_conv_train = np.load('/Users/tomgeorge/Documents/MyDocuments/Summer2018/CaltechSURF/Code/QGtf/arrays/outfileconv2eof64nodrop.npz').items()[4][1][::100]
plt.plot(L_fc,label='Fully connected',color='orange')
plt.plot(L_conv,label='Deep CNN',color='blue')
plt.plot(L_fc_train,alpha=0.2,color='orange')
plt.plot(L_conv_train,alpha=0.2,color='blue')
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.xticks([0,99,198,297])
plt.ylim(0.01,0.08)
ax.set_xticklabels(('0','10','20','30'))
plt.savefig("./figures/fcvsconvL.png", dpi=dpi, bbox_inches = 'tight',transparent=True)




#plot EOF basis vectors 
eof_basis = Eof(input)
basis = eof_basis.eofs() 


for i in [1023]:
    Z_eof = np.reshape(basis[i],(32,32))
    plt.figure()
    plt.subplot(111)
    plt.imshow(Z_eof, interpolation='bilinear', cmap=cm.viridis,
                    origin='lower', extent=[0, 1000, 0, 1000],
                    vmax=abs(Z_eof).max(), vmin=-abs(Z_eof).max())
    plt.yticks([0,500,1000])
    plt.axis('off')
    plt.savefig("./figures/eofbasis%g.png" %i, dpi=dpi, bbox_inches = 'tight', transparent=True)


variances = eof_basis.eigenvalues() 

fig = plt.figure(50, (5,10))
plot = plt.subplot(111)
plt.plot(variances[0:200],linewidth = 5)
plt.xlabel('EOF index', fontsize=30)
plt.ylabel('Variance', fontsize=30)
plt.xticks([0,50,100,150,200])
plot.tick_params(axis='both', which='major', labelsize=24)
plot.tick_params(axis='both', which='minor', labelsize=24)
plt.savefig("./figures/eofvariance.png", dpi=dpi, bbox_inches = 'tight', transparent=True)












#EOF vs CNN plots 

from matplotlib import rc 
rc('text', usetex=True)
rc('font', family='serif')

flux_true = np.load('./data256_4000/flux_test_6hour_cropped.npz').items()[0][1][0:1200]
flux_guess_eof = np.load('./arrays/eof_bestflux.npz').items()[0][1][0:1200]

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(200,500,0.25),flux_true,label='Truth')
plt.plot(np.arange(200,500,0.25),flux_guess_eof, label='Guess: PCA')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_ylim(-1,1)
plt.xlabel('Time (days)')
plt.ylabel('Flux')
ax.set_aspect(60)
plt.savefig("./figures3/results_trueeof.pdf",dpi=300,bbox_inches = 'tight', transparent=True)




