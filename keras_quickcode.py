from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 100
epochs = 20
reload_data = False
data_path = './data256_4000/'
flux = "PSI2"  #flux to learn, probably PSI2 (unfiltered) or PSI2_f (filtered)
field = "PSI1"  #field to learn flux, probably PSI1 or PSI1_f (filtered)





#Some functions 
# def keras_skill(yp,yt):
#     skill = 1 - K.sqrt(K.mean(K.square(yp-yt)))/K.std(yt)
#     return skill

def keras_skill(yp,yt):
    skill = keras.losses.mean_squared_error(yt,yp)
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

def skill1(yp,yt):
    skill = 1 - np.sqrt(np.mean(np.square(yp-yt)))/np.std(yt)
    return skill


def time_series(prediction):
    flux_nontriv = y_test
    flux_nontriv_recon = linearly_regressed(prediction,flux_nontriv)
    fig, ax = plt.subplots(figsize=(15, 4))
    plt.plot(np.arange(200,600,0.25),flux_nontriv[0:1600],color='#ff7f0e',alpha=0.5, label = r'Truth')
    plt.plot(np.arange(200,600,0.25),flux_nontriv_recon[0:1600],color='#ff7f0e',alpha=1, label = r'Prediction, Skill: %.4f, $R^{2}$: %.2f' %(skill1(flux_nontriv_recon,flux_nontriv),R_squared(flux_nontriv_recon,flux_nontriv)))
    plt.xlabel(r'Days', fontsize=18)
    plt.ylabel(r'Eddy Thickness Flux', fontsize=18)
    plt.xticks([200,250,300,350,400,450,500,550,600],fontsize=14)
    plt.yticks([-1,-0,1],fontsize=14)
    plt.yticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14,frameon=False,loc=8)
    plt.margins(x=0)
    plt.show()
    
    
    
    
    
    
if reload_data == True:
    print('Loading data...')
    #this is all the available saved data, it is normalised by dividing by 2x the standard deviation of the test data
    in_data = np.load(data_path + 'fields/' + field + ".npz").items()
    x_train = in_data[0][1]
    x_test = in_data[1][1]; del in_data
    out_data = np.load(data_path + 'fluxes/' + flux + ".npz").items()
    y_train = np.reshape(out_data[0][1],(-1,1))
    y_test = np.reshape(out_data[1][1],(-1,1)); del out_data

x_train = x_train.reshape(112000,64,64,1)
x_test = x_test.reshape(16000,64,64,1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


model = Sequential()
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, kernel_size=(4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam())

for i in range(epochs):
    print('Epoch %g / %g' %(i+1, epochs))
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=1,
              verbose=1,
              validation_data=(x_test, y_test))
    prediction = model.predict(x_test)
    time_series(prediction)



