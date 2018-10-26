import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from eofs.standard import Eof
from time import time
import sys 
sys.path.append('./networks/')
from NETthreelayerconv_2fc import neuralnetwork #NET------ file contains the neural net architecture, as a function 




#DEFINE VARIABLES
eps = 1e-3 #learning rate
K = 100 #learning batch size
savekey = '-' #explain network and make the save recognisable
reload_data = True #if data is already loaded, save time by setting False
testfreq = 100
drop_prob = 0.7
data_path = './data256_4000/'

flux = "PSI2_f"
field = "PSI1_f"



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
    
#    print("Calculating EOFs")
#    eof_basis = Eof(trainimages_) #.eofs() give the eof basic vectors, .eigenvalues() give the variances
#    print("Done")

#manipulate data



##1) No manipulation
trainimages = trainimages_
testimages = testimages_
trainoutput = trainoutput_
testoutput = testoutput_
    
# testoutput_f = np.reshape(np.load(data_path + "flux_psi2_f_test" + ".npz").items()[0][1],(-1,1))
# testoutput_full = np.reshape(np.load(data_path + "flux_test" + ".npz").items()[0][1],(-1,1))
# plt.plot(np.arange(len(testoutput))[0:500],testoutput[0:500],label='Flux of psi2')
# plt.plot(np.arange(len(testoutput_f))[0:500],testoutput_f[0:500],label='Flux of psi2 filtered')
# plt.plot(np.arange(len(testoutput_full))[0:500],testoutput_full[0:500],label='Full flux of q1')
# plt.legend()
# plt.show()





##2) Take only the first M0 EOF modes
#n=50
#trainimages_eof = eof_basis.eofs(neofs=n)
#trainimages = np.dot(np.dot(trainimages_,trainimages_eof.T),eof_basis.eofs(neofs=n))
#testimages = np.dot(np.dot(testimages_,trainimages_eof.T),eof_basis.eofs(neofs=n))
#plt.plot(np.arange(200),eof_basis.eigenvalues()[0:200])
#plt.xlabel("EOF index, i")
#plt.ylabel("EOF Variance")
#plt.savefig("./figures3/variances.png", dpi=300, bbox_inches='tight',transparent=True)


##3) Train on only the first s images in your total training data
# s = 3300
# trainimages = trainimages_[0:s,:,:]
# trainoutput = trainoutput_[0:s]
# testimages = testimages_
# testoutput = testoutput_


#4) Add noise
#noise_ratio = 5
#noise = np.random.normal(0, scale = noise_ratio*np.std(trainimages_), size = (len(trainimages_),64,64))
#trainimages = trainimages_+noise
#testimages = testimages_
#trainoutput = trainoutput_
#testoutput = testoutput_



#6) Add in some of PSI1
#n = 1024
#PSI1_eof = eof_basis.eofs(neofs=n)[:,1024:2048]
#PSI1trainimages = np.dot(np.dot(trainimages_[:,1024:2048],PSI1_eof.T),PSI1_eof)
#PSI1testimages = np.dot(np.dot(testimages_[:,1024:2048],PSI1_eof.T),PSI1_eof)
#
#trainimages = np.concatenate((trainimages_[:,0:1024],PSI1trainimages),1)
#testimages = np.concatenate((testimages_[:,0:1024],PSI1testimages),1)
#
#
#
#bases = eof_basis.eofs(neofs=2048)
#plt.imshow(np.reshape(bases[1],(64,32)))
#plt.axis('off')
#plt.savefig("./figures/eofconcat1.png", dpi=300, bbox_inches='tight',transparent=True)


#7) Add in a second field in the fourth dimension

#trainimages_ = np.empty((len(PSI1_train),64,64,2))
#trainimages_[:,:,:,0] = PSI1_train; del PSI1_train
#trainimages_[:,:,:,1] = PV1_train; del PV1_train
#trainoutput = flux_train; del flux_train
#testimages_ = np.empty((len(PSI1_test),64,64,2))
#testimages_[:,:,:,0] = PSI1_test; del PSI1_test
#testimages_[:,:,:,1] = PV1_test; del PV1_test
#testoutput = flux_test; del flux_test






#SOME FUNCTIONS AND ARRAY INITIALISATION
#functions
def next_batch(k):
    idx = np.random.choice(np.arange(len(trainoutput)),k,replace=False)
    return trainimages[idx], trainoutput[idx]

def accuracy(yp,yt):
    return stats.mstats.linregress(yp,yt)[2]

def skill(yp,yt):
    return 1 - np.sqrt(((np.dot((yt-yp).T,(yt-yp)))/(len(yt))))/np.std(yt)



#initialie arrays
cost_array = np.array([])
cost_test_array = np.array([])
accuracy_array = np.array([])
accuracy_test_array = np.array([])





#MAIN ROUTINE (forward propagation, back pragation and print/save analysis data)
def main(_):
    x = tf.placeholder(tf.float32, [None, 64, 64])
    yt = tf.placeholder(tf.float32, [None, 1]) #truth y
    yp, keep_prob = neuralnetwork(x) #predicted y     keep_prob = tf.placeholder(tf.as_dtype(int),shape=())

    with tf.name_scope('loss'):
        cost = tf.losses.mean_squared_error(yt,yp)

    with tf.name_scope('optimizer'):
        train_step = tf.train.AdamOptimizer(eps).minimize(cost)

    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()
        print("Initialising Neural Network...")
        sess.run(init_op)
        print("Done.")
        saver = tf.train.Saver()

        i = 0
        max_idx = 0
        t0 = time()
        while (i - max_idx*testfreq)*K/len(trainimages) < 3*(115200/len(trainimages)): #terminates training when the accuracy hasn't increased over this many  epochs
        # while (i/testfreq - max_idx) < 5: #terminates training when it hasnt improved in the last 5 testing iterations 

            is_training = True
            batch = next_batch(K)
            train_step.run(feed_dict={x: batch[0], yt: batch[1], keep_prob: drop_prob})

            yp_ = sess.run(yp,feed_dict={x: batch[0], yt: batch[1], keep_prob: drop_prob}) #explicitly calculate yp as a numpy array

            #evaluate loss and accuracy
            loss = cost.eval(feed_dict={x: batch[0], yt: batch[1], keep_prob: drop_prob})
            global cost_array
            cost_array = np.append(cost_array,loss)
            accuracy_ = accuracy(yp_,batch[1])
            global accuracy_array
            accuracy_array = np.append(accuracy_array,accuracy_)

            #run a test
            if i % testfreq == 0:
                is_training = False
                x_test = testimages
                yt_test = testoutput
                yp_test = sess.run(yp,feed_dict={x: x_test, yt: yt_test, keep_prob: 1})


                loss_test = cost.eval(feed_dict={x: testimages, yt: yt_test, keep_prob: 1})
                global cost_test_array
                cost_test_array = np.append(cost_test_array,loss_test)
                accuracy_test = accuracy(yp_test,yt_test)
                global accuracy_test_array
                accuracy_test_array = np.append(accuracy_test_array,accuracy_test)
                skill_test = skill(yp_test,yt_test)
                #plot relevant figures
                plt.figure(1,(6.5,6))

                #scatter plot
                plt.subplot(221)
                plt.scatter(yt_test,yp_test,s=0.1)
                plt.plot([-1,1],[-1,1],":",color="green")
                plt.axis([-1,1,-1,1],'equal')
                plt.yticks([-1,-0.5,0,0.5,1])
                plt.xlabel("Truth"); plt.ylabel("Guess")

                #time series
                plt.subplot(222)
                series_start = int(i/testfreq)*2
                x_axis = np.arange(series_start,series_start+500,1)
                plt.plot(x_axis,yp_test[series_start:series_start+500],label='Predicted')
                plt.plot(x_axis,yt_test[series_start:series_start+500],label='Truth')
                plt.ylim((-1,1))
                plt.xlabel("Days"); plt.ylabel("Output")
                plt.legend()

                #loss function
                plt.subplot(223)
                if i<10*len(trainimages)/K:
                    plt.plot(np.arange(i)*K/len(trainimages),cost_array[:i],label="Training")
                    plt.plot(np.arange(int(i/testfreq)+1)*testfreq*K/len(trainimages),cost_test_array[:int(i/testfreq)+1],label="Testing")
                else:
                    plt.plot(np.arange(int(i-10*len(trainimages)/K),i)*K/len(trainimages),cost_array[int(i-10*len(trainimages)/K):i],label="Training")
                    plt.plot(np.arange(int((i-10*len(trainimages)/K)/testfreq),int(i/testfreq)+1)*testfreq*K/len(trainimages),cost_test_array[int((i-10*len(trainimages)/K)/testfreq):int(i/testfreq)+1],label="Testing")
                plt.xlabel("Epochs"); plt.ylabel("Loss")

                #R plot
                plt.subplot(224)
                if i<10*len(trainimages)/K:
                    plt.plot(np.arange(i)*K/len(trainimages),accuracy_array[:i],label="Training")
                    plt.plot(np.arange(int(i/testfreq)+1)*testfreq*K/len(trainimages),accuracy_test_array[:int(i/testfreq)+1],label="Testing")
                else:
                    plt.plot(np.arange(int(i-10*len(trainimages)/K),i)*K/len(trainimages),accuracy_array[int(i-10*len(trainimages)/K):i],label="Training")
                    plt.plot(np.arange(int((i-10*len(trainimages)/K)/testfreq),int(i/testfreq)+1)*testfreq*K/len(trainimages),accuracy_test_array[int((i-10*len(trainimages)/K)/testfreq):int(i/testfreq)+1],label="Testing")
                plt.xlabel("Epochs"); plt.ylabel("R")

                plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=1.0)
                plt.show()

                max_idx = np.argmax(accuracy_test_array)
                print('R: %.4f, Epoch: %.1f, Cost: %g, Time = %.2f mins, Skill = %.4f' % (accuracy_test, i*K/len(trainoutput), loss_test, (time()-t0)/60, skill_test))
                yp_ = sess.run(yp,feed_dict={x: batch[0], yt: batch[1], keep_prob: drop_prob}) #explicitly calculate yp as a numpy array

            i += 1
            if i % int(115200/K) == 0: #saves every epoch in case it is needed to quit early 
                yt_train = trainoutput
                yp_train = sess.run(yp,feed_dict={x: trainimages, yt: yt_train, keep_prob: 1})
                np.savez('./arrays/outfile' + savekey, yp_train, yt_train, yp_test, yt_test, cost_array, cost_test_array, accuracy_array, accuracy_test_array)
                print('Saved')
            
        text = 'Max R: %.4f, Training rate: %E, Epoch: %g, Time: %.2f mins, SaveKey: %s, keep_prob: %.2f, Batch size: %g \n' %(np.max(accuracy_test_array), eps, int(i*K/len(trainoutput)), (time()-t0)/60, savekey, drop_prob, K)
        yt_train = trainoutput
        yp_train = sess.run(yp,feed_dict={x: trainimages, yt: yt_train, keep_prob: 1})
        text = 'Max R: %.4f, Training rate: %E, Epoch: %g, Time: %.2f mins, SaveKey: %s, keep_prob: %.2f, Batch size: %g \n' %(np.max(accuracy_test_array), eps, int(i*K/len(trainoutput)), (time()-t0)/60, savekey, drop_prob, K)
        print(text)

        model_save_path = "./models" + "/model" + savekey + ".ckpt"
        save_path = saver.save(sess, model_save_path)
        print("Model saved in path: %s" % save_path)


    #write results into a file
    with open('./results/QGtfResults.txt', 'a+') as file:
        file.write(text)

    #saves important arrays to file so they can be accessed and analysed by other programs
    np.savez('./arrays/outfile' + savekey, yp_train, yt_train, yp_test, yt_test, cost_array, cost_test_array, accuracy_array, accuracy_test_array)






#CALL MAIN ROUTINE
if __name__ == "__main__":
    main(0)
