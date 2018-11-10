#QGmain.py Author: Tom George
#To use, set user input variables, check global variable, 


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from time import time
import sys 
sys.path.append('./networks/')



#USER INPUT VARIABLES
from NET4c3f_heavy import neuralnetwork #NET------ file contains the neural net architecture
savekey = '4c3f_heavy'; print('WARNING! SAVEKEY = %s, IS THIS CORRECT?' %savekey) #unique key the results are saved under with warning to prevent accidental overwrite
flux = "PSI2"  #flux to learn, probably PSI2 (unfiltered) or PSI2_f (filtered)
field = "PSI1"  #field to learn flux, probably PSI1 or PSI1_f (filtered)



#SOME GLOBAL VARIABLES
eps = 1e-3 #adamoptimizer learning rate
K = 100 #learning batch size
reload_data = True #if data is already loaded, save time by setting False
testfreq = 100 #how often testing is done 
drop_prob = 0.7 #this is the keep-probability
data_path = './data256_4000/'




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
    

#manipulate data
##1) No manipulation
trainimages = trainimages_
testimages = testimages_
trainoutput = trainoutput_
testoutput = testoutput_
    

##2) Train on only the first s images in your total training data
# s = 33000
# trainimages = trainimages_[0:s,:,:]
# trainoutput = trainoutput_[0:s]
# testimages = testimages_
# testoutput = testoutput_








#SOME FUNCTIONS AND ARRAY INITIALISATION
#functions
def next_batch(k):
    idx = np.random.choice(np.arange(len(trainoutput)),k,replace=False)
    return trainimages[idx], trainoutput[idx]

def accuracy(yp,yt):
    return stats.mstats.linregress(yp,yt)[2]

def skill(yp,yt):
    return 1 - np.sqrt(((np.dot((yt-yp).T,(yt-yp)))/(len(yt))))/np.std(yt)



#initialise arrays
cost_array = np.array([])
cost_test_array = np.array([])
skill_array = np.array([])
skill_test_array = np.array([])
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
            skill_ = skill(yp_,batch[1])
            global skill_array
            skill_array = np.append(skill_array,skill_)


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
                global skill_test_array
                skill_test_array = np.append(skill_test_array,skill_test)
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

                #skill function
                plt.subplot(223)
                if i<0.5*len(trainimages)/K:
                    plt.plot(np.arange(i)*K/len(trainimages),skill_array[:i],label="Training")
                    plt.plot(np.arange(int(i/testfreq)+1)*testfreq*K/len(trainimages),skill_test_array[:int(i/testfreq)+1],label="Testing")
                elif i<3*len(trainimages)/K:
                    plt.plot(np.arange(int(i-0.5*len(trainimages)/K),i)*K/len(trainimages),skill_array[int(i-0.5*len(trainimages)/K):i],label="Training")
                    plt.plot(np.arange(int((i-0.5*len(trainimages)/K)/testfreq),int(i/testfreq)+1)*testfreq*K/len(trainimages),skill_test_array[int((i-0.5*len(trainimages)/K)/testfreq):int(i/testfreq)+1],label="Testing")
                else:
                    plt.plot(np.arange(int(i-3*len(trainimages)/K),i)*K/len(trainimages),skill_array[int(i-3*len(trainimages)/K):i],label="Training")
                    plt.plot(np.arange(int((i-3*len(trainimages)/K)/testfreq),int(i/testfreq)+1)*testfreq*K/len(trainimages),skill_test_array[int((i-3*len(trainimages)/K)/testfreq):int(i/testfreq)+1],label="Testing")
                plt.xlabel("Epochs"); plt.ylabel("Skill")

                #R plot
                plt.subplot(224)
                if i<3*len(trainimages)/K:
                    plt.plot(np.arange(i)*K/len(trainimages),accuracy_array[:i],label="Training")
                    plt.plot(np.arange(int(i/testfreq)+1)*testfreq*K/len(trainimages),accuracy_test_array[:int(i/testfreq)+1],label="Testing")
                else:
                    plt.plot(np.arange(int(i-3*len(trainimages)/K),i)*K/len(trainimages),accuracy_array[int(i-3*len(trainimages)/K):i],label="Training")
                    plt.plot(np.arange(int((i-3*len(trainimages)/K)/testfreq),int(i/testfreq)+1)*testfreq*K/len(trainimages),accuracy_test_array[int((i-3*len(trainimages)/K)/testfreq):int(i/testfreq)+1],label="Testing")
                plt.xlabel("Epochs"); plt.ylabel("R")

                plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=1.0)
                plt.show()

                max_idx = np.argmax(skill_test_array)
                print('Skill = %.4f, R: %.4f, Epoch: %.1f, Time = %.2f mins' % (skill_test, accuracy_test, i*K/len(trainoutput), (time()-t0)/60))
                yp_ = sess.run(yp,feed_dict={x: batch[0], yt: batch[1], keep_prob: drop_prob}) #explicitly calculate yp as a numpy array

            i += 1
            if i % int(115200/K) == 0: #saves every epoch in case it is needed to quit early 
                yt_train = trainoutput
                yp_train = sess.run(yp,feed_dict={x: trainimages, yt: yt_train, keep_prob: 1})
                np.savez('./arrays/outfile' + savekey, yp_train, yt_train, yp_test, yt_test, skill_array, skill_test_array, accuracy_array, accuracy_test_array)
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
    np.savez('./arrays/outfile' + savekey, yp_train, yt_train, yp_test, yt_test, skill_array, skill_test_array, accuracy_array, accuracy_test_array)






#CALL MAIN ROUTINE
if __name__ == "__main__":
    main(0)
