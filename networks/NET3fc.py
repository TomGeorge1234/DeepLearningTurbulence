import tensorflow as tf 


#parameters 87525
    
#the network 
def neuralnetwork(x):
    #builds our graph. x is an input tensor with shape (batch size, 64, 64) 
    #returns y, the output, at tensor of shape (batch size, 1)
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 64, 64, 1])
            
    with tf.name_scope('pool1'):
        y_pool1 = avg_pool_2x2(x_image)
        
    with tf.name_scope('reshape'):
        y_in = tf.reshape(y_pool1, [-1, 1024])

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([1024, 100])
        B_fc1 = bias_variable([100])
        y_fc1 = tf.nn.relu(tf.matmul(y_in, W_fc1) + B_fc1)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([100, 10])
        B_fc2 = bias_variable([10])
        y_fc2 = tf.nn.relu(tf.matmul(y_fc1, W_fc2) + B_fc2)
                
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        y_fc2_drop = tf.nn.dropout(y_fc2, keep_prob)

    with tf.name_scope('fc3'):
        W_fc3 = weight_variable([10, 1])
        B_fc3 = bias_variable([1])
        y_fc3 = tf.matmul(y_fc2_drop, W_fc3) + B_fc3
        
    return y_fc3, keep_prob 

#define variables and functions 
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype = tf.float32)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0, shape=shape, dtype = tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
  """avg_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')