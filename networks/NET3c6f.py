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
    
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([4, 4, 1, 8])
        B_conv1 = bias_variable([8])
        y_conv1 = tf.nn.relu(conv2d(y_pool1, W_conv1) + B_conv1)
    
    with tf.name_scope('pool2'):
        y_pool2 = max_pool_2x2(y_conv1)
        
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([4, 4, 8, 16])
        B_conv2 = bias_variable([16])
        y_conv2 = tf.nn.relu(conv2d(y_pool2, W_conv2) + B_conv2)  
        
    with tf.name_scope('pool3'):
        y_pool3 = max_pool_2x2(y_conv2)
    
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([4, 4, 16, 32])
        B_conv3 = bias_variable([32])
        y_conv3 = tf.nn.relu(conv2d(y_pool3, W_conv3) + B_conv3)  
        
    with tf.name_scope('pool4'):
        y_pool4 = max_pool_2x2(y_conv3)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([4 * 4 * 32, 150])
        B_fc1 = bias_variable([150])
        y_conv3_flat = tf.reshape(y_pool4, [-1, 4*4*32])
        y_fc1 = tf.nn.relu(tf.matmul(y_conv3_flat, W_fc1) + B_fc1)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([150, 50])
        B_fc2 = bias_variable([50])
        y_fc2 = tf.nn.relu(tf.matmul(y_fc1, W_fc2) + B_fc2)

    with tf.name_scope('fc3'):
        W_fc3 = weight_variable([50, 20])
        B_fc3 = bias_variable([20])
        y_fc3 = tf.nn.relu(tf.matmul(y_fc2, W_fc3) + B_fc3)

    with tf.name_scope('fc4'):
        W_fc4 = weight_variable([20, 20])
        B_fc4 = bias_variable([20])
        y_fc4 = tf.nn.relu(tf.matmul(y_fc3, W_fc4) + B_fc4)

    with tf.name_scope('fc5'):
        W_fc5 = weight_variable([20, 20])
        B_fc5 = bias_variable([20])
        y_fc5 = tf.nn.relu(tf.matmul(y_fc4, W_fc5) + B_fc5)
        
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        y_fc5_drop = tf.nn.dropout(y_fc5, keep_prob)

    with tf.name_scope('fc6'):
        W_fc6 = weight_variable([20, 1])
        B_fc6 = bias_variable([1])
        y_fc6 = tf.matmul(y_fc5_drop, W_fc6) + B_fc6
        
    return y_fc6, keep_prob

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