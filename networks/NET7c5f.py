import tensorflow as tf 



    
#the network 
def neuralnetwork(x):
    #builds our graph. x is an input tensor with shape (batch size, 64, 64) 
    #returns y, the output, at tensor of shape (batch size, 1)
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 64, 64, 1])
        
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([8, 8, 1, 8])
        B_conv1 = bias_variable([8])
        y_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + B_conv1)

    with tf.name_scope('pool1'):
        y_pool1 = max_pool_2x2(y_conv1)
    
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([6, 6, 8, 8])
        B_conv2 = bias_variable([8])
        y_conv2 = tf.nn.relu(conv2d(y_pool1, W_conv2) + B_conv2)
        
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([5, 5, 8, 16])
        B_conv3 = bias_variable([16])
        y_conv3 = tf.nn.relu(conv2d(y_conv2, W_conv3) + B_conv3)  
    
    with tf.name_scope('pool2'):
        y_pool2 = max_pool_2x2(y_conv3)
            
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([4, 4, 16, 32])
        B_conv4 = bias_variable([32])
        y_conv4 = tf.nn.relu(conv2d(y_pool2, W_conv4) + B_conv4)  
        
    with tf.name_scope('conv5'):
        W_conv5 = weight_variable([3, 3, 32, 64])
        B_conv5 = bias_variable([64])
        y_conv5 = tf.nn.relu(conv2d(y_conv4, W_conv5) + B_conv5)  
        
    with tf.name_scope('pool3'):
        y_pool3 = max_pool_2x2(y_conv5)
        
    with tf.name_scope('conv6'):
        W_conv6 = weight_variable([3, 3, 64, 128])
        B_conv6 = bias_variable([128])
        y_conv6 = tf.nn.relu(conv2d(y_pool3, W_conv6) + B_conv6)  

    with tf.name_scope('pool4'):
        y_pool4 = max_pool_2x2(y_conv6)

    with tf.name_scope('conv7'):
        W_conv7 = weight_variable([2, 2, 128, 256])
        B_conv7 = bias_variable([256])
        y_conv7 = tf.nn.relu(conv2d(y_pool4, W_conv7) + B_conv7)  
        
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([4 * 4 * 256, 200])
        B_fc1 = bias_variable([200])
        y_conv7_flat = tf.reshape(y_conv7, [-1, 4*4*256])
        y_fc1 = tf.nn.relu(tf.matmul(y_conv7_flat, W_fc1) + B_fc1)
        
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([200, 150])
        B_fc2 = bias_variable([150])
        y_fc2 = tf.nn.relu(tf.matmul(y_fc1, W_fc2) + B_fc2)
        
    with tf.name_scope('fc3'):
        W_fc3 = weight_variable([150, 100])
        B_fc3 = bias_variable([100])
        y_fc3 = tf.nn.relu(tf.matmul(y_fc2, W_fc3) + B_fc3)

    with tf.name_scope('fc4'):
        W_fc4 = weight_variable([100, 30])
        B_fc4 = bias_variable([30])
        y_fc4 = tf.matmul(y_fc3, W_fc4) + B_fc4
        
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        y_fc4_drop = tf.nn.dropout(y_fc4, keep_prob)    
        
    with tf.name_scope('fc5'):
        W_fc5 = weight_variable([30, 1])
        B_fc5 = bias_variable([1])
        y_fc5 = tf.matmul(y_fc4_drop, W_fc5) + B_fc5
        
    return y_fc5, keep_prob

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