import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import tensorflow as tf


def tfClassify(Folder):
    loadtfcmodel = 'ResultsAndData' + os.path.sep + 'EDSSpectra' + os.path.sep + Folder + os.path.sep + 'tfClfTrain' + os.path.sep
    
#x_test = s.reshape(1, -1)
#    y_test = np.zeros(2)
#    y_test[1] = 1
#    y_test = y_test.reshape(2,1)

    ### load the pre-trained classifier
#    global sess, new_saver1
#    global W_conv1, W_conv2, W_fc1, W_fc2, W_fc3, W_fco, b_conv1, b_conv2, b_fc1, b_fc2, b_fc3, b_fco
#    global x_image, x, y_, h_conv1, h_conv2, h_pool1, h_pool2, size_hp, h_flat, h_fc1, h_fc2, h_fc3, keep_prob, h_fc1_drop, y_conv
    
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(loadtfcmodel + 'tfClfmodel.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(loadtfcmodel + './'))
    W_conv1 = tf.get_collection('W_conv1')[0]
    b_conv1 = tf.get_collection('b_conv1')[0]
    W_conv2 = tf.get_collection('W_conv2')[0]
    b_conv2 = tf.get_collection('b_conv2')[0]
    W_fc1 = tf.get_collection('W_fc1')[0]
    b_fc1 = tf.get_collection('b_fc1')[0]
    W_fc2 = tf.get_collection('W_fc2')[0]
    b_fc2 = tf.get_collection('b_fc2')[0]
    W_fc3 = tf.get_collection('W_fc3')[0]
    b_fc3 = tf.get_collection('b_fc3')[0]
    W_fco = tf.get_collection('W_fco')[0]
    b_fco = tf.get_collection('b_fco')[0]
    
    num = 2040
    
    x = tf.placeholder(tf.float32, shape=[None, num])   # x image data, has batch size not defined, dimension 28*28
    y_ = tf.placeholder(tf.float32, shape=[None, 2])   # true distribution (one-hot vector)
    
    
    #W = tf.Variable(tf.zeros([2042, 2]))
    #b = tf.Variable(tf.zeros([2]))
    
    
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)
    
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides = [1,1,2,1], padding = 'SAME')
    #
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,1,10,1], strides = [1,1,2,1], padding = 'SAME')
    
    
    
    # reshape x: ?, width, height, color channel
    x_image = tf.reshape(x, [-1, 1, num, 1])
    
    # first layer
    #W_conv1 = weight_variable([1,10,1,8]) # The convolution will compute 32 features for each 5x5 patch
    #b_conv1 = bias_variable([8])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) ) #+ b_conv1
    h_pool1 = max_pool_2x2(h_conv1)
    
    # second layer
    #W_conv2 = weight_variable([1, 10, 8, 16])
    #b_conv2 = bias_variable([16])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) ) #+ b_conv2
    h_pool2 = max_pool_2x2(h_conv2)
    
    
    size_hp = (int)(np.str(tf.Tensor.get_shape(h_pool2)[2]))
    
    #x_image = tf.reshape(x, [-1, 1, 2042, 1])
    #
    #h_pool1 = max_pool_2x2(x_image)
    h_flat = tf.reshape(h_pool2, [-1, size_hp*16])
    
    
    #densely (fully) connected layer
    #W_fc1 = weight_variable([size_hp*16, 32]) 
    #b_fc1 = bias_variable([32])
    #
    #W_fc2 = weight_variable([32, 32]) 
    #b_fc2 = bias_variable([32])
    #
    #W_fc3 = weight_variable([32, 8]) 
    #b_fc3 = bias_variable([8])
    
    
    h_fc1 = (tf.matmul(h_flat, W_fc1) ) #+ b_fc1
    
    h_fc2 = (tf.matmul(h_fc1, W_fc2) ) #+ b_fc2
    
    h_fc3 = (tf.matmul(h_fc2, W_fc3) ) #+ b_fc3
    
    
    #h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*2046*2])
    #h_pool2_flat = tf.reshape(x, [-1, 2042])
    
    
    
    
    # dropout layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc3, keep_prob)
    
    # readout layer
    #W_fco = weight_variable([8, 2])
    #b_fco = bias_variable([2])
    
    
    
    # softmax layer
    #y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    #h_fc1 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
    y_conv=tf.nn.softmax(tf.matmul(h_fc3, W_fco) ) #+ b_fco
    
#    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    
    
    #init = tf.global_variables_initializer()
    #
    #sess.run(init)
    
    
    
    
    
    
    d = 1
    #    x_test = EDStestErr
    
    
    EDStestlabel = np.zeros((d*1, 2))
    EDStestlabel[0:d, 0] = 1
    EDStestlabel[d:d*1, 1] = 1
    
    #t_label = sess.run(y_conv, feed_dict={x: x_test, y_: EDStestlabel, keep_prob: 1.0})
    #t_label_ = np.argmax(t_label)
    #print(t_label_)
    
    
    #    acc = sess.run(accuracy, feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
    
    #print(t_label)
    #print("test accuracy is: " + np.str(np.round(acc*100)) + "%")    



    return [sess, W_conv1, W_conv2, W_fc1, W_fc2, W_fc3, W_fco, b_conv1, b_conv2, x_image, x, y_, h_conv1, h_conv2, h_pool1, h_pool2, size_hp, h_flat, h_fc1, h_fc2, h_fc3, keep_prob, h_fc1_drop, y_conv]



# import numpy as np
# noise3 = np.random.poisson(lam=20.0, size=((1,2042)))








#y_t = np.zeros((1,2))
#
#cd 
#cd Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_1_tf/code/
#import tfmodel
#import numpy as np
#noise3 = np.random.poisson(lam=20.0, size=((1,2042)))
#sess, W_conv1, W_conv2, W_fc1, W_fc2, W_fc3, W_fco, b_conv1, b_conv2, x_image, x, y_, h_conv1, h_conv2, h_pool1, h_pool2, size_hp, h_flat, h_fc1, h_fc2, h_fc3, keep_prob, h_fc1_drop, y_conv = tfmodel.tfClassify()
#tlabel = np.argmax(sess.run(y_conv, feed_dict={x: noise3, y_: y_t, keep_prob: 1.0}))
#





























































