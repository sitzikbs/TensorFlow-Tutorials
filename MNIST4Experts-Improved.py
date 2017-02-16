"""
Created on Wed Feb  8 14:03:08 2017

@author: Itzik Ben Shabat www.itzikbs.com
"""

import tensorflow as tf
import numpy as np
import math
# Get the Data
from tensorflow.examples.tutorials.mnist import input_data

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def CNNLayer(InputTensor, InputDim, OutputDim, PatchDim, LayerName, ActivationType = tf.nn.relu):
    with tf.name_scope(LayerName):
        with tf.name_scope('weights'):
            weights = weight_variable([PatchDim, PatchDim, InputDim, OutputDim])
            variable_summaries(weights, LayerName + '/weights')
        with tf.name_scope('biases'):    
            biases = bias_variable([OutputDim])
            variable_summaries(biases, LayerName + '/biases')
        with tf.name_scope('preactivatoins'): 
            preactivations = conv2d(InputTensor, weights) + biases
    activations = ActivationType(preactivations)
    return activations

def NNLayer(InputTensor, InputDim, OutputDim, LayerName, ActivationType = tf.nn.relu):
    with tf.name_scope(LayerName):
        with tf.name_scope('weights'):
            weights = weight_variable([InputDim, OutputDim])
            variable_summaries(weights, LayerName + '/weights')
        with tf.name_scope('biases'):    
            biases = bias_variable([OutputDim])
            variable_summaries(biases, LayerName + '/biases')
        with tf.name_scope('preactivatoins'): 
            preactivations = tf.matmul(InputTensor, weights) + biases
    activations = ActivationType(preactivations)
    return activations            
    
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
SummariesDirectory = "C:/Users/Itzik/Google Drive/Research/Python Files/TensorFlow Tutorials/Summaries"
BatchSize = 50
nEpochs = 100
nTrainImages = len(mnist.train.images)
Image_size = np.sqrt(len(mnist.train.images[0]))
nIterations = math.ceil(nTrainImages/BatchSize)

sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

# %% Building a multilayer neural network
with tf.name_scope('Input_Reshape'):
    x_image = tf.reshape(x, [-1,28,28,1])
tf.summary.image('input_Images', x_image, 10)
    
#First Convolutional Layer - take 28X28 images, get 32 activation maps and reduce to 14X14
CNNHidden1Activations = CNNLayer(x_image, 1, 32, 5, 'CNN_Hidden_Layer_1')    
h_pool1 = max_pool_2x2(CNNHidden1Activations)
#Second Convolutional Layer - take 14X14 images, get 64 activation maps and reduce to 7X7
CNNHidden2Activations = CNNLayer(h_pool1, 32, 64, 5, 'CNN_Hidden_Layer_2')  
h_pool2 = max_pool_2x2(CNNHidden2Activations)

#Fully Connected Layer - flatten the 64 activation maps of 7X7 images, get 1X1024 descriptor of the image
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
FCHiddenActivations = NNLayer(h_pool2_flat, 7*7*64, 1024, 'Fully_Connected_Hidden_Layer_3')

#Dropout
with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)
    dropped = tf.nn.dropout(FCHiddenActivations, keep_prob)

#Output
with tf.name_scope('Output'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(dropped, W_fc2) + b_fc2

#Computing the prediction and accuracy
    with tf.name_scope('cross_entropy'):
          cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    cross_entropy_summary = tf.summary.scalar('cross entropy', cross_entropy)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    with tf.name_scope('cross_entropy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    
#Add inputs and outputs for model import
tf.add_to_collection("keep_prob", keep_prob)
tf.add_to_collection("x", x)
tf.add_to_collection("y_", y_)
tf.add_to_collection("y_conv", y_conv)

merged_summary = tf.summary.merge([accuracy_summary, cross_entropy_summary])

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep=3)
CeckPointFilename = "C:/Users/Itzik/Google Drive/Research/Python Files/TensorFlow Tutorials/CheckPoints/MNIST-model"
Summaries_Path = "C:/Users/Itzik/Google Drive/Research/Python Files/TensorFlow Tutorials/Summaries/"
#merged = tf.merge_all_summaries() #merge all summaries
train_writer = tf.summary.FileWriter(Summaries_Path + 'train')
validation_writer = tf.summary.FileWriter(Summaries_Path + 'validation')
test_writer = tf.summary.FileWriter(Summaries_Path + 'test')
file_writer = tf.summary.FileWriter(Summaries_Path, sess.graph)

#Training - may take a long time to complete
for iEpoch in range(nEpochs):
    for i in range(nIterations):
        batch = mnist.train.next_batch(BatchSize)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i%100 == 0:
            train_ce, train_accuracy, train_summ = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={x : batch[0], y_ : batch[1], keep_prob: 1.0})
            validation_ce, Validation_acc, Validation_summ = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={x : mnist.validation.images, y_ : mnist.validation.labels, keep_prob: 1.0})
            test_ce, test_acc, test_summ = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={x :  mnist.test.images, y_ : mnist.test.labels, keep_prob: 1.0})
            print("Epoch %d, step %d, training accuracy %g"%(iEpoch,i, train_accuracy))
            print("Epoch %d, step %d, validation accuracy %g"%(iEpoch,i, Validation_acc)) 
            print("Epoch %d, step %d, test accuracy %g"%(iEpoch, i, test_acc))  
          
    train_ce, train_accuracy, train_summ = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={x : batch[0], y_ : batch[1], keep_prob: 1.0})
    train_writer.add_summary(train_summ, iEpoch)
    validation_ce, Validation_acc, Validation_summ = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={x : mnist.validation.images, y_ : mnist.validation.labels, keep_prob: 1.0})
    validation_writer.add_summary(Validation_summ, iEpoch)
    test_ce, test_acc, test_summ = sess.run([cross_entropy, accuracy, merged_summary], feed_dict={x :  mnist.test.images, y_ : mnist.test.labels, keep_prob: 1.0})
    test_writer.add_summary(test_summ, iEpoch)
    print("Finished an Epoch")
    print("Epoch %d, training accuracy %g"%(iEpoch, train_accuracy))
    print("Epoch %d, validation accuracy %g"%(iEpoch, Validation_acc)) 
    print("Epoch %d, test accuracy %g"%(iEpoch, test_acc))        
    saver.save(sess, CeckPointFilename , global_step = i + iEpoch*nIterations)

print("test accuracy %g"%accuracy.eval(feed_dict={ x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
