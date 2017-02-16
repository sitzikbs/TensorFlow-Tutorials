# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:03:08 2017

@author: Itzik Ben Shabat www.itzikbs.com
"""

import tensorflow as tf

#Get the Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32,[None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax( tf.matmul(x,W) + b )

y_ = tf.placeholder( tf.float32, [None, 10] )
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
correct_predictions = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

test_Accuracy = sess.run(accuracy,feed_dict={x:  mnist.test.images, y_: mnist.test.labels})
print(test_Accuracy)

