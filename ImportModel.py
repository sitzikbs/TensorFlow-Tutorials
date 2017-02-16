"""
Created on Wed Feb 15 2017

@author: Itzik Ben Shabat www.itzikbs.com
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
# Get the Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#===============================================================================
# Restore trained Graph
#===============================================================================
SummariesDirectory = "C:/Users/Itzik/Google Drive/Research/Python Files/TensorFlow Tutorials/CheckPoints/"
ModelName = "MNIST-model-400"
sess = tf.Session()
new_saver = tf.train.import_meta_graph(SummariesDirectory + ModelName + ".meta")
new_saver.restore(sess, SummariesDirectory + ModelName)

#all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
y_conv = tf.get_collection("y_conv")[0]
x = tf.get_collection("x")[0]
y_ = tf.get_collection("y_")[0]
keep_prob =  tf.get_collection("keep_prob")[0]

ImageIndex = 3
InputImage = mnist.test.images[ImageIndex].reshape(1,784)
CorrectLabel = mnist.test.labels[ImageIndex].reshape(1,10)

logit = sess.run(y_conv,feed_dict={ x: InputImage, y_: CorrectLabel, keep_prob: 1.0})
prediction = sess.run(tf.argmax(logit,1))
digit = sess.run(tf.argmax(CorrectLabel,1))
print("Prediction : %d, Actual : %d"% (prediction, digit))

plt.imshow(InputImage.reshape(28,28),cmap = 'gray')
plt.show()
#"predicted: %d, Actual:%d", prediction,
