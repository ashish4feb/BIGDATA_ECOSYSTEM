import numpy as np
import pandas as pd
import tensorflow as tf

#Model Sttings
Learning_Rate = 1e-4
#Can be modified to update result
Training_Iter = 10000

Drop_Outs = 0.5
Batch_Size = 100

#Will reduce after validating other settings
Validation_Size = 0

#Read data from csv file
train_file = "train.csv"
test_file = "test.csv"
train_data = pd.read_csv(train_file)

#get the image data seperate from labels
images = train_data.iloc[:,1:].values
images = images.astype(np.float)

images = np.multiply(images, 1.0/255.0)

image_size = images.shape[1]

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

#get the label data unrolled
labels_unroll = train_data[[0]].values.ravel()
#unique labels, will be used as no. of classes
unique_labels = np.unique(labels_unroll).shape[0]

#define func to convert dense labels into one_hot_vectors
def one_hot_convert(dense_labels,num_of_classes):
	num_of_labels = dense_labels.shape[0]
	index_offset = np.arange(num_of_labels) * num_of_classes
	one_hot_lables = np.zeros((num_of_labels, num_of_classes))
	one_hot_lables.flat[index_offset + dense_labels.ravel()] = 1
	return one_hot_lables
	
#get the full matrix of one_hot_vectors
labels = one_hot_convert(labels_unroll, unique_labels)
labels = labels.astype(np.uint8)

#set the validation images and labels
validation_images = images[:Validation_Size]
validation_labels = labels[:Validation_Size]

#set the training images seperate from validation images
train_images = images[Validation_Size:]
train_labels = labels[Validation_Size:]

#--------------
#Setting up TensorFlow

#can be experimented with but do not set to 0
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
#can be experimented with but do not set to 0
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#can be modified for experimenting with the model
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

