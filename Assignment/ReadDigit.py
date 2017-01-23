import numpy as np
import pandas as pd
import tensorflow as tf

#Model Sttings
Learning_Rate = 1e-4
#Can be modified to update result
Training_Iter = 10000

Drop_Outs = 0.5
Batch_Size = 50

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

# images
x = tf.placeholder('float', shape=[None, image_size])
# labels
y_ = tf.placeholder('float', shape=[None, unique_labels])

# first convolutional layer (will try with [4,4,1,32])
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#recreating images (i.e. making a matrix reshape as a image is i.e. 28x28 )
image = tf.reshape(x, [-1,image_width , image_height,1])

#convolution and max_pooling for 1st layer
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# Prepare for visualization
# display 32 fetures in 4 by 8 grid
layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, 4 ,8))  

# reorder so the channels are in the first dimension, x and y follow.
layer1 = tf.transpose(layer1, (0, 3, 1, 4, 2))
layer1 = tf.reshape(layer1, (-1, image_height*4, image_width*8)) 

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)

# Prepare for visualization
# display 64 fetures in 4 by 16 grid
layer2 = tf.reshape(h_conv2, (-1, 14, 14, 4 ,16))

# reorder so the channels are in the first dimension, x and y follow.
layer2 = tf.transpose(layer2, (0, 3, 1, 4,2))
layer2 = tf.reshape(layer2, (-1, 14*4, 14*16)) 


# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# readout layer for deep net
W_fc2 = weight_variable([1024, unique_labels])
b_fc2 = bias_variable([unique_labels])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


# optimisation function(can test with the optimizer function)
train_step = tf.train.AdamOptimizer(Learning_Rate).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# prediction function
#[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
#return index of max probablity
predict = tf.argmax(y,1)


