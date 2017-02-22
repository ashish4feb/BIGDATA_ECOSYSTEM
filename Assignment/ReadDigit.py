import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

#Model Settings
Learning_Rate = 1e-4
#Can be modified to update result
Training_Iter = 4000

#tested on 0.5 also
Drop_Outs = 0.50
Batch_Size = 100

#Will be 0 after validating other settings
Validation_Size = 0

#Read data from csv file
train_file = "train.csv"
test_file = "test.csv"
train_data = pd.read_csv(train_file)

#get the image data seperate from labels
images = train_data.iloc[:,1:].values
images = images.astype(np.float)

#this to be commented when learning on sheered data
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
# Standard Deviation for Gaussian Noise
#std = tf.placeholder('float');

# first convolutional layer (will try with [4,4,1,32])
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


#recreating images (i.e. making a matrix reshape as a image is i.e. 28x28 )
image = tf.reshape(x, [-1,image_width , image_height,1])

n = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
image = image + n;
#convolution and max_pooling for 1st layer
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)



# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# second convolutional layer
W_conv3 = weight_variable([1,1,64,64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#h_pool3 = max_pool_2x2(h_conv3)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_conv3, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

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
predict = tf.argmax(y,1)

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

def get_next_batch(batch_size):
    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

# start TensorFlow session

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

saver = tf.train.Saver()

#saver = tf.train.import_meta_graph("save.ckpt"+'.meta');
#saver.restore(sess, "save.ckpt")
motive=input('Is this a training session ? ')
if motive=='yes' or motive=='YES':
    load=input('Load previous model ? ');
    if load=='yes' or load=='YES':
        Model=input("Name of the model to load : ");
        try:
            saver = tf.train.import_meta_graph(Model+'.meta');
            saver.restore(sess, Model)
        except :
            print('Error : File Does not exist!!')
    else:
        sess.run(init)
        Model=input("Name of new model to save : ");

    # visualisation variables
    train_accuracies = []
    validation_accuracies = []
    x_range = []

    display_timer = 1

    for i in range(Training_Iter):

        #get new batch
        batch_xs, batch_ys = get_next_batch(Batch_Size)        

        # check progress on every 1st,2nd,...,10th,20th,...,100th... step
        if i%display_timer == 0 or (i+1) == Training_Iter:
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
                                              y_: batch_ys, 
                                              keep_prob: 1.0})       
            if(Validation_Size):
                validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:Batch_Size], y_: validation_labels[0:Batch_Size], keep_prob: 1.0})                                 
                print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
                validation_accuracies.append(validation_accuracy)
            else:
                print('training_accuracy => %.4f for step %d'%(train_accuracy, i))

            train_accuracies.append(train_accuracy)
            x_range.append(i)

            # increase display_step
            if i%(display_timer*10) == 0 and i:
                display_timer *= 10
        # train on batch
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: Drop_Outs})


    if(Validation_Size):
        validation_accuracy = accuracy.eval(feed_dict={x: validation_images, y_: validation_labels, keep_prob: 1.0})
        print('validation_accuracy => %.4f'%validation_accuracy)
        plt.plot(x_range, validation_accuracies,'-g', label='Validation')
    
    if load=='Yes' or load=='yes':
        save_same=input("Would you like to save this ? ");
        if(save_same=='no'):
            temp=input("Enter Another File Name : ");
            saver.save(sess,temp);
        else:
            saver.save(sess,Model);
    else:
        saver.save(sess, Model);
      
    plt.plot(x_range, train_accuracies,'-b', label='Training')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax = 1.1, ymin = 0.7)
    plt.ylabel('Accuracy->')
    plt.xlabel('Step->')
    plt.show()
    
else:
    Model=input("Name of the model to test on : ");
    try:
        saver = tf.train.import_meta_graph(Model+'.meta');
        saver.restore(sess, Model)
    except :
        print('Error : File Does not exist!!')
        sys.exit()
    #Testing data used for prediction now
    test_data = pd.read_csv(test_file)
    test_data = test_data.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    test_data = np.multiply(test_data, 1.0/255.0)

    # predict test set
    predicted_lables = np.zeros(test_data.shape[0])
    for i in range(0,test_data.shape[0]//Batch_Size):
        predicted_lables[i*Batch_Size : (i+1)*Batch_Size] = predict.eval(feed_dict={x: test_data[i*Batch_Size : (i+1)*Batch_Size], 
                                                                                keep_prob: 1.0})

    # save results
    np.savetxt('submission_softmax.csv', 
           np.c_[range(1,len(test_data)+1),predicted_lables], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')

#TensorFlow session closed
sess.close()
