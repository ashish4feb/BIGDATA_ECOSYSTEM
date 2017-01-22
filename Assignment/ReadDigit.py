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
