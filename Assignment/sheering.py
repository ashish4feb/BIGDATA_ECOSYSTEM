import numpy as np
import pandas as pd
from skimage import transform as tf

from PIL import Image


#Read data from csv file
train_file = "train.csv"
test_file = "test.csv"
train_data = pd.read_csv(train_file)

images = train_data.iloc[:,1:].values
images = images.astype(np.float)

labels_unroll = train_data[[0]].values.ravel()
print(labels_unroll[2])

#images = np.multiply(images, 1.0/255.0)

image_size = images.shape[1]

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

print(image_width,image_height)

one_image = images[0].reshape(image_width,image_height)

print(one_image.shape)

for x in range(0,32000, 2000):
    nTrain = np.empty((1,785))
    for i in range(x,x+2000):
       lab = labels_unroll[i]
       for r in range(-49,49,96):
            image = Image.fromarray(images[i].reshape(28,28))
            image = np.multiply(image, 1.0/255.0)
            afine_tf = tf.AffineTransform(shear=r/70)
            modified = tf.warp(image, afine_tf)
            a = np.array(modified.flatten())
            b = (np.insert(a,0,lab)).reshape(1,785)
            nTrain = np.append(nTrain,b,axis=0)
       print(i)
    np.savetxt("foo%s.csv" % x, nTrain, delimiter=",");
