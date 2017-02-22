import numpy as np
import pandas as pd

from PIL import Image


#Read data from csv file
train_file = "train.csv"
train_data = pd.read_csv(train_file)

images = train_data.iloc[:,1:].values
images = images.astype(np.float)

labels_unroll = train_data[[0]].values.ravel()

image_size = images.shape[1]

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

for x in range(0,32001, 2000):
    nTrain1 = np.empty((1,785))
    for i in range(x,x+2000):
        one_img = images[i].reshape(image_width,image_height)
        lab = labels_unroll[i]
        for r in range(0,4,2):
            for c in range(0,4,2):
                tmp = np.zeros((28,28))
                tmp[4-r:4-r+24,4-c:4-c+24] = one_img[r:r+24,c:c+24]
                a = np.array(tmp.flatten())
                #print(lab)
                b = (np.insert(a,0,lab)).reshape(1,785)
                nTrain1 = np.append(nTrain1,b,axis=0)
    print(x)
    np.savetxt("foo%s.csv" % x, nTrain1, delimiter=",");

