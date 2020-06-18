"""
Looking at output of vgg-19 layer
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from Models.Conv import load_EuroSat as lE
from tqdm import tqdm
### get data ###
"""
using eurosat dataset, this dataset uses the sentenial-2 collected satellite images
"""
euro_path = r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\EuroSat"

### Hyperparameters ###
batch_size = 1

### initalize loaders ###
train_data = lE.training_data_loader(
    base_dir=os.path.join(euro_path, "train_data"))
test_data = lE.testing_data_loader(
    base_dir=os.path.join(euro_path, "test_data"))
### load data ###
train_data.load_data()
test_data.load_data()

### prep train-data ###
train_data.prepare_for_training(batch_size=batch_size)
test_data.prepare_for_testing()


### initialize model ###
vgg = tf.keras.applications.VGG19(
                            include_top=False,
                            weights='imagenet',
                            input_tensor=None,
                            input_shape=[64, 64, 3],
                            pooling=None,
                            classes=1000,
                            classifier_activation="softmax"
                        )

"""Model: "vgg19"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 64, 64, 3)]       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 64, 64, 64)        1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 64, 64, 64)        36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 32, 32, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 32, 32, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 32, 32, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 16, 16, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 16, 16, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 16, 16, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 16, 16, 256)       590080    
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 16, 16, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 8, 8, 256)         0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 8, 8, 512)         1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 8, 8, 512)         2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 8, 8, 512)         2359808   
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 8, 8, 512)         2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 4, 4, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 4, 4, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 4, 4, 512)         2359808   
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 4, 4, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 2, 2, 512)         0         
=================================================================
Total params: 20,024,384
Trainable params: 20,024,384
Non-trainable params: 0
_________________________________________________________________"""
### pass through ###
batch = train_data.get_train_batch()
x = batch[0].numpy()
output = vgg(x)

# pass through input
_x = vgg.get_layer(index=0)(x)
### get layers ###
for i in range(2):
    # up to block1_conv2 (Conv2D)
    _x = vgg.get_layer(index=i)(_x)

### plot different convolution outputs from second layer ###
fig, ax = plt.subplots(4,4)
ax[0, 0].imshow(x[0])

for i in range(2):
    ax[0, i+1].imshow(_x[0][i])
for i in range(3):
    ax[1, i].imshow(_x[0][i+4])
#for i in range(3):
 #   ax[2, i].imshow(_x[0][i+8])
#for i in range(3):
   # ax[3, i].imshow(_x[0][i+12])

plt.show()