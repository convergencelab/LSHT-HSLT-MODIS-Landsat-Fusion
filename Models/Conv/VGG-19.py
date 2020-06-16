"""
The SR-GAN uses the 2nd layer of the VGG-19 to include feature detection
in the perceptual loss function.
--rather than using a model pretrained on image net, it may be more useful to use a pre-trained model, trained on
  data more similar to that of the scenes we are using for landsat-modis super resolution

  -> idea 1) train a binary classifier to differentiate landsat from modis: this does not really achieve the goal
  of deriving meaningful features from the image. The major difference between landsat and modis is the resolution
  so this sort of classifier would likely produce a model that distinguishes high res from low res.
  -> idea 2) explore different landcover/other feature classification approaches on both landsat and modis images:
          a) train both and then average weights
          b) scale up modis and train on same model ( may cause too much variance between scenes )

"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import tensorflow_datasets as tfds
import load_EuroSat as lE
_CITATION = """
    @misc{helber2017eurosat,
    title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
    author={Patrick Helber and Benjamin Bischke and Andreas Dengel and Damian Borth},
    year={2017},
    eprint={1709.00029},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}"""

### get data ###
"""
using eurosat dataset, this dataset uses the sentenial-2 collected satellite images
"""
euro_path = r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\EuroSat"

### Hyperparameters ###
batch_size = 12
classes_to_include = [
    'Residential'
]
### initalize loaders ###
train_data = lE.training_data_loader(
    base_dir=os.path.join(euro_path, "train_data"))
test_data = lE.testing_data_loader(
    base_dir=os.path.join(euro_path, "test_data"))
### load data ###
train_data.load_data(selected_classes=classes_to_include)
test_data.load_data()

### prep train-data ###
train_data.prepare_for_training(batch_size=batch_size)
test_data.prepare_for_testing()


r"""
### initialize model ###
vgg = tf.keras.applications.VGG19(
                            include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=None,
                            pooling=None,
                            classes=1000,
                            classifier_activation="softmax",
                            training=True
                        )
"""
### loss function ###
"""
Use MSE loss:
  
    ref -> "https://towardsdatascience.com/loss-functions-based-on-feature-activation-and-style-loss-2f0b72fd32a9"
"""
r"""
m_loss = tf.keras.losses.MSE

### adam optimizer for SGD ###
optimizer = tf.keras.optimizers.Adam()

### intialize metrics ###
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_vgg-19_acc')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_vgg-19_acc')


### train step ###
@tf.function
def train_step(sample, label):
  with tf.GradientTape() as tape:
    predictions = vgg(sample, training=True)
    # mean squared error in prediction
    loss = tf.keras.losses.MSE(label, predictions)

  # apply gradients
  gradients = tape.gradient(loss, vgg.trainable_variables)
  optimizer.apply_gradients(zip(gradients, vgg.trainable_variables))

  # update metrics
  train_loss(loss)
  train_accuracy(vgg, predictions)

### generator test step ###
@tf.function
def test_step(sample, label):
  # feed test sample in
  predictions = vgg(sample, training=False)
  t_loss = tf.keras.losses.MSE(label, predictions)

  # update metrics
  test_loss(t_loss)
  test_accuracy(label, predictions)

### Weights Dir ###
if not os.isdir('./checkpoints'):
    os.mkdir('./checkpoints')

### TRAIN ###
EPOCHS = 1000
NUM_CHECKPOINTS_DIV = int(EPOCHS/4)
save_c = 1

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    # train step
    for sample, label in train_data:
      train_step(sample, label)
    # test step
    for sample, label in test_data:
      test_step(sample, label)

    ### save weights ###
    if not epoch % NUM_CHECKPOINTS_DIV:
        vgg.save_weights('./checkpoints/my_checkpoint_{}'.format(save_c))
        save_c += 1
    if not epoch % 100:
        ### outputs every 100 epochs so .out file from slurm is not huge. ###
        template = 'Training VGG-19:\nEpoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
"""