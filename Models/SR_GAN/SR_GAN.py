"""
Super-resolution generative adversarial network
applies a deep network in combination with an adversarial network
GAN upsamples a low res image to super resolution images (LR->SR)

following design from: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
                        Network  https://arxiv.org/pdf/1609.04802.pdf

composes of convolution layers, batch normalization and parameterized ReLU (PRelU)

loss functions, comprises of reconstruction loss and adversarial loss:
    -uses perceptual loss, measuring MSE of features extracted by a VGG-19 network
        ->for a specific layer, we want their features to be matched st MSE is minimized
    -discriminator is trained using the typical GAN discriminator loss

"Goal is to train a generating function G that estimates for a given LR input image,
its corresponding HR counterpart."

general idea: train a generative model G with the goal of fooling a differentiable
discriminator D that is tained to distinguish super-resolved images from real images
"""

import tensorflow as tf
from Generator import Generator
from Discriminator import Discriminator
from Models.NPYDataGenerator import NPYDataGeneratorSR
import Data_Extraction.util as util
tf.keras.backend.set_floatx('float64')

### data ###
training_generator = NPYDataGeneratorSR(file_dir=util.OUTPUT_DIR + r"\deep_learning\PatternNet\TRAIN\super_res_NPY"
									  )
validation_generator = NPYDataGeneratorSR(file_dir=util.OUTPUT_DIR +r"\deep_learning\PatternNet\TEST\super_res_NPY"
						)

### models ###
generator = Generator()
discriminator = Discriminator()
vgg = tf.keras.applications.VGG19()
# only use 1st and 2nd layer
vgg = tf.keras.Sequential(vgg.layers[:7])

### loss functions ###
gen_loss = tf.keras.losses.BinaryCrossentropy()
discrim_loss = tf.keras.losses.BinaryCrossentropy()


### adam optimizer for SGD ###
optimizer = tf.keras.optimizers.Adam()

### intialize train metrics ###
gen_train_loss = tf.keras.metrics.Mean(name='gen_train_loss')
gen_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='gen_train_accuracy')
dis_train_loss = tf.keras.metrics.Mean(name='dis_train_loss')
dis_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='dis_train_accuracy')

### intialize test metrics ###
gen_test_loss = tf.keras.metrics.Mean(name='gen_test_loss')
gen_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='gen_test_accuracy')
dis_test_loss = tf.keras.metrics.Mean(name='dis_test_loss')
dis_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='dis_test_accuracy')

### generator train step ###
@tf.function
def gen_train_step(high_res, low_res):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = generator(low_res, training=True)
    # mean squared error in prediction
    m_loss = tf.keras.losses.MSE(high_res, predictions)

    # content loss
    v_pass = vgg(high_res)
    v_loss = tf.keras.losses.MSE(v_pass, predictions)

    # GAN loss + mse loss + feature loss
    loss = gen_loss(high_res, predictions) + v_loss + m_loss

  # apply gradients
  gradients = tape.gradient(loss, generator.trainable_variables)
  optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

  # update metrics
  gen_train_loss(loss)
  gen_train_accuracy(generator, predictions)



### discriminator train step ###
@tf.function
def dis_train_step(high_res, low_res):
  with tf.GradientTape() as tape:
    # discrim is a simple conv that perfroms binary classification
    # either SR or HR
    super_res = generator(low_res, training=False)
    # predict on gen output
    predictions = discriminator(super_res, training=True)
    loss = discrim_loss(high_res, predictions)

  # apply gradients
  gradients = tape.gradient(loss, discriminator.trainable_variables)
  optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

  # update metrics
  dis_train_loss(loss)
  dis_train_accuracy(high_res, predictions)

### generator test step ###
@tf.function
def gen_test_step(high_res, low_res):
  # feed test sample in
  predictions = generator(low_res, training=False)
  t_loss = gen_loss(high_res, predictions)

  # update metrics
  gen_test_loss(t_loss)
  gen_test_accuracy(high_res, predictions)


### discriminator test step ###
@tf.function
def dis_test_step(high_res, low_res):

    # feed test sample in
    super_res = generator(low_res, training=False)
    # predict on gen output
    predictions = discriminator(super_res, training=False)
    t_loss = gen_loss(high_res, predictions)

    # update metrics
    gen_test_loss(t_loss)
    gen_test_accuracy(high_res, predictions)


### TRAIN ###
EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  gen_train_loss.reset_states()
  gen_train_accuracy.reset_states()
  gen_test_loss.reset_states()
  gen_test_accuracy.reset_states()

  dis_train_loss.reset_states()
  dis_train_accuracy.reset_states()
  dis_test_loss.reset_states()
  dis_test_accuracy.reset_states()

  # reinitialize iter
  train_ds = iter(training_generator)
  test_ds = iter(validation_generator)
  # alternating training pattern
  if epoch%2:
      # train generator on even epochs
      for high_res, low_res in train_ds:
           gen_train_step(high_res, low_res)

      for test_high_res, test_low_res in test_ds:
          gen_test_step(test_high_res, test_low_res)

      template = 'Training Generator:\nEpoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
      print(template.format(epoch + 1,
                            gen_train_loss.result(),
                            gen_train_accuracy.result() * 100,
                            gen_test_loss.result(),
                            gen_test_accuracy.result() * 100))
  else:
      # train discriminator on odd epochs
      for high_res, low_res in train_ds:
          gen_train_step(high_res, low_res)

      for test_high_res, test_low_res in test_ds:
          gen_test_step(test_high_res, test_low_res)

      template = 'Training Discriminator:\nEpoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
      print(template.format(epoch + 1,
                            dis_train_loss.result(),
                            dis_train_accuracy.result() * 100,
                            dis_test_loss.result(),
                            dis_test_accuracy.result() * 100))




