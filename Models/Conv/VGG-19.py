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
import load_EuroSat as lE
from datetime import datetime
import csv
tf.autograph.set_verbosity(3)

_CITATION = """
    @misc{helber2017eurosat,
    title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
    author={Patrick Helber and Benjamin Bischke and Andreas Dengel and Damian Borth},
    year={2017},
    eprint={1709.00029},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}"""

### HYPER-PARAMS ###
# it is noted in https://arxiv.org/pdf/1609.04802.pdf
# momentum=0.9, learning rate = 0.0001
learning_rate = 0.01
weight_decay = 0
momentum = 0.9
batch_size = 256

### manually store data ###
fname = "/project/6026587/x2017sre/manual_logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_train_results.csv"
# manually write info
with open(fname, "w+") as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["train loss", "train acc.","test loss", "test acc."])
### get data ###
"""
using eurosat dataset, this dataset uses the sentenial-2 collected satellite images
"""
euro_path = r"/project/6026587/x2017sre/EuroSat/"



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

### initialize model with pre-trained weights ###
base_vgg = tf.keras.applications.VGG19(
    include_top=False,
    weights=r"/home/x2017sre/projects/def-jhughe54/x2017sre/.keras/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",
    input_tensor=None,
    input_shape=[224, 224, 3],
    pooling=None,
    classes=1000,
    classifier_activation="softmax"
)
 # build top for model. #
x = base_vgg.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x = tf.keras.layers.Dense(512,activation='relu')(x) #dense layer 2
preds = tf.keras.layers.Dense(10,activation='softmax')(x) # probability of image
vgg = tf.keras.models.Model(inputs=base_vgg.input, outputs=preds)



### loss function ###
"""
Use MSE loss:

    ref -> "https://towardsdatascience.com/loss-functions-based-on-feature-activation-and-style-loss-2f0b72fd32a9"
"""

m_loss = tf.keras.losses.MSE

### adam optimizer for SGD ###
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=momentum)

### intialize metrics ###
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_vgg-19_acc')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_vgg-19_acc')


### train step ###
@tf.function
def train_step(idx, sample, label):
    with tf.GradientTape() as tape:
        # preprocess for vgg-19
        sample = tf.image.resize(sample, (224, 224))
        sample = tf.keras.applications.vgg19.preprocess_input(sample)

        predictions = vgg(sample, training=True)
        # mean squared error in prediction
        loss = tf.keras.losses.MSE(label, predictions)

    # apply gradients
    gradients = tape.gradient(loss, vgg.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vgg.trainable_variables))

    # update metrics
    train_loss(loss)
    train_accuracy(label, predictions)

### generator test step ###
@tf.function
def test_step(idx, sample, label):
    # preprocess for vgg-19
    sample = tf.image.resize(sample, (224, 224))
    sample = tf.keras.applications.vgg19.preprocess_input(sample)
    # feed test sample in
    predictions = vgg(sample, training=False)
    t_loss = tf.keras.losses.MSE(label, predictions)

    # update metrics
    test_loss(t_loss)
    test_accuracy(label, predictions)



### tensorboard ###

# initialize logs #
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = './logs/gradient_tape/' + current_time + '/train'
test_log_dir = './logs/gradient_tape/' + current_time + '/test'
graph_log_dir = './logs/gradient_tape/' + current_time + '/graph'
# image_log_dir = './logs/gradient_tape/' + current_time + '/image'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
graph_writer = tf.summary.create_file_writer(graph_log_dir)



### Weights Dir ###
if not os.path.isdir('./checkpoints'):
    os.mkdir('./checkpoints')

### TRAIN ###
EPOCHS = 1000
NUM_CHECKPOINTS_DIV = int(EPOCHS / 4)
save_c = 1

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()


    for idx in range(train_data.get_ds_size() // batch_size):

        # train step
        batch = train_data.get_train_batch()
        for sample, label in zip(batch[0], batch[1]):
            # testing to see what the input is looking like
            if epoch == 0 == idx:
                fig, ax = plt.subplots(1,1)
                ax.imshow(sample)
                # ax.set_title(label)
                plt.savefig(r"/project/6026587/x2017sre")
            sample = np.array(sample)[np.newaxis, ...]
            label = np.array(label)[np.newaxis, ...]

            tf.summary.trace_on()
            train_step(idx, sample, label)
        # trace graph #
        with graph_writer.as_default():
            tf.summary.trace_export(name='vgg-19', step=0)

        # write to train-log #
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        # test step #
        batch = test_data.get_test_batch(batch_size=batch_size)
        for sample, label in zip(batch[0], batch[1]):
            sample = np.array(sample)[np.newaxis, ...]
            label = np.array(label)[np.newaxis, ...]
            test_step(idx, sample, label)

        # write to test-log #
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

    # manually write info
    with open(fname, "w+") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([train_loss.result(),
                             train_accuracy.result(),
                             test_loss.result(),
                              test_accuracy.result()])
    ### save weights ###
    if not epoch % NUM_CHECKPOINTS_DIV:
        vgg.save_weights('/project/6026587/x2017sre/checkpoints/{}/{}'.format(current_time, save_c))
        save_c += 1

    if not epoch % 100:
        ### outputs every 100 epochs so .out file from slurm is not huge. ###
        template = 'Training VGG-19:\nEpoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
