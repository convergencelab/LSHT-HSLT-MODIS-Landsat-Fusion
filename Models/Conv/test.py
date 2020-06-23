import tensorflow as tf
import load_EuroSat as lE
import os
import numpy as np

### initialize model ###
vgg = tf.keras.applications.VGG19(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=[224, 224, 3],
    pooling=None,
    classes=10,
    classifier_activation="softmax"
)

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
train_data.prepare_for_training(batch_size=10)
test_data.prepare_for_testing()

batch = train_data.get_train_batch()

sample = batch[0][0]
sample = np.array(sample)[np.newaxis, ...]

sample = tf.image.resize(sample, (224, 224))
sample = tf.keras.applications.vgg19.preprocess_input(sample)

out = vgg.predict(sample)