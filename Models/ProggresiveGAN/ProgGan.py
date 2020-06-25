"""
Progressive Growing Gan
Following: https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/


The Progressive Growing GAN is an extension to the GAN training procedure that
involves training a GAN to generate very small images, such as 4×4,
and incrementally increasing the size of the generated images to 8×8, 16×16,
until the desired output size is met. This has allowed the progressive GAN to generate
photorealistic synthetic faces with 1024×1024 pixel resolution.

described in the 2017 paper by Tero Karras, et al. from Nvidia
titled “Progressive Growing of GANs for Improved Quality, Stability, and Variation.”
"""

from math import sqrt
from numpy import load, asarray, zeros, ones
from numpy.random import randn, randint
from skimage.transform import resize
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU, Layer, Add
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import backend
from matplotlib import pyplot

### Custom Components ###

# Weighted Sum #
class WeightedSum(Add):
    """
    Merge layer, combines activations from two input layers
    such as two input paths in a discriminator or two output
    layers in a generator

    This is used during the growth phase of training when model
    is in transition from one image size to a new image size
    i.e 4x4 -> 8x8
    """
    # init with default value
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # merge two inputs with weight measured by alpha #
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output

# Minibatch Stdev Layer #
class MinibatchStdev(Layer):
    """
    Only used in output block of the discriminator layer
    This layer provides a statistical summary of the batch of activations.
    The discriminator can learn to better detect batches of fake samples
    from batches of real samples. Therefore this layer encourages the generator
    (trained via discriminator) to create batches of samples with realistic
    batch statistics.
    """
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate the mean value for each pixel across channels
        mean = backend.mean(inputs, axis=0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = backend.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8
        # square root of the variance (stdev)
        stdev = backend.sqrt(mean_sq_diff)
        # calculate the mean standard deviation across each pixel coord
        mean_pix = backend.mean(stdev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = backend.shape(inputs)
        output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        # concatenate with the output
        combined = backend.concatenate([inputs, output], axis=-1)
        return combined

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
        return tuple(input_shape)

# Pixel Normalization #
class PixelNormalization(Layer):
    """
    The generator and discriminator in Progressive growing GAN differs from
    most as it does not use Batch Normalization. instead each pixel in activation
    maps are normalized to unit length. this is known as pixelwise feature vector
    normalization. Normalization is only usd in the generator.
    """
    # initialize the layer
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate square pixel values
        values = inputs ** 2.0
        # calculate the mean pixel values
        mean_values = backend.mean(values, axis=-1, keepdims=True)
        # ensure the mean is not zero
        mean_values += 1.0e-8
        # calculate the sqrt of the mean squared value (L2 norm)
        l2 = backend.sqrt(mean_values)
        # normalize values by the l2 norm
        normalized = inputs / l2
        return normalized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape


# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    """
    using wasserstein loss to simplify implementation
    :param y_true: groundtruth img
    :param y_pred: prediction img
    :return: wasserstein loss
    """
    return backend.mean(y_true * y_pred)


### Define Models ###

# Disriminator Model #

def add_discriminator_block(old_model, n_input_layers=3):
    """
    build a discriminator block,
    this is to be implemented in a growth phase
    :param old_model:
    :param n_input_layers:
    :return:
    """
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = max_norm(1.0)
    # get shape of existing model
    in_shape = list(old_model.input.shape)
    # define new input shape as double the size
    input_shape = (in_shape[-2].value*2, in_shape[-2].value*2, in_shape[-1].value)
    in_image = Input(shape=input_shape)
    # define new input processing layer
    d = Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # define new block
    d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = AveragePooling2D()(d)
    block_new = d
    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    # define straight-through model
    model1 = Model(in_image, d)
    # compile model
    model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    # downsample the new larger image
    downsample = AveragePooling2D()(in_image)
    # connect old input processing to downsampled new input
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    # fade in output of old model input layer with new input
    d = WeightedSum()([block_old, block_new])
    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    # define straight-through model
    model2 = Model(in_image, d)
    # compile model
    model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    return [model1, model2]

def define_discriminator(n_blocks, input_shape=(4,4,3)):
    """
    define the discriminator model
    :param n_blocks: number of blocks in current phase
    :param input_shape: shape of input vec
    :return: The function returns a list where each element in the list
    contains two models. The first model is the ‘normal model‘ or
    straight through model, and the second is the version of the model
    that includes the old 1×1 and new block with the weighted sum,
    used for the transition or growth phase of training.
    """
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = max_norm(1.0)
    model_list = list()
    # base model input
    in_image = Input(shape=input_shape)
    # conv 1x1
    d = Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 3x3 (output block)
    d = MinibatchStdev()(d)
    d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 4x4
    d = Conv2D(128, (4,4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # dense output layer
    d = Flatten()(d)
    out_class = Dense(1)(d)
    # define model
    model = Model(in_image, out_class)
    # compile model
    model.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    # store model
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_discriminator_block(old_model)
        # store model
        model_list.append(models)
    return model_list

# Generator Model #

def add_generator_block(old_model):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = max_norm(1.0)
    # get the end of the last block: -2 bc -1 is dense layer
    block_end = old_model.layers[-2].output
    # upsample, and define new block
    # upsamples by a factor of 2, using nearest neighbour by default
    upsampling = UpSampling2D()(block_end)
    g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(upsampling)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    # if > 0 nothing, if < 0; multiply by alpha
    # leaky relu prevents dying gradient: allows for negative activations

    g = LeakyReLU(alpha=0.2)(g)
    # add new output layer
    out_image = Conv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    # define model
    model1 = Model(old_model.input, out_image)
    # get the output layer from old model
    out_old = old_model.layers[-1]
    # connect the upsampling to the old output layer
    out_image2 = out_old(upsampling)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([out_image2, out_image])
    # define model
    model2 = Model(old_model.input, merged)
    return [model1, model2]


# define generator models
def define_generator(latent_dim, n_blocks, in_dim=4):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = max_norm(1.0)
    model_list = list()
    # base model latent input
    in_latent = Input(shape=(latent_dim,))
    # linear scale up to activation maps
    g = Dense(128 * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const)(in_latent)
    g = Reshape((in_dim, in_dim, 128))(g)
    # conv 4x4, input block
    g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    # conv 3x3
    g = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g).
    # conv 1x1, output block
    out_image = Conv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    # define model
    model = Model(in_latent, out_image)
    # store model
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_generator_block(old_model)
        # store model
        model_list.append(models)
    return model_list

# Progressive Growing GAN #

# define composite models for training generators via discriminators
def define_composite(discriminators, generators):
	model_list = []
	# create composite models
	for g_models, d_models in zip(generators, discriminators):
		# straight-through model
		d_models[0].trainable = False
		model1 = Sequential()
		model1.add(g_models[0])
		model1.add(d_models[0])
		model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
		# fade-in model
		d_models[1].trainable = False
		model2 = Sequential()
		model2.add(g_models[1])
		model2.add(d_models[1])
		model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
		# store
		model_list.append([model1, model2])
	return model_list

### training ###

# load dataset
def load_real_samples(filename):
    # TODO: make loading function for my data
	# load dataset
	data = load(filename)
	# extract numpy array
	X = data['arr_0']
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X

# select real samples
def generate_real_samples(dataset, n_samples):
    # TODO: generalize to SR problem
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = -ones((n_samples, 1))
	return X, y

"""
Training occurs in two phases:
    1. a fade in phase that involves the transition from a lower-resolution 
    to a higher resolution image
    2. normal phase involving the training of the model
    
During phase in, the alpha value of the wieghted sum layers in the discriminator
and generator model at a given layer requires a linear transition from 0.0 to 1.0 
"""

# update the alpha value on each instance of WeightedSum
def update_fadein(models, step, n_steps):
	# calculate current alpha (linear from 0 to 1)
	alpha = step / float(n_steps - 1)
	# update the alpha for each model
	for model in models:
		for layer in model.layers:
			if isinstance(layer, WeightedSum):
				backend.set_value(layer.alpha, alpha)

"""
Training:
    a single training iteration involves first selecting a half batch
    of real images from the dataset and generating a half batch of fake 
    images from the current state of the generator model. Generator is 
    updated based on the result of the discriminator
"""

# train a generator and discriminator
# this will be called within training loop
def train_epochs(g_model, d_model, gan_model, dataset, n_epochs, n_batch, fadein=False):
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_steps):
		# update alpha for all WeightedSum layers when fading in new blocks
		if fadein:
			update_fadein([g_model, d_model, gan_model], i, n_steps)
		# prepare real and fake samples
		X_real, y_real = generate_real_samples(dataset, half_batch)
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator model
		d_loss1 = d_model.train_on_batch(X_real, y_real)
		d_loss2 = d_model.train_on_batch(X_fake, y_fake)
		# update the generator via the discriminator's error
		z_input = generate_latent_points(latent_dim, n_batch)
		y_real2 = ones((n_batch, 1))
		g_loss = gan_model.train_on_batch(z_input, y_real2)
		# summarize loss on this batch
		print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, d_loss1, d_loss2, g_loss))

# scale images to preferred size
def scale_dataset(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


# generate samples and save as a plot and save the model
def summarize_performance(status, g_model, latent_dim, n_samples=25):
	# devise name
	gen_shape = g_model.output_shape
	name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
	# generate images
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# normalize pixel values to the range [0,1]
	X = (X - X.min()) / (X.max() - X.min())
	# plot real images
	square = int(sqrt(n_samples))
	for i in range(n_samples):
		pyplot.subplot(square, square, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X[i])
	# save plot to file
	filename1 = 'plot_%s.png' % (name)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%s.h5' % (name)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))


# train the generator and discriminator
def train(g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, n_batch):
	# fit the baseline model
	g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]
	# scale dataset to appropriate size
	gen_shape = g_normal.output_shape
	scaled_data = scale_dataset(dataset, gen_shape[1:])
	print('Scaled Data', scaled_data.shape)
	# train normal or straight-through models
	train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[0], n_batch[0])
	summarize_performance('tuned', g_normal, latent_dim)
	# process each level of growth
	for i in range(1, len(g_models)):
		# retrieve models for this level of growth
		[g_normal, g_fadein] = g_models[i]
		[d_normal, d_fadein] = d_models[i]
		[gan_normal, gan_fadein] = gan_models[i]
		# scale dataset to appropriate size
		gen_shape = g_normal.output_shape
		scaled_data = scale_dataset(dataset, gen_shape[1:])
		print('Scaled Data', scaled_data.shape)
		# train fade-in models for next level of growth
		train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, e_fadein[i], n_batch[i], True)
		summarize_performance('faded', g_fadein, latent_dim)
		# train normal or straight-through models
		train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[i], n_batch[i])
		summarize_performance('tuned', g_normal, latent_dim)

### TRAIN ###
n_blocks = 6
# size of the latent space
latent_dim = 100
# define models
d_models = define_discriminator(n_blocks)
# define models
g_models = define_generator(latent_dim, n_blocks)
# define composite models
gan_models = define_composite(d_models, g_models)
# load image data
dataset = load_real_samples('img_align_celeba_128.npz')
print('Loaded', dataset.shape)
# train model
n_batch = [16, 16, 16, 8, 4, 4]
# 10 epochs == 500K images per training phase
n_epochs = [5, 8, 8, 10, 10, 10]
train(g_models, d_models, gan_models, dataset, latent_dim, n_epochs, n_epochs, n_ba