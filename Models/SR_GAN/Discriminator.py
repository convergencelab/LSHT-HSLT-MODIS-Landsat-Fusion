from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense, PReLU, Flatten, Conv2D, UpSampling2D
from tensorflow.keras.models import Model


class Discriminator(Model):
    def __init__(self,
                 momentum=0.8,
                 leakyrelu_alpha=0.2,
                 **kwargs
                 ):
        # call the parent constructor
        super(Discriminator, self).__init__(**kwargs)

        ###############
        # HyperParams #
        ###############
        self.momentum = momentum
        self.leakyrelu_alpha = leakyrelu_alpha

        #################
        # Discriminator #
        #################
        self.conv2f = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.activation4 = LeakyReLU(alpha=self.leakyrelu_alpha)

        # convolution layers #
        # Add the 2nd convolution block
        self.conv2g = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')
        self.activation5 = LeakyReLU(alpha=leakyrelu_alpha)
        self.bn2 = BatchNormalization(momentum=momentum)

        # Add the third convolution block
        self.conv2h = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
        self.activation6 = LeakyReLU(alpha=leakyrelu_alpha)
        self.bn3 = BatchNormalization(momentum=momentum)

        # Add the fourth convolution block
        self.conv2i = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')
        self.activation7 = LeakyReLU(alpha=leakyrelu_alpha)
        self.bn4 = BatchNormalization(momentum=0.8)

        # Add the fifth convolution block
        self.conv2j = Conv2D(256, kernel_size=3, strides=1, padding='same')
        self.activation8 = LeakyReLU(alpha=leakyrelu_alpha)
        self.bn5 = BatchNormalization(momentum=momentum)

        # Add the sixth convolution block
        self.conv2k = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')
        self.activation9 = LeakyReLU(alpha=leakyrelu_alpha)
        self.bn6 = BatchNormalization(momentum=momentum)

        # Add the seventh convolution block
        self.conv2l = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')
        self.activation10 = LeakyReLU(alpha=leakyrelu_alpha)
        self.bn7 = BatchNormalization(momentum=momentum)

        # Add the eight convolution block
        self.conv2m = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')
        self.activation11 = LeakyReLU(alpha=leakyrelu_alpha)
        self.bn8 = BatchNormalization(momentum=momentum)

        # dense layer #
        self.dense1 = Dense(units=1024)
        self.activation12 = LeakyReLU(alpha=leakyrelu_alpha)

        # output probability with sigmoid function #
        self.Dense2 = Dense(units=1, activation='sigmoid')

    def call(self, inputs):
        #################
        # Discriminator #
        #################

        # input generators output #
        x = self.conv2f(inputs)
        x = self.activation4(x)

        # convolution layers #
        # Add the 2nd convolution block
        x = self.conv2g(x)
        x = self.activation5(x)
        x = self.bn2(x)

        # Add the third convolution block
        x = self.conv2h(x)
        x = self.activation6(x)
        x = self.bn3(x)

        # Add the fourth convolution block
        x = self.conv2i(x)
        x = self.activation7(x)
        x = self.bn4(x)

        # Add the fifth convolution block
        x = self.conv2j(x)
        x = self.activation8(x)
        x = self.bn5(x)

        # Add the sixth convolution block
        x = self.conv2k(x)
        x = self.activation9(x)
        x = self.bn6(x)

        # Add the seventh convolution block
        x = self.conv2l(x)
        x = self.activation10(x)
        x = self.bn7(x)

        # Add the eight convolution block
        x = self.conv2m(x)
        x = self.activation11(x)
        x = self.bn8(x)

        # dense layer #
        x = self.dense1(x)
        x = self.activation12(x)

        return self.Dense2(x)