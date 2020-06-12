from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense, PReLU, Flatten, Conv2D, UpSampling2D
from tensorflow.keras.models import Model
from ResLayer import ResLayer

class Generator(Model):
    def __init__(self,
                 residual_blocks=16,
                 momentum=0.8,
                 **kwargs
                 ):
        # call the parent constructor
        super(Generator, self).__init__(**kwargs)

        ###############
        # HyperParams #
        ###############
        self.residual_blocks = residual_blocks
        self.momentum = momentum

        #############
        # Generator #
        #############
        # feed in layer
        self.conv2a = Conv2D(filters=64,
                             kernel_size=9,
                             strides=1,
                             padding='same',
                             activation='relu')

        # res blocks #
        self.res1 = ResLayer(kernel_size=3,
                             filters=[64, 64],
                             strides=1,
                             momentum=momentum)
        self.resblocks = []
        for i in range(residual_blocks - 1):
            self.resblocks.append(ResLayer(kernel_size=3,
                                           filters=[64, 64],
                                           strides=1,
                                           momentum=momentum))

        # post res blocks #
        self.con2b = Conv2D(filters=64,
                            kernel_size=3,
                            strides=1,
                            padding='same')

        self.bn1 = BatchNormalization(momentum=momentum)
        self.upspl1 = UpSampling2D(size=2)
        self.conv2c = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.activation1 = Activation('relu')

        self.upspl2 = UpSampling2D(size=2)
        self.conv2d = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.activation2 = Activation('relu')

        # output uses tanh as output activation #
        self.conv2e = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')
        self.activation3 = Activation('tanh')

    def call(self, inputs):

        #############
        # Generator #
        #############
        gen1 = self.conv2a(inputs)
        x = self.res1(gen1)

        # pass through all resblocks #
        for r in self.resblocks:
            x = r(x)

        # post res blocks #
        x = self.con2b(x)
        gen2 = self.bn1(x)

        # take the sum of the output from the pre-residual block,
        # and the output from the post-residual block
        x = Add()([gen2, gen1])

        # upsample 1
        x = self.upspl1(x)
        x = self.conv2c(x)
        x = self.activation1(x)

        # upsample 2
        x = self.upspl2(x)
        x = self.conv2d(x)
        x = self.activation2(x)

        # output
        x = self.conv2e(x)
        return self.activation3(x)