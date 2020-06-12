import numpy as np
import tensorflow as tf
import glob

class NPYDataGenerator(tf.keras.utils.Sequence):
    """
    Keras datagenerator for .npy datasets

    dir structure is assumed to contain all .npy files
    .npy files assumed to store X, y in file, independently
    """
    def __init__(self, file_dir, labels,dim=(256, 256), n_channels=3,  batch_size=1,
                  shuffle=True):
        # call the parent constructor
        super(NPYDataGenerator, self).__init__()

        self.batch_size = batch_size
        self.labels = labels
        self.files = glob.glob(file_dir+"\*")
        self.shuffle = shuffle
        self.on_epoch_end()
        self.dim = dim
        self.n_channels = n_channels

    def on_epoch_end(self):
      # list of indexes reffering to indexes for files #
      # self.indexes = np.arange(len(self.files))
      # shuffle after every epoch #
      if self.shuffle == True:
          np.random.shuffle(self.files)

    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      files = self.files[index * self.batch_size:(index + 1) * self.batch_size]

      # Generate data
      X, y = self._datagen(files)

      return X, y

    def _datagen(self, files):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = np.empty((self.batch_size, *self.dim, self.n_channels))
      y = np.empty((self.batch_size, len(self.labels)))
      # Generate data
      for i, f in enumerate(files):
          ### each file is stored with val then label ###
          # Store sample
          with open(f, 'rb') as file:

              X[i, ] = np.load(file)[0]
              # Store class

             
              y[i] = np.load(file)[0]

      return X, y

class NPYDataGeneratorSR(NPYDataGenerator):
    def __init__(self, file_dir, labels=False, dim=(256, 256), lr_dim=(64,64), n_channels=3,  batch_size=3,
                  shuffle=True):
        # call the parent constructor
        super(NPYDataGeneratorSR, self).__init__(file_dir=file_dir, labels=False)
        self.lr_dim = lr_dim
    def _datagen(self, files):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.lr_dim, self.n_channels))
        # Generate data
        for i, f in enumerate(files):
            ### each file is stored with val then label ###
            # Store sample
            with open(f, 'rb') as file:
                X[i,] = np.load(file)
                # Store class

                y[i] = np.load(file)

        return X, y