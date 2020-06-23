import tensorflow as tf
import load_EuroSat as lE
import os
import numpy as np


euro_path = r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\EuroSat"


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
indexes = []
while True:
    try:
        batch = train_data.get_train_batch()
        for i in batch[1]:
            if i[0] not in indexes:
                indexes.append(i)
    except:
        break

