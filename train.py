import numpy as np
import pandas as pd
from preprocess import get_train_data
from utils import Lookback, visualise_prediction
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import datetime
print("tf version:",tf.__version__)
print(tf.test.gpu_device_name())

from tensorflow import keras
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Conv2D, LSTM, Flatten, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.utils import shuffle
import wandb
from wandb.keras import WandbCallback

from models.LSTM import build_model
wandb.login()

TARGET_MONTHS = [1,2,3,4,5,6,7,8,9,10,11,12]
TARGET_FEATURE = 'temp'
HORIZON = 21 #21 days
# LOOKBACK = Lookback(past=14)
# LOOKBACK = Lookback(past=[7, 14, 28], future=[7, 14, 28], years=2)
LOOKBACK = Lookback(past=28)
BATCH_SIZE = 256
SPATIAL_FEATURES=['rf','temp', 'pres', 'slp', 'rhum']
# SPATIAL_FEATURES=[TARGET_FEATURE]
# TEMPORAL_FEATURES=['phase_cos', 'phase_sin', 'mei', 'iod', 'amplitude']
TEMPORAL_FEATURES=[]
MODEL_NAME = f"{TARGET_FEATURE}-{HORIZON}-{LOOKBACK}"

AVERAGE = True
spatial, temporal, target, spatial_grid_shape = get_train_data(target_months=TARGET_MONTHS, target_feature=TARGET_FEATURE, 
        horizon=HORIZON, lookback=LOOKBACK, spatial_features=SPATIAL_FEATURES, temporal_features=TEMPORAL_FEATURES, split=(1979, 2011, 2016, 2020), average=AVERAGE)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, spatial, temporal, target, lookback, horizon, spatial_grid_shape, target_shape, batch_size=1, shuffle=True):
        'Initialization'
        self.spatial = spatial
        self.temporal = temporal
        self.target = target
        self.target_dates = target.index.unique()
        self.lookback = lookback
        self.horizon = horizon
        self.target_shape = target_shape
        self.spatial_dim = (*spatial_grid_shape, len(self.spatial.columns))
        self.temporal_dim = (len(self.temporal.columns),)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    @property
    def ground_truth(self):
        return self.target.loc[self.target_dates].to_numpy().reshape(-1, self.target_shape)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.target_dates) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_target_dates = self.target_dates[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(batch_target_dates)

        return X, y

    def on_epoch_end(self):
        'Shuffle target_dates after each epoch if shuffle is set to True'
        if self.shuffle == True:
           self.target_dates = shuffle(self.target_dates)

    def __data_generation(self, batch_target_dates):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        spatial_input = np.empty((self.batch_size, self.lookback.length(), *self.spatial_dim))
        temporal_input = np.empty((self.batch_size, self.lookback.length(), *self.temporal_dim))
        y = np.empty((self.batch_size, self.target_shape))

        # Generate data
        for i, date in enumerate(batch_target_dates):
            # Store sample
            dates, mask = self.lookback.dates(date=date, horizon=self.horizon)
            spatial_tensor = self.spatial.loc[dates].to_numpy()
            spatial_input[i,] = spatial_tensor.reshape(-1, *self.spatial_dim)[mask,:,:,:]
            
            if len(self.temporal.columns):
                temporal_tensor = self.temporal.loc[dates].to_numpy()
                temporal_input[i,] = temporal_tensor.reshape(-1, *self.temporal_dim)[mask,:,:]
            # Store ground truth
            y[i] = self.target.loc[[date]].to_numpy()

        return {'spatial_input': spatial_input, 'temporal_input': temporal_input}, y

if __name__ == '__main__':
    train_generator = DataGenerator(spatial=spatial, temporal=temporal, target=target['train'], lookback=LOOKBACK, horizon=HORIZON, spatial_grid_shape=spatial_grid_shape, target_shape=target['shape'], batch_size=BATCH_SIZE)

    model, architechture = build_model(lookback=LOOKBACK, spatial_grid_shape=spatial_grid_shape, spatial_features=SPATIAL_FEATURES, temporal_features=TEMPORAL_FEATURES, target_shape=target['shape'], batch_size=BATCH_SIZE, model_name=MODEL_NAME)

    run = wandb.init(project='subseasonal-forecasting',
                    name=f"{MODEL_NAME}-{architechture}{'-avg' if AVERAGE else ''}",
                    config={
                        "learning_rate": 0.001,
                        "epochs": 10,
                        "batch_size": BATCH_SIZE,
                        "patience": 10,
                        "loss_function": "mse",
                        "architecture": architechture,
                        "dataset": "IMD-IMDAA",
                        "spatial": SPATIAL_FEATURES,
                        "temporal": TEMPORAL_FEATURES,
                        "target months": TARGET_MONTHS,
                        "average": AVERAGE
                    })
    config = wandb.config

    ## Early stopping
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=config.patience, restore_best_weights=True)  # val_loss

    # Automatically save latest best model to file
    filepath = os.path.join("models", "model_saves", architechture, f"{MODEL_NAME}.h5")
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    log_dir = os.path.join("models", "logs", architechture)
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Set callbacks
    callbacks_list = [checkpoint, earlystopping, WandbCallback(), tensorboard]

    model.compile(loss=config.loss_function, optimizer=Adam(learning_rate=config.learning_rate), metrics=["cosine_similarity"])

    validation_generator = DataGenerator(spatial=spatial, temporal=temporal, target=target['validation'], lookback=LOOKBACK, horizon=HORIZON, spatial_grid_shape=spatial_grid_shape, target_shape=target['shape'], batch_size=BATCH_SIZE)
    model.fit(train_generator, epochs=config.epochs, callbacks=callbacks_list, validation_data=validation_generator)

    test_generator = DataGenerator(spatial=spatial, temporal=temporal, target=target['test'], lookback=LOOKBACK, horizon=HORIZON, spatial_grid_shape=spatial_grid_shape, target_shape=target['shape'], batch_size=BATCH_SIZE, shuffle=False)
   
    predictions = model.predict(test_generator)
    
    plots = []
    # fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    plt.figure(figsize=(15,5))

    for prediction, ground_truth, date in zip(predictions[::21], test_generator.ground_truth, test_generator.target_dates):
        # p, g = visualise_prediction(prediction, ground_truth, spatial_grid_shape, target['mask'])
        # vmin = min(p.min(), g.min())
        # vmax = max(p.max(), g.max())

        # ax[0].imshow(p, vmin=vmin, vmax=vmax)
        # ax[0].set_title(f"{TARGET_FEATURE}-pred {date.date()}")

        # im = ax[1].imshow(g, vmin=vmin, vmax=vmax)
        # ax[1].set_title(f"{TARGET_FEATURE}-gt {date.date()}")
       
        # bar = fig.colorbar(im, ax=ax)

        # images.append(wandb.Image(plt))
        
        # bar.remove()
        # plt.cla()
        plt.plot(prediction, label=f"{TARGET_FEATURE}-pred {date.date()}")
        plt.plot(ground_truth, label=f"{TARGET_FEATURE}-gt {date.date()}")
        plt.legend()
        plt.xlabel('Target locations')
        plots.append(wandb.Image(plt))
        plt.clf()

    wandb.log({f"spatio-temporal predictions": plots})
    evaluation = model.evaluate(test_generator)
    print(evaluation)