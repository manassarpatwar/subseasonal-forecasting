import numpy as np
import pandas as pd
from processing.preprocess import get_train_data
import tensorflow as tf
print("tf version:",tf.__version__)
print(tf.test.gpu_device_name())

from tensorflow import keras
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Conv2D, LSTM, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import RootMeanSquaredError, mae
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
import wandb
from wandb.keras import WandbCallback

wandb.login()

TARGET_MONTHS = [6,7,8,9]
TARGET_FEATURE = 'rf'
HORIZON = 56 #56 days
LOOKBACK = 26
BATCH_SIZE = 32
MODEL_NAME = f"{TARGET_FEATURE}-{HORIZON}-{LOOKBACK}"
features=['rf', 'temp', 'pres', 'slp', 'rhum']

train_years = [year for year in range(1979, 2018)]
test_years = [year for year in range(train_years[-1], 2020)]

train, target, train_grid, target_shape = get_train_data(target_months=TARGET_MONTHS, target_feature=TARGET_FEATURE, 
                                horizon=HORIZON, lookback=LOOKBACK, features=features)

X_train, y_train = train[train.index.year.isin(train_years)], target[target.index.year.isin(train_years)]
X_test, y_test = train[train.index.year.isin(test_years)], target[target.index.year.isin(test_years)]

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train, target, lookback, horizon, train_grid, target_shape, batch_size, shuffle=True):
        'Initialization'
        self.train = train
        self.target = target
        self.target_dates = target.index.unique()
        self.lookback = lookback
        self.horizon = horizon
        self.target_shape = target_shape
        self.dim = (lookback, *train_grid, len(self.train.columns))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

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
            shuffle(self.target_dates)

    def __data_generation(self, batch_target_dates):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.target_shape))

        # Generate data
        for i, date in enumerate(batch_target_dates):
            # Store sample
            start_date = date-pd.Timedelta(self.horizon+self.lookback, unit='days')
            end_date = start_date+pd.Timedelta(self.lookback-1, unit='days')

            X[i,] = self.train.loc[start_date:end_date].to_numpy().reshape(*self.dim)

            # Store ground truth
            y[i] = self.target.loc[date].to_numpy()

        return X, y

def build_model(lookback, train_grid, batch_size, features, target_shape, model_name, print_summary=True):
    spatial_input = Input(shape=[lookback, *train_grid, len(features)], batch_size=batch_size, name='spatial_input')
    model = TimeDistributed(Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='same', name='conv2d'), name='td_conv2d')(spatial_input)
    model = Flatten(name='spatial_flatten')(model)
    model = Reshape(target_shape=(lookback, -1), name='spatial_reshape')(model)
    model = TimeDistributed(Dense(units=24, activation='relu', name='conv2d_out'), name='td_conv2d_out')(model)
     # LSTM Architecture
    lstm_model = LSTM(units=64, return_sequences=True, name='lstm_1')(model)
    lstm_model = LSTM(units=64, return_sequences=False, name='lstm_2')(lstm_model)
    model_output = Dense(units=target_shape, name='target_output')(lstm_model)
    model = Model(inputs=[spatial_input], outputs=[model_output], name=model_name)

    if print_summary:
        print(model.summary())

    return (model)

if __name__ == '__main__':
    
    train_generator = DataGenerator(train=X_train, target=y_train, lookback=LOOKBACK, horizon=HORIZON, train_grid=train_grid, target_shape=target_shape, batch_size=BATCH_SIZE)

    model = build_model(lookback=LOOKBACK, train_grid=train_grid, features=features, target_shape=target_shape, batch_size=BATCH_SIZE, model_name=MODEL_NAME)

    run = wandb.init(project='dissertation',
           config={
              "learning_rate": 0.001,
              "epochs": 25,
              "batch_size": BATCH_SIZE,
              "patience": 10,
              "loss_function": "mean-absolute-error",
              "architecture": "CNN-LSTM",
              "dataset": "IMD-IMDAA"
           })
    config = wandb.config

      ## Early stopping
    earlystopping = EarlyStopping(monitor='loss', min_delta=0.00001, patience=config.patience, restore_best_weights=True)  # val_loss

    # Automatically save latest best model to file
    filepath = f"models/model_saves/{TARGET_FEATURE}.hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')

    # Set callbacks
    callbacks_list = [checkpoint, earlystopping, WandbCallback()]

    model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=config.learning_rate), metrics=[mae, RootMeanSquaredError(), Huber()])

    model.fit(train_generator, epochs=config.epochs, use_multiprocessing=True, verbose=1, callbacks=callbacks_list)

    test_generator = DataGenerator(train=X_test, target=y_test, lookback=LOOKBACK, horizon=HORIZON, train_grid=train_grid, target_shape=target_shape, batch_size=BATCH_SIZE)

    evaluation = model.evaluate(test_generator)
    print(evaluation)