import numpy as np
import pandas as pd
from processing.preprocess import get_train_data
import tensorflow as tf
import datetime
print("tf version:",tf.__version__)
print(tf.test.gpu_device_name())

from tensorflow import keras
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Conv2D, LSTM, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import MeanSquaredError, mae, CosineSimilarity
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.utils import shuffle
import wandb
from wandb.keras import WandbCallback

wandb.login()

TARGET_MONTHS = [6,7,8,9]
TARGET_FEATURE = 'rf'
HORIZON = 35 #56 days
LOOKBACK = 0
BATCH_SIZE = 32
MODEL_NAME = f"{TARGET_FEATURE}-{HORIZON}-{LOOKBACK}"
spatial_features=['rf']
# temporal_features=['phase_cos', 'phase_sin', 'mei', 'iod', 'amplitude']
temporal_features=[]

train, validation, test, lat_lon_grid, target_shape = get_train_data(target_months=TARGET_MONTHS, target_feature=TARGET_FEATURE, 
        horizon=HORIZON, lookback=LOOKBACK, spatial_features=spatial_features, temporal_features=temporal_features, years=(1979, 2011, 2016, 2020))

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, spatial, temporal, target, lookback, horizon, lat_lon_grid, target_shape, batch_size, shuffle=True):
        'Initialization'
        self.spatial = spatial
        self.temporal = temporal
        self.target = target
        self.target_dates = target.index.unique()
        self.lookback = lookback
        self.horizon = horizon
        self.target_shape = target_shape
        self.spatial_dim = (lookback+1, *lat_lon_grid, len(self.spatial.columns))
        self.temporal_dim = (lookback+1, len(self.temporal.columns))
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
           self.target_dates = shuffle(self.target_dates)

    def __data_generation(self, batch_target_dates):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        spatial_input = np.empty((self.batch_size, *self.spatial_dim))
        temporal_input = np.empty((self.batch_size, *self.temporal_dim))
        y = np.empty((self.batch_size, self.target_shape))

        # Generate data
        for i, date in enumerate(batch_target_dates):
            # Store sample
            start_date = date-pd.DateOffset(days=self.horizon+self.lookback)

            # Pandas slicing is inclusive [start, end]
            end_date = start_date+pd.DateOffset(days=self.lookback)

            spatial_input[i,] = self.spatial.loc[start_date:end_date].to_numpy().reshape(*self.spatial_dim)
            temporal_input[i,] = self.temporal.loc[start_date:end_date].to_numpy().reshape(*self.temporal_dim)
            # Store ground truth
            y[i] = self.target.loc[date].to_numpy()

        return {'spatial_input': spatial_input, 'temporal_input': temporal_input}, y

def build_model(lookback, lat_lon_grid, batch_size, spatial_features, temporal_features, target_shape, model_name, print_summary=True):
    spatial_input = Input(shape=[lookback+1, *lat_lon_grid, len(spatial_features)], batch_size=batch_size, name='spatial_input')
    temporal_input = Input(shape=[lookback+1, len(temporal_features)], batch_size=batch_size, name='temporal_input')
    # cnn = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), name='conv2d', activation='relu')(spatial_input)
    spatial_flatten = Flatten(name='spatial_flatten')(spatial_input)
    temporal_flatten = Flatten(name='temporal_flatten')(temporal_input)

    # cnn = Reshape(target_shape=(lookback+1, -1), name='spatial_reshape')(cnn)
    # # LSTM Architecture
    # lstm = LSTM(units=64, return_sequences=True, name='lstm_1', activation='relu')(cnn)
    # lstm = LSTM(units=64, return_sequences=False, name='lstm_2', activation='relu')(lstm)
    concat = K.concatenate([spatial_flatten, temporal_flatten], axis=-1)
    fnn = Dense(units=1024, name='hidden_1', activation='relu')(concat)
    fnn = Dense(units=1024, name='hidden_2', activation='relu')(fnn)
    output = Dense(units=target_shape, name='target_output', activation='linear')(fnn)
    model = Model(inputs=[spatial_input, temporal_input], outputs=[output], name=model_name)

    if print_summary:
        print(model.summary())

    return (model)

def train_model():
    train_generator = DataGenerator(spatial=train['spatial'], temporal=train['temporal'], target=train['y'], lookback=LOOKBACK, horizon=HORIZON, lat_lon_grid=lat_lon_grid, target_shape=target_shape, batch_size=BATCH_SIZE)

    model = build_model(lookback=LOOKBACK, lat_lon_grid=lat_lon_grid, spatial_features=spatial_features, temporal_features=temporal_features, target_shape=target_shape, batch_size=BATCH_SIZE, model_name=MODEL_NAME)

    run = wandb.init(project='dissertation',
           config={
              "learning_rate": 0.001,
              "epochs": 50,
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

    log_dir = "models/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Set callbacks
    callbacks_list = [checkpoint, earlystopping, WandbCallback(), tensorboard_callback]

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=config.learning_rate), metrics=[mae, MeanSquaredError(), Huber(), CosineSimilarity(axis=1)])

    validation_generator = DataGenerator(spatial=validation['spatial'], temporal=validation['temporal'], target=validation['y'], lookback=LOOKBACK, horizon=HORIZON, lat_lon_grid=lat_lon_grid, target_shape=target_shape, batch_size=BATCH_SIZE)
    model.fit(train_generator, epochs=config.epochs, callbacks=callbacks_list, validation_data=validation_generator)

    return model

def get_test_generator():
    test_generator = DataGenerator(spatial=test['spatial'], temporal=test['temporal'], target=test['y'], lookback=LOOKBACK, horizon=HORIZON, lat_lon_grid=lat_lon_grid, target_shape=target_shape, batch_size=BATCH_SIZE)
    return test_generator

if __name__ == '__main__':
    model = train_model()
    test_generator = get_test_generator()
    evaluation = model.evaluate(test_generator)
    print(evaluation)