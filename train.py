import numpy as np
import pandas as pd
from preprocess import get_train_data
from utils import Lookback, visualise_prediction, inverse, get_image, normalize, get_label, get_feature_name, dataset_data
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
from scipy.spatial.distance import cdist

import tensorflow as tf
print("tf version:",tf.__version__)
print(tf.test.gpu_device_name())

from tensorflow import keras
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Conv2D, LSTM, Flatten, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.utils import shuffle
import wandb
from wandb.keras import WandbCallback

import importlib

cache_dir = os.path.join('/', 'fastdata', 'aca18mms')
os.environ['WANDB_DIR'] = cache_dir
os.environ['WANDB_IGNORE_GLOBS'] = '*.h5'

wandb.login()


hyperparameters = {
    "target_months": [1,2,3,4,5,6,7,8,9,10,11,12],
    "target_feature": 'tmp2m',
    "horizon": 28,
    "lookback": {'past': [7, 14, 28], 'future': [7, 14, 28], 'years': 2},
    "learning_rate": 0.0005,
    "epochs": 25,
    "batch_size": 128,
    "patience": 10,
    "loss_function": "mse",
    "architecture": "CNNLSTM",
    "spatial_features": ['tmp2m', 'precip', 'rhum', 'slp', 'pres'],
    "temporal_features": ['phase', 'amplitude', 'phase_cos', 'phase_sin', 'mei'],
    "dataset": "RODEO"
}


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
            # Pandas datetime slicing is faster than datetime indexing
            spatial_tensor = self.spatial.loc[dates].to_numpy()
            # Applying numpy mask over temporal dimension is faster than pandas datetime indexing
            spatial_input[i,] = spatial_tensor.reshape(-1, *self.spatial_dim)[mask,:,:,:]
            
            if len(self.temporal.columns):
                # Pandas datetime slicing is faster than datetime indexing
                temporal_tensor = self.temporal.loc[dates].to_numpy()
                # Applying numpy mask over temporal dimension is faster than pandas datetime indexing
                temporal_input[i,] = temporal_tensor.reshape(-1, *self.temporal_dim)[mask,:]
            # Store ground truth and climatology
            y[i] = self.target.loc[date].to_numpy()

        return {'spatial_input': spatial_input, 'temporal_input': temporal_input}, y

if __name__ == '__main__':

    run = wandb.init(project='dissertation',
                    config=hyperparameters)

    config = wandb.config

    lookback = Lookback(**config['lookback'])
    MODEL_NAME = f"{config['target_feature']}-{config['horizon']}-{lookback}"
    wandb.run.name = f"{config['dataset']}-{MODEL_NAME}-{config['architecture']}"

    spatial, temporal, target, spatial_grid_shape = get_train_data(target_months=config['target_months'], target_feature=config['target_feature'], horizon=config['horizon'], lookback=lookback, spatial_features=config['spatial_features'], temporal_features=config['temporal_features'], dataset=config['dataset'])
    spatial = spatial[config['spatial_features']]

    train_generator = DataGenerator(spatial=spatial, temporal=temporal, target=target['train'], lookback=lookback, horizon=config['horizon'], spatial_grid_shape=spatial_grid_shape, target_shape=target['shape'], batch_size=config['batch_size'])

    build_model = importlib.import_module(f"models.{config['architecture']}").build_model
    model = build_model(lookback=lookback, spatial_grid_shape=spatial_grid_shape, spatial_features=config['spatial_features'], temporal_features=config['temporal_features'], target_shape=target['shape'], model_name=MODEL_NAME)

    # Early stopping
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=config['patience'], restore_best_weights=True)  # val_loss

    # Automatically save latest best model to file
    filepath = os.path.join(cache_dir, "runs", wandb.run.id, "model.h5")
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_cosine_similarity', verbose=0, save_best_only=True, mode='max')

    log_dir = os.path.join(cache_dir, "runs", wandb.run.id)
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Set callbacks
    callbacks_list = [checkpoint, earlystopping, WandbCallback(), tensorboard]

    model.compile(loss=config['loss_function'], optimizer=Adam(learning_rate=config['learning_rate']), metrics=[CosineSimilarity()])

    validation_generator = DataGenerator(spatial=spatial, temporal=temporal, target=target['validation'], lookback=lookback, horizon=config['horizon'], spatial_grid_shape=spatial_grid_shape, target_shape=target['shape'], batch_size=config['batch_size'])
    history = model.fit(train_generator, epochs=config['epochs'], callbacks=callbacks_list, validation_data=validation_generator)

    test_generator = DataGenerator(spatial=spatial, temporal=temporal, target=target['test'], lookback=lookback, horizon=config['horizon'], spatial_grid_shape=spatial_grid_shape, target_shape=target['shape'], shuffle=False)
    predictions = model.predict(test_generator)
    ground_truth = test_generator.ground_truth
    target_dates = test_generator.target_dates

    training_plots = []
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Training loss')
    plt.ylabel('Mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    training_plots.append(wandb.Image(plt))
    plt.clf()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['cosine_similarity'], label='train')
    plt.plot(history.history['val_cosine_similarity'], label='val')
    plt.title('Training spatial cosine similarity')
    plt.ylabel('Cosine similarity')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    training_plots.append(wandb.Image(plt))
    plt.clf()

    wandb.log({f"training plots": training_plots})


    plt.figure(figsize=(15, 5))
    for year in target_dates.year.unique():
        temporal_plots = []
        year_mask = target_dates.year == year
        xlabels = target_dates[year_mask]
        locs = dataset_data(config['dataset'])['visualize_locations']
        for ix, state, (lat, lon) in zip(target['locations'].isin(locs.values()).nonzero()[0], locs.keys(), locs.values()):
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.tick_params(axis="x", which="major", pad=12)
            
            plt.plot(xlabels, ground_truth[year_mask, ix].T, label="gt")
            plt.plot(xlabels, predictions[year_mask, ix].T, label="pred")
         
            plt.ylabel(get_label(config['target_feature']))
            plt.legend()
            plt.title(f"{get_feature_name(config['target_feature']).capitalize()} anomaly in {state} ({lat}°N {lon}°E) in {year}")
            temporal_plots.append(wandb.Image(plt))
            plt.clf()
        wandb.log({f"temporal predictions in {year}": temporal_plots})

    spatial_plots = []
    fig = plt.figure()
    ax = plt.axes()
    ax.axis('off')

    for ix, date in enumerate(target_dates):
        loss = (ground_truth[ix, :]-predictions[ix, :])/((np.abs(ground_truth[ix, :]) + np.abs(predictions[ix, :]))/2)
        im = ax.imshow(get_image(loss, spatial_grid_shape, target['mask']), cmap='coolwarm')
        ax.axis('off')
        plt.title(f"{config['target_feature']} anomaly prediction relative error on {date.date()}")
        bar = plt.colorbar(im, fraction=0.046, pad=0.04)
        spatial_plots.append(wandb.Image(plt))
        plt.cla()
        bar.remove()

    temporal_cosine_similarity = np.diag(1-cdist(predictions.T, ground_truth.T, 'cosine'))
    temporal_image = get_image(temporal_cosine_similarity, spatial_grid_shape, target['mask'])
    fig = plt.figure()
    ax = plt.axes()
    ax.axis('off')
    im = ax.imshow(temporal_image, vmin=-0.5, vmax=0.5, cmap='PiYG')
    bar = plt.colorbar(im, fraction=0.046, pad=0)
    plt.title(f"{config['architecture']} temporal similarity")
    temporal_image =  wandb.Image(plt)
    plt.clf()

    [test_loss, test_cosine_similarity] = model.evaluate(test_generator)

    os.remove(os.path.join(cache_dir, "wandb", wandb.run.dir, "model-best.h5"))

    wandb.run.summary["run id"] = wandb.run.id
    wandb.run.summary["test_loss"] = test_loss
    wandb.run.summary["test_cosine_similarity"] = test_cosine_similarity
    wandb.log({"spatial relative error": spatial_plots,
                "temporal cosine similarity": temporal_image})
