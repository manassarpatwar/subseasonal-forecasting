import numpy as np
import pandas as pd
from preprocess import get_train_data
from utils import Lookback, visualise_prediction, inverse, get_image, normalize, get_labelY, get_feature_name
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy

import tensorflow as tf
import datetime
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

os.environ['WANDB_DIR'] = os.path.join('/', 'fastdata', 'aca18mms')


wandb.login()


hyperparameters = {
    "target_months": [1,2,3,4,5,6,7,8,9,10,11,12],
    "target_feature": 'tmp2m',
    "horizon": 7,
    "lookback": {},
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 256,
    "patience": 10,
    "loss_function": "mse",
    "architecture": "Linear",
    "spatial_features": ['tmp2m'],
    "temporal_features": [],
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
            spatial_tensor = self.spatial.loc[dates].to_numpy()
            spatial_input[i,] = spatial_tensor.reshape(-1, *self.spatial_dim)[mask,:,:,:]
            
            if len(self.temporal.columns):
                temporal_tensor = self.temporal.loc[dates].to_numpy()
                temporal_input[i,] = temporal_tensor.reshape(-1, *self.temporal_dim)[mask,:]
            # Store ground truth
            y[i] = self.target.loc[date].to_numpy()

        return {'spatial_input': spatial_input, 'temporal_input': temporal_input}, y

if __name__ == '__main__':

    run = wandb.init(project='dissertation',
                    config=hyperparameters)

    config = wandb.config

    lookback = Lookback(**config['lookback'])
    MODEL_NAME = f"{config['target_feature']}-{config['horizon']}-{lookback}"
    wandb.run.name = f"{wandb.run.id}-{MODEL_NAME}-{config['architecture']}"

    spatial, temporal, target, spatial_grid_shape = get_train_data(target_months=config['target_months'], target_feature=config['target_feature'], 
        horizon=config['horizon'], lookback=lookback, spatial_features=config['spatial_features'], temporal_features=config['temporal_features'], split=[1979, 2011, 2016, 2020], dataset=config['dataset'])
        
    train_generator = DataGenerator(spatial=spatial, temporal=temporal, target=target['train'], lookback=lookback, horizon=config['horizon'], spatial_grid_shape=spatial_grid_shape, target_shape=target['shape'], batch_size=config['batch_size'])

    build_model = importlib.import_module(f"models.{config['architecture']}").build_model
    model = build_model(lookback=lookback, spatial_grid_shape=spatial_grid_shape, spatial_features=config['spatial_features'], temporal_features=config['temporal_features'], target_shape=target['shape'], model_name=MODEL_NAME)

    # Early stopping
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=config['patience'], restore_best_weights=True)  # val_loss

    # Automatically save latest best model to file
    filepath = os.path.join(wandb.run.dir, "model-best.h5")
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    # log_dir = os.path.join("models", "logs", config['architecture'])
    # tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Set callbacks
    callbacks_list = [earlystopping, checkpoint, WandbCallback()]

    model.compile(loss=config['loss_function'], optimizer=Adam(learning_rate=config['learning_rate']), metrics=[CosineSimilarity(axis=1)])

    validation_generator = DataGenerator(spatial=spatial, temporal=temporal, target=target['validation'], lookback=lookback, horizon=config['horizon'], spatial_grid_shape=spatial_grid_shape, target_shape=target['shape'], batch_size=config['batch_size'])
    model.fit(train_generator, epochs=config['epochs'], callbacks=callbacks_list, validation_data=validation_generator)

    test_generator = DataGenerator(spatial=spatial, temporal=temporal, target=target['test'], lookback=lookback, horizon=config['horizon'], spatial_grid_shape=spatial_grid_shape, target_shape=target['shape'], shuffle=False)
   
    predictions = model.predict(test_generator)
    ground_truth = test_generator.ground_truth

    temporal_cosine_similarity = np.diag(1-scipy.spatial.distance.cdist(predictions.T, ground_truth.T, 'cosine'))
    temporal_image = get_image(temporal_cosine_similarity, spatial_grid_shape, target['mask'])
    fig = plt.figure()
    ax = plt.axes()
    ax.axis('off')
    im = ax.imshow(temporal_image, vmin=-1.0, vmax=1.0, cmap='PiYG')
    bar = plt.colorbar(im, fraction=0.046, pad=0)
    plt.title(f"{config['architecture']} temporal similarity")
    temporal_image =  wandb.Image(plt)
    plt.clf()

    target_dates = test_generator.target_dates

    ground_truth = pd.melt(pd.DataFrame(ground_truth, index=target_dates, columns=target['locations']), ignore_index=False, value_name=config['target_feature'])
    predictions = pd.melt(pd.DataFrame(predictions, index=target_dates, columns=target['locations']), ignore_index=False, value_name=config['target_feature'])
    
    ground_truth['gt'] = (ground_truth*target['std'] + target['mean']).dropna(how='all')[config['target_feature']].to_numpy()
    ground_truth = ground_truth.rename(columns={config['target_feature']: "norm-gt"})

    predictions['pred'] = (predictions*target['std'] + target['mean']).dropna(how='all')[config['target_feature']].to_numpy()
    predictions = predictions.rename(columns={config['target_feature']: "norm-pred"})

    spatial_results = ground_truth.merge(predictions, left_on=['start_date', 'lat', 'lon'], right_on=['start_date', 'lat', 'lon'], how='inner')
    spatial_results = spatial_results.sort_values(by=['start_date', 'lat', 'lon'])            

    spatial_plots = []
    plt.figure(figsize=(15,5))
    for year in spatial_results.index.year.unique():
        temporal_plots = []
        for (lat, lon) in target['locations'][25::10]:
            spatial_result = spatial_results.loc[(spatial_results['lat'] == lat) & (spatial_results['lon'] == lon)]
            plt.plot(spatial_result.loc[str(year), 'gt'], label=f"gt")
            plt.plot(spatial_result.loc[str(year), 'pred'], label=f"pred")
            plt.ylabel(get_labelY(config['target_feature']))
            plt.legend()
            plt.title(f"{get_feature_name(config['target_feature']).capitalize()} prediction at {lat}°N {lon}°E in {year}")
            temporal_plots.append(wandb.Image(plt))
            plt.clf()
        wandb.log({f"temporal predictions in {year}": temporal_plots})

    fig = plt.figure()
    ax = plt.axes()
    ax.axis('off')

    monthly_dates = target_dates.to_period('M').unique().to_timestamp().shift(14, freq='D')
    for date in monthly_dates:
        result = spatial_results.loc[date]
        loss = result['gt']-result['pred']
        loss = loss.abs()/result['gt']
        im = ax.imshow(get_image(loss.to_numpy(), spatial_grid_shape, target['mask']), cmap='Reds')
        ax.axis('off')
        plt.title(f"{get_feature_name(config['target_feature']).capitalize()} prediction error on {date.date()}")
        bar = plt.colorbar(im, fraction=0.046, pad=0)
        spatial_plots.append(wandb.Image(plt))
        plt.cla()
        bar.remove()

    [test_loss, test_cosine_similarity] = model.evaluate(test_generator)
    print(test_loss, test_cosine_similarity)
    wandb.run.summary["run id"] = wandb.run.id
    wandb.run.summary["test loss"] = test_loss
    wandb.run.summary["test spatial cosine similarity"]= test_cosine_similarity

   

    wandb.log({"spatial predictions": spatial_plots,
                "temporal cosine similarity": temporal_image})
