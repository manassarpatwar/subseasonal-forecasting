"""
Title: Timeseries forecasting for weather prediction
Authors: [Prabhanshu Attri](https://prabhanshu.com/github), [Yashika Sharma](https://github.com/yashika51), [Kristi Takach](https://github.com/ktakattack), [Falak Shah](https://github.com/falaktheoptimist)
Date created: 2020/06/23
Last modified: 2020/07/20
Description: This notebook demonstrates how to do timeseries forecasting using a LSTM model.
"""

"""
## Setup
This example requires TensorFlow 2.3 or higher.
"""

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

"""
## Climate Data Time-Series
We will be using Jena Climate dataset recorded by the
[Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/).
The dataset consists of 14 features such as temperature, pressure, humidity etc, recorded once per
10 minutes.
**Location**: Weather Station, Max Planck Institute for Biogeochemistry
in Jena, Germany
**Time-frame Considered**: Jan 10, 2009 - December 31, 2016
The table below shows the column names, their value formats, and their description.
Index| Features      |Format             |Description
-----|---------------|-------------------|-----------------------
1    |Date Time      |01.01.2009 00:10:00|Date-time reference
2    |p (mbar)       |996.52             |The pascal SI derived unit of pressure used to quantify internal pressure. Meteorological reports typically state atmospheric pressure in millibars.
3    |T (degC)       |-8.02              |Temperature in Celsius
4    |Tpot (K)       |265.4              |Temperature in Kelvin
5    |Tdew (degC)    |-8.9               |Temperature in Celsius relative to humidity. Dew Point is a measure of the absolute amount of water in the air, the DP is the temperature at which the air cannot hold all the moisture in it and water condenses.
6    |rh (%)         |93.3               |Relative Humidity is a measure of how saturated the air is with water vapor, the %RH determines the amount of water contained within collection objects.
7    |VPmax (mbar)   |3.33               |Saturation vapor pressure
8    |VPact (mbar)   |3.11               |Vapor pressure
9    |VPdef (mbar)   |0.22               |Vapor pressure deficit
10   |sh (g/kg)      |1.94               |Specific humidity
11   |H2OC (mmol/mol)|3.12               |Water vapor concentration
12   |rho (g/m ** 3) |1307.75            |Airtight
13   |wv (m/s)       |1.03               |Wind speed
14   |max. wv (m/s)  |1.75               |Maximum wind speed
15   |wd (deg)       |152.3              |Wind direction in degrees
"""

from zipfile import ZipFile
import os
from preprocess import *
from utils import Lookback

norm = True

# df = pd.read_hdf('data/rodeo/gt-contest_tmp2m-14d-1979-2018.h5')
# df = df.set_index('start_date')
# df = df.sort_values(['start_date', 'lat', 'lon'])
# df = get_rodeo_spatial_dataframe()
TARGET_FEATURE = 'tmp2m'
df, _, _, _ = get_train_data(target_months=[1,2,3,4,5,6,7,8,9,10,11,12], horizon=7, lookback=Lookback(past=0), spatial_features=[TARGET_FEATURE], target_feature=TARGET_FEATURE, split=[1979, 2020, 2020,2020], dataset='RODEO', temporal_features=[])
norm = False

# df = pd.read_hdf('data/dataframes/temp-14d-avg.h5')
# TARGET_FEATURE = 'temp'

(lat, lon) = df.set_index(['lat', 'lon']).dropna().sample(1).index[0]
print(lat, lon)
df = df.loc[(df['lat'] == 41.0) & (df['lon'] == 241.0)]

split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
step = 1

past = 1
future = 36
learning_rate = 0.001
batch_size = 256
epochs = 50


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


"""
We can see from the correlation heatmap, few parameters like Relative Humidity and
Specific Humidity are redundant. Hence we will be using select features, not all.
"""

selected_features = [TARGET_FEATURE]
features = df[selected_features]
features.head()

if norm:
    features = normalize(features, train_split)
features.head()

train_data = features.iloc[0 : train_split - 1]
val_data = features.iloc[train_split:]

"""
# Training dataset
The training dataset labels starts from the 792nd observation (720 + 72).
"""

start = past + future
end = start + train_split

x_train = train_data.to_numpy()
y_train = features[TARGET_FEATURE].iloc[start:end-1]

sequence_length = int(past / step)

"""
The `timeseries_dataset_from_array` function takes in a sequence of data-points gathered at
equal intervals, along with time series parameters such as length of the
sequences/windows, spacing between two sequence/windows, etc., to produce batches of
sub-timeseries inputs and targets sampled from the main timeseries.
"""

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

"""
## Validation dataset
The validation dataset must not contain the last 792 rows as we won't have label data for
those records, hence 792 must be subtracted from the end of the data.
The validation label dataset must start from 792 after train_split, hence we must add
past + future (792) to label_start.
"""

x_end = len(val_data) - past - future

label_start = train_split + past + future

x_val = val_data.iloc[:x_end][selected_features]
y_val = features[TARGET_FEATURE].iloc[label_start:]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)

"""
## Training
"""

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=['cosine_similarity'])
model.summary()

"""
We'll use the `ModelCheckpoint` callback to regularly save checkpoints, and
the `EarlyStopping` callback to interrupt training when the validation loss
is not longer improving.
"""

path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)

"""
We can visualize the loss with the function below. After one point, the loss stops
decreasing.
"""


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(history, "Training and Validation Loss")