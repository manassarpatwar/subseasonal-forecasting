# Author: Manas Saraptwar
# Date: 19/05/2021

from tensorflow import keras
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Conv2D, LSTM, Flatten, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def build_model(lookback, spatial_grid_shape, spatial_features, temporal_features, target_shape, model_name, print_summary=True):
    spatial_input = Input(shape=[lookback.length(), *spatial_grid_shape, len(spatial_features)], name='spatial_input')
    cnn = TimeDistributed(Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), name='conv2d', padding='same', activation='relu'), name='td-cnn-7x7')(spatial_input)
    cnn = TimeDistributed(Dropout(0.3))(cnn)
    cnn = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), name='conv2d', padding='same', activation='relu'), name='td-cnn-5x5')(cnn)
    cnn = TimeDistributed(Dropout(0.3))(cnn)
    cnn = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), name='conv2d', padding='same', activation='relu'), name='td-cnn-3x3')(cnn)
    cnn = TimeDistributed(Dropout(0.3))(cnn)
    spatial_flatten = TimeDistributed(Flatten(name='spatial_flatten'))(cnn)
    
    temporal_input = Input(shape=[lookback.length(), len(temporal_features)], name='temporal_input')
    temporal_flatten = TimeDistributed(Flatten(name='temporal_flatten'))(temporal_input)

    concat = K.concatenate([spatial_flatten, temporal_flatten], axis=-1) if temporal_features else spatial_flatten

    # LSTM Architecture
    lstm = LSTM(units=512, return_sequences=True, name='lstm_1', dropout=0.3, recurrent_dropout=0.3)(concat)
    lstm = LSTM(units=512, return_sequences=False, name='lstm_2', dropout=0.3, recurrent_dropout=0.3)(lstm)

    output = Dense(units=target_shape, name='target_output', activation='linear')(lstm)

    model = Model(inputs=[spatial_input, temporal_input], outputs=[output], name=model_name)

    if print_summary:
        print(model.summary())

    return (model)