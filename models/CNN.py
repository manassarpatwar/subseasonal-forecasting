from tensorflow import keras
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Conv2D, LSTM, Flatten
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def build_model(lookback, spatial_grid_shape, batch_size, spatial_features, temporal_features, target_shape, model_name, print_summary=True):
    spatial_input = Input(shape=[lookback.length(), *spatial_grid_shape, len(spatial_features)], batch_size=batch_size, name='spatial_input')
    cnn = TimeDistributed(Conv2D(filters=16, kernel_size=(5, 5), strides=(3, 3), name='conv2d'), name='td_cnn'), name='td_cnn')(spatial_input)
    spatial_flatten = TimeDistributed(Flatten(name='spatial_flatten'))(cnn)
    
    temporal_input = Input(shape=[lookback.length(), len(temporal_features)], batch_size=batch_size, name='temporal_input')
    temporal_flatten = TimeDistributed(Flatten(name='temporal_flatten'))(temporal_input)

    concat = K.concatenate([spatial_flatten, temporal_flatten], axis=-1) if temporal_features else spatial_flatten
    
    fnn = Dense(units=1024, name='fnn')(concat)
    output = Dense(units=target_shape, name='target_output', activation='linear')(fnn)
    model = Model(inputs=[spatial_input, temporal_input], outputs=[output], name=model_name)

    if print_summary:
        print(model.summary())

    return (model), 'cnn'