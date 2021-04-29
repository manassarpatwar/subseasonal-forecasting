from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def build_model(lookback, spatial_grid_shape, batch_size, spatial_features, temporal_features, target_shape, model_name, print_summary=True):
    spatial_input = Input(shape=[lookback.length(), *spatial_grid_shape, len(spatial_features)], batch_size=batch_size, name='spatial_input')
    spatial_flatten = Flatten(name='spatial_flatten')(spatial_input)
    
    temporal_input = Input(shape=[lookback.length(), len(temporal_features)], batch_size=batch_size, name='temporal_input')
    temporal_flatten = Flatten(name='temporal_flatten')(temporal_input)

    concat = K.concatenate([spatial_flatten, temporal_flatten], axis=-1) if temporal_features else spatial_flatten
    
    fnn = Dense(units=1024, name='hidden1')(concat)
    fnn = Dropout(rate=0.5)(fnn)
    fnn = Dense(units=1024, name='hidden2')(concat)
    fnn = Dropout(rate=0.5)(fnn)

    output = Dense(units=target_shape, name='target_output', activation='linear')(fnn)
    model = Model(inputs=[spatial_input, temporal_input], outputs=[output], name=model_name)

    if print_summary:
        print(model.summary())

    return (model), 'fnn'