from tensorflow import keras

def get_model(model_param, nx, ny, batch_size, seq_length, stateful=True, return_sequences=True):
    n_a = model_param['RNN_param']['n_activation']

    implementation_mode = model_param['RNN_param']['implementation']

    model = keras.models.Sequential()
    if 'SimpleRNN' in model_param['RNN_param']['RNN_type']:
        model.add(keras.layers.SimpleRNN(n_a, input_shape=(seq_length, nx), batch_size=batch_size,
                                         return_sequences=return_sequences, stateful=stateful))
    elif 'GRU' in model_param['RNN_param']['RNN_type']:
        model.add(keras.layers.GRU(n_a, input_shape=(seq_length, nx), recurrent_activation='sigmoid', batch_size=batch_size,
                                   return_sequences=return_sequences, stateful=stateful, implementation=implementation_mode))
    elif 'LSTM' in model_param['RNN_param']['RNN_type']:
        model.add(keras.layers.LSTM(n_a, input_shape=(seq_length, nx), recurrent_activation='sigmoid', batch_size=batch_size,
                                    return_sequences=return_sequences, stateful=stateful, implementation=implementation_mode))

    for units, activation in zip(model_param['n_units'], model_param['activation']):
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(units, activation=activation)))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(ny, activation='linear')))
    return model
