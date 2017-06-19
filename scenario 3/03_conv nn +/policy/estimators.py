#!/usr/bin/python


def get_std_nn(policy, optim, w_init="zero", regularization=1E-3):
    # Return an std_nn Keras model
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.regularizers import l2
    from keras.layers.advanced_activations import LeakyReLU

    input_shape = (policy.state_dimension,)

    # Model Definition
    model = Sequential()
    model.add(Dense(128, activation='linear', input_shape=input_shape,
              init=w_init, W_regularizer=l2(regularization)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(128, activation='linear', init=w_init,
              W_regularizer=l2(regularization)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(128, activation='linear', init=w_init,
              W_regularizer=l2(regularization)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(len(policy.valid_actions), init=w_init,
              W_regularizer=l2(regularization)))

    # Compile model
    model.compile(optimizer=optim, loss='mse')

    return model, input_shape


def get_conv_nn(policy, optim, w_init="zero", regularization=1E-4):
    # Return an conv_nn Keras model
    from keras.models import Sequential
    from keras.layers import Dense, Convolution2D, Flatten
    from keras.layers.advanced_activations import LeakyReLU

    input_shape = policy.state_dimension
    print "Conv NN input {}".format(input_shape)

    # Model Definition
    model = Sequential()

    model.add(Convolution2D(32, 6, 6, subsample=(3, 3), init=w_init,
              activation='linear', border_mode='valid', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), init=w_init,
              activation='linear', border_mode='valid'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init=w_init,
              activation='linear', border_mode='valid'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Flatten())
    model.add(Dense(512, init=w_init, activation="linear"))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(len(policy.valid_actions), init=w_init))

    # Compile Model
    model.compile(optimizer=optim, loss='mse')

    return model, input_shape


def get_conv_nn_plus(policy, optim, w_init="zero", regularization=1E-4):
    # Return an conv_nn Keras model
    from keras.models import Sequential
    from keras.layers import Dense, Convolution2D, Flatten
    from keras.layers.advanced_activations import LeakyReLU

    input_shape = policy.state_dimension
    print "Conv NN input {}".format(input_shape)

    # Model Definition
    model = Sequential()

    model.add(Convolution2D(64, 6, 6, subsample=(3, 3), init=w_init,
              activation='linear', border_mode='valid', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Convolution2D(128, 4, 4, subsample=(2, 2), init=w_init,
              activation='linear', border_mode='valid'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Convolution2D(128, 3, 3, subsample=(1, 1), init=w_init,
              activation='linear', border_mode='valid'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Flatten())
    model.add(Dense(1024, init=w_init, activation="linear"))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(len(policy.valid_actions), init=w_init))

    # Compile Model
    model.compile(optimizer=optim, loss='mse')

    return model, input_shape
