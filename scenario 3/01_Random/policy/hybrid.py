#!/usr/bin/python

import numpy as np
import keras.backend as K
from policy import DQN
from policy import Policy
from estimators import get_simple_nn, get_conv_nn


class DQN_Merge(DQN):
    def __init__(self, model_type, valid_actions, state_dimension, eps=1, params_file=None,
                 w_init="zero", learning_rate=1e-4):
        Policy.__init__(self, valid_actions, state_dimension, eps)
        from keras.optimizers import Adam
        from keras.layers import Input
        from keras.models import Model, Sequential
        self.nb_actions = len(self.valid_actions)

        self.optimizer = Adam(lr=learning_rate, beta_1=0.90, beta_2=0.999, epsilon=1e-8, clipnorm=.5)

        model_init = None
        if model_type == "conv":
            model_init = get_conv_nn
        elif model_type == "linear":
            model_init = get_simple_nn
        else:
            print "Invalid model type!"
        self.model, self.input_shape = model_init(self, self.optimizer, w_init)

        self.mask = Input((len(self.valid_actions),))

        def finite_action_value_loss(y_true, y_pred):
            # custom loss function for applying gradient only from the selected action value
            delta = y_true - y_pred
            delta *= self.mask  # apply element-wise mask
            loss = K.mean(K.square(delta), axis=-1)  # mean squared error
            loss *= float(self.nb_actions)  # multiply by nb_actions because 0 mask element are included in the mean
            return loss

        self.combined_model = Model(input=[self.model.input, self.mask],
                                    output=self.model.output)
        self.combined_model.compile(optimizer=self.optimizer, loss=finite_action_value_loss)

        # Initialize the target model as a copy of the main one but with weight to 0
        self.target_model = Sequential.from_config(self.model.get_config())
        zero_weights = [np.zeros_like(w) for w in self.model.get_weights()]
        self.target_model.set_weights(zero_weights)

        # Load weights from file
        if params_file:
            print "Loading Model Weights from {}".format(params_file)
            self.model.load_weights(params_file)
            self.target_model.load_weights(params_file)

