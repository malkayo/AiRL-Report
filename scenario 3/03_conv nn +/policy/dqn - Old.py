#!/usr/bin/python

import numpy as np
import keras.backend as K
from policy import Policy
from estimators import get_std_nn, get_conv_nn, get_conv_nn_plus


class DQN(Policy):
    # Implements a deep q learning policy
    def __init__(self, model_type, valid_actions, state_dimension, eps=1,
                 params_file=None, params_folder=None, input_mean=0., input_std=1.,
                 w_init="zero", learning_rate=1e-4):
        Policy.__init__(self, valid_actions, state_dimension,
                        eps, input_mean, input_std)

        # Define the action-state value estimator (neural net)
        from keras.optimizers import Adam
        from keras.layers import Input
        from keras.models import Model, Sequential
        self.params_file = params_file
        self.params_folder = params_folder
        self.nb_actions = len(self.valid_actions)
        self.optimizer = Adam(lr=learning_rate, beta_1=0.90,
                              beta_2=0.999, epsilon=1e-8, clipnorm=.5)

        # Use conv_nn or std_nn architecture
        model_init = None
        if model_type == "conv":
            model_init = get_conv_nn
        elif model_type == "conv+":
            model_init = get_conv_nn_plus
        elif model_type == "std":
            model_init = get_std_nn
            self.state_dimension = np.prod(self.state_dimension)
            self.input_shape = (state_dimension)
        else:
            print "Invalid model type!"
        self.model, self.input_shape = model_init(self, self.optimizer, w_init)

        # Define custom loss function
        self.mask = Input((len(self.valid_actions),))  # action mask

        def finite_action_value_loss(y_true, y_pred):
            # custom loss function for applying gradient only from the selected
            # action value
            delta = y_true - y_pred
            delta *= self.mask  # apply element-wise mask
            loss = K.mean(K.square(delta), axis=-1)  # mean squared error
            # multiply by nb_actions because 0 mask element are included in the mean
            loss *= float(self.nb_actions)
            return loss

        # The combined model receives both the model input and the action mask
        self.combined_model = Model(input=[self.model.input, self.mask],
                                    output=self.model.output)
        self.combined_model.compile(
            optimizer=self.optimizer, loss=finite_action_value_loss)

        # Initialize the target model as a copy of the main one but with weight
        # to 0
        self.target_model = Sequential.from_config(self.model.get_config())
        zero_weights = [np.zeros_like(w) for w in self.model.get_weights()]
        self.target_model.set_weights(zero_weights)

        # Load weights from file
        if params_file:
            file_location = params_folder + params_file
            print "Loading Model Weights from {}".format(file_location)
            self.model.load_weights(file_location)
            self.target_model.load_weights(file_location)

    # update the model using a batch of transitions
    def minibatch_step(self, states, actions, target_action_values, is_weights, verbose):
        n_record = states.shape[0]

        # set the action mask at 1 at the selected action
        # and 0 elsewhere
        masks = np.zeros((n_record, self.nb_actions))
        for i, a in enumerate(actions):
            action_idx = self.valid_actions.index(a)
            masks[i, action_idx] = 1.

        # defined the combined model target values
        tg = np.repeat(np.array(target_action_values),
                       self.nb_actions).reshape(n_record, self.nb_actions)

        # perform a step of gradient descent update
        metrics = self.combined_model.train_on_batch([states, masks], tg)
        print "metrics {}".format(metrics)
        return [metrics]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def Q_Target(self, trainer, sel_idx):
        # the network outputs actions values for all actions (not only the one
        # that was chosen by the agent)
        # we have to build a target such that the backprop is only for this action
        target = []
        # compute the max part of the target in a vectorized fashion
        next_states = [trainer.training_experience[i][3] for i in sel_idx]
        next_states = np.array(next_states)
        policy_values = np.max(self.target_model.predict(next_states), axis=1)
        for i in range(len(sel_idx)):
            sel_ind = sel_idx[i]
            if not trainer.training_experience[sel_ind][4]:
                # Max Q Value estimation by the Fixed Policy
                q_value = trainer.gamma * policy_values[i]
                # Normalized reward
                norm_reward = (trainer.training_experience[sel_ind][
                               2] - trainer.reward_mean) / trainer.reward_std
                target.append(norm_reward + q_value)
            else:
                # If there is no next state, it mean that the agent reached a final state
                # The value of this state is exactly the reward
                # Normalized reward
                norm_reward = (trainer.training_experience[sel_ind][
                               2] - trainer.reward_mean) / trainer.reward_std
                target.append(norm_reward)
        return target

    def return_Target(self, trainer, sel_idx):
        target = []
        for i in range(len(sel_idx)):
            sel_ind = sel_idx[i]
            norm_return = (trainer.training_experience[sel_ind][
                           2] - trainer.reward_mean) / trainer.reward_std
            target.append(norm_return)
        return target
