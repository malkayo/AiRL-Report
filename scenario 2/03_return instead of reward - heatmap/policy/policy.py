#!/usr/bin/python

import random
import numpy as np
import os


class Policy:
    model_folder = "./models/"

    def __init__(self, valid_actions, state_dimension, eps, input_mean=0.,
                 input_std=1.):
        self.valid_actions = valid_actions  # set of valid agent actions
        self.state_dimension = state_dimension
        self.input_shape = (state_dimension)
        self.state_input_number = np.product(state_dimension)
        self.eps = eps  # eps policy parameter
        self.norm_input_mean = input_mean  # policy input mean
        self.norm_input_std = input_std  # policy input standard deviation

    def new_env_reset(self, valid_actions, state_dimension):
        self.valid_actions = valid_actions
        self.state_dimension = state_dimension
        self.state_input_number = np.product(state_dimension)
        self.norm_input_mean = 0.
        self.norm_input_std = 1.

    def get_formatted_states(self, states):
        # Formatting the environment state grid into the right format for the
        # policy's value estimator
        n_record = states.shape[0]
        # Normalize values
        formatted_states = (states - self.norm_input_mean) / self.norm_input_std
        formatted_states = formatted_states.reshape((n_record,) + self.input_shape)
        return formatted_states.astype(np.float16)

    def set_eps(self, new_eps):
        self.eps = new_eps

    def eps_policy(self, state):
        # use a random action eps % of the time
        use_optimal = np.random.choice(a=[True, False], p=[1.-self.eps, self.eps])
        if use_optimal:  # use the action recommended by current policy
            action = self.optimal_action(state)
            print "use optimal"
        else:  # pick the action randomly
            action = random.choice(self.valid_actions)
            print "use random"
        return action

    def eps_policy_batch(self, states):
        # eps_policy for multiple agents using the same policy
        n_record = states.shape[0]
        actions = []
        use_optimal = np.random.choice(a=[True, False], p=[1.-self.eps, self.eps],
                                       size=n_record)
        use_optimal_ind = [i for i, v in enumerate(use_optimal) if v]
        if use_optimal_ind:
            optimal_actions = self.optimal_action_batch(states[use_optimal_ind, ])

        for i in range(n_record):
            if use_optimal[i]:
                opt_ind = use_optimal_ind.index(i)
                actions.append(optimal_actions[opt_ind])
            else:
                actions.append(random.choice(self.valid_actions))
        return actions

    def optimal_action(self, state):
        # return the action with max value predicted by the policy's model
        values = self.model.predict(state, batch_size=1).tolist()[0]
        return self.valid_actions[values.index(max(values))]

    def max_value(self, state):
        # return max value predicted by the policy's model
        value = self.model.predict(state.reshape((1,)+self.input_shape), batch_size=1).tolist()[0]
        return max(value)

    def optimal_action_batch(self, states):
        # optimal_action for multiple agents using the same policy
        values = self.model.predict(states)
        max_values_ind = np.argmax(values, axis=1)
        print "values {}".format(values)
        actions = [self.valid_actions[i] for i in max_values_ind]
        return actions

    def save_model(self, modelf_location):
        # Serialize model architecture to JSON
        model_json = self.model.to_json()
        with open(modelf_location + ".json", "w") as json_file:
            json_file.write(model_json)

        # Save model weights
        self.model.save_weights(modelf_location + ".h5", overwrite=True)

    def load_model(self, modelf_location):
        from keras.models import model_from_json

        # Load model architecture from JSON
        model_file = modelf_location + ".json"
        if os.path.isfile(model_file):
            json_file = open(model_file, 'r')
            model_json = json_file.read()
            json_file.close()
            model = model_from_json(model_json)
        else:
            print "Error - Policy model loading - {} missing".format(
                model_file)
            return None

        # Load weights from h5 file
        weights_file = model_name + ".h5"
        if os.path.isfile(weights_file):
            print "Loading weights from {}".format(weights_file)
            self.model.load_weights(weights_file)
        else:
            print "Error - Policy weight loading - {} missing".format(
                weights_file)
            return None

        return model
