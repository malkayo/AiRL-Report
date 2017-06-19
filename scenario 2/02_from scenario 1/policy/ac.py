#!/usr/bin/python

import numpy as np
import keras.backend as K
from policy import Policy
from estimators import get_simple_nn, get_simple_nn_soft


class ActorCritic(Policy):
    def __init__(self, valid_actions, state_dimension, eps=1, params_file=None,
                 w_init="zero", actor_lr=1E-5, critic_lr=1E-4):
        Policy.__init__(self, valid_actions, state_dimension, eps)
        from keras.optimizers import Adam
        from keras.layers import Input
        from keras.models import Model, Sequential
        from keras.layers.core import Activation

        self.nb_actions = len(self.valid_actions)

        ######################################################
        # Initialize the Actor Model
        ######################################################
        self.actor_optimizer = Adam(lr=actor_lr, clipvalue=.01)
        self.actor_model, self.input_shape = get_simple_nn_soft(self, self.actor_optimizer, w_init)
        self.actor_mask = Input((len(self.valid_actions),))

        def actor_loss(critic_q, y_pred):
            # custom loss function for applying gradient only from the selected action probability
            delta = -critic_q*K.log(y_pred)
            delta *= self.actor_mask  # apply element-wise mask
            loss = K.mean(delta, axis=-1)
            loss *= float(self.nb_actions)  # multiply by nb_actions because 0 mask element are included in the mean
            return loss

        self.combined_actor_model = Model(
            input=[self.actor_model.input, self.actor_mask],
            output=self.actor_model.output)
        self.combined_actor_model.compile(optimizer=self.actor_optimizer, loss=actor_loss)

        # Initialize the critic target model as a copy of the critic model with
        # weights initialized at 0
        self.actor_target_model = Sequential.from_config(self.actor_model.get_config())
        zero_weights = [np.zeros_like(w) for w in self.actor_model.get_weights()]
        self.actor_target_model.set_weights(zero_weights)

        ######################################################
        # Initialize the Critic Model
        ######################################################
        self.critic_optimizer = Adam(lr=critic_lr, beta_1=0.90, beta_2=0.999,
                                     epsilon=1e-8, clipnorm=1.)
        self.critic_model, _ = get_simple_nn(self, self.critic_optimizer, w_init)
        self.critic_mask = Input((len(self.valid_actions),))  # usual action-value estimator

        def critic_loss(y_true, y_pred):
            # custom loss function for applying gradient only from the selected action value
            delta = y_true - y_pred
            delta *= self.critic_mask  # apply element-wise mask
            loss = K.mean(K.square(delta), axis=-1)  # mean squared error
            loss *= float(self.nb_actions)  # multiply by nb_actions because 0 mask element are included in the mean
            return loss

        self.combined_critic_model = Model(
            input=[self.critic_model.input, self.critic_mask],
            output=self.critic_model.output)
        self.combined_critic_model.compile(optimizer=self.critic_optimizer, loss=critic_loss)

        # TODO load weights from multiple param files

        # Initialize the critic target model as a copy of the critic model with
        # weights initialized at 0
        self.critic_target_model = Sequential.from_config(self.critic_model.get_config())
        zero_weights = [np.zeros_like(w) for w in self.critic_model.get_weights()]
        self.critic_target_model.set_weights(zero_weights)

    def minibatch_step(self, states, actions, critic_target_action_values, is_weights, verbose):
        n_record = states.shape[0]

        # TODO use action_mask function
        masks = np.zeros((n_record, self.nb_actions))
        for i, a in enumerate(actions):
            action_idx = self.valid_actions.index(a)
            masks[i, action_idx] = 1.

        # Actor Update
        critic_q = self.critic_target_model.predict(states)
        actor_metric = self.combined_actor_model.train_on_batch(
            [states, masks], critic_q)

        if np.isnan(actor_metric):
            print "Nan actor Loss"
            print "critic q {}".format(critic_q)

        # Critic Update
        critic_tg = np.repeat(np.array(critic_target_action_values), self.nb_actions)
        critic_tg = critic_tg.reshape(n_record, self.nb_actions)
        critic_metric = self.combined_critic_model.train_on_batch(
            [states, masks], critic_tg)

        return critic_metric, actor_metric

    """
    Select Action Depending on Softmax Probability
    """
    def optimal_action_batch(self, states):
        actions = []
        probas = self.actor_model.predict(states)
        n_record = states.shape[0]
        for i in range(n_record):
            proba = probas[i]
            print "Actions Probabilities {}".format(proba)
            sel_action = np.random.choice(a=self.valid_actions, p=proba)
            if np.isnan(proba[0]):
                print "NAN proba"
                print "NAN state {}".format(states[i])
                print "NAN formatted state {}".format(states[i])
                # print "{}".format(1/0)
            print "Selected Action {}".format(sel_action)
            actions.append(sel_action)
        return actions

    def update_target_model(self):
        self.critic_target_model.set_weights(self.critic_model.get_weights())

    def vect_Q_Target(self, trainer, sel_idx):
        target = []
        # compute the max part of the target in a vectorized fashion
        next_states = [trainer.training_experience[i][3] for i in sel_idx]
        # if the next state is None (final) replace it with an input full of 0
        input_shape = (2L, 40L, 40L)
        zero_input = np.zeros(input_shape)
        next_states = [s if s is not None else zero_input for s in next_states]
        next_states = np.array(next_states)
        policy_values = np.max(self.critic_target_model.predict(next_states, batch_size=256), axis=1)
        for i in range(len(sel_idx)):
            sel_ind = sel_idx[i]
            if trainer.training_experience[sel_ind][3] is not None:
                # Max Q Value estimation by the Fixed Policy
                q_value = trainer.gamma*policy_values[i]
                # Normalized reward
                norm_reward = (trainer.training_experience[sel_ind][2]-trainer.reward_mean)/trainer.reward_std
                target.append(norm_reward+q_value)
            else:
                # If there is no next state, it mean that the agent reached a final state
                # The value of this state is exactly the reward
                # Normalized reward
                norm_reward = (trainer.training_experience[sel_ind][2]-trainer.reward_mean)/trainer.reward_std
                target.append(norm_reward)
        return target

    def save_params(self, file_prefix):
        h5_suffix = ".h5"
        self.actor_model.save_weights(file_prefix+"_actor"+h5_suffix, overwrite=True)
        self.critic_model.save_weights(file_prefix+"_critic"+h5_suffix, overwrite=True)
