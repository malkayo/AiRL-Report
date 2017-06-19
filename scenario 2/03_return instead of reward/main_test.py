#!/usr/bin/python

from simulator.environment import Environment
from policy.dqn import DQN
from simulator.simulator import multiple_run
import numpy as np
import timeit
import random

np.set_printoptions(threshold=np.inf, linewidth=200)


class Trainer:
    def __init__(self):
        self.params_folder = "./models/"
        self.params_file = "best_model_7522810.h5"
        self.estimator_type = "conv"
        self.nb_simu = 1000
        self.n_aircraft = 3
        self.eps = 0.01
        self.valid_actions = ["Straight", "Small Left", "Left", "Small Right", "Right"]
        self.airspace_width = 1500  # Airspace width (meters)
        self.airspace_height = 1500  # Airspace height (meters)
        self.wp_heading_bool = False
        self.random_wp_position_bool = False

        # number of frames the agent sees before choosing its next action
        self.frame_buffer_size = 3
        self.input_mean = 2.13E-3
        self.input_std = 7.25E-2

        self.seed = 1234  # random generators seed
        np.random.seed(self.seed)  # for numpy
        random.seed(self.seed)  # for python random library

    def run(self):
        start = timeit.default_timer()
        print "Simu Start"
        env = Environment(self.airspace_width, self.airspace_height)
        state_dimension = (self.frame_buffer_size, ) + env.get_state_dimension()

        # Initialize policy
        self.policy = DQN(
            model_type=self.estimator_type, valid_actions=self.valid_actions,
            state_dimension=state_dimension, eps=self.eps,
            params_file=self.params_file, params_folder=self.params_folder,
            input_mean=self.input_mean, input_std=self.input_std)
        batch_landing_rate, eval_score, valid_episodes = multiple_run(
            self, self.nb_simu, self.n_aircraft, 1,
            self.frame_buffer_size, mverbose=False, update_model=False,
            update_experience=False)

        print "landing rate {}".format(batch_landing_rate)
        print "eval score {}".format(eval_score)
        print "valid episodes {}".format(valid_episodes)
        # print "Simulation hist {}".format(hist)
        stop = timeit.default_timer()
        print "training duration {}".format(stop-start)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
