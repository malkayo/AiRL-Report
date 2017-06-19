#!/usr/bin/python

from simulator.environment import Environment
from policy.dqn import DQN
from main_train import Trainer
from simulator.vizsimulator import run_simu
import numpy as np
import timeit

np.set_printoptions(threshold=np.inf, linewidth=200)


class Viz_Trainer(Trainer):
    def __init__(self):
        # self.params_file = None
        Trainer.__init__(self)
        self.params_folder = "./models/"
        self.params_file = "best_model_15044377.h5"
        self.estimator_type = "conv"
        self.nb_simu = 1000
        self.n_aircraft = 3
        self.eps = 0.01
        self.valid_actions = ["Straight", "Small Left", "Left", "Small Right", "Right"]
        self.speed = 0.05
        self.airspace_width = 1500  # Airspace width (meters)
        self.airspace_height = 1500  # Airspace height (meters)
        self.wp_heading_bool = True
        self.random_wp_position_bool = False

        # number of frames the agent sees before choosing its next action
        self.frame_buffer_size = 3
        self.input_mean = 2.13E-3
        self.input_std = 7.25E-2

    def run(self):
        start = timeit.default_timer()
        print "Viz Start"
        env = Environment(self.airspace_width, self.airspace_height)
        state_dimension = (self.frame_buffer_size, ) + env.get_state_dimension()

        # Initialize policy
        self.policy = DQN(
            model_type=self.estimator_type, valid_actions=self.valid_actions,
            state_dimension=state_dimension, eps=self.eps,
            params_file=self.params_file, params_folder=self.params_folder,
            input_mean=self.input_mean, input_std=self.input_std)

        for n in range(self.nb_simu):
            print "!!!! Simulation {}".format(n + 1)
            print "input std {}".format(self.input_std)
            print "input mean {}".format(self.input_mean)
            run_simu(self, self.n_aircraft, self.frame_buffer_size,
                     verbose=True,
                     render_bool=True,
                     frame_delay=int(1./(self.speed+1e-6)),
                     wp_heading_bool=self.wp_heading_bool,
                     random_wp_position_bool=self.random_wp_position_bool)
            # print "Simulation hist {}".format(hist)
        stop = timeit.default_timer()
        print "training duration {}".format(stop-start)


if __name__ == '__main__':
    trainer = Viz_Trainer()
    trainer.run()
