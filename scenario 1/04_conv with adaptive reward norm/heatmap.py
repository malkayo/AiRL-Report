#!/usr/bin/python

from simulator.environment import Environment
from policy.dqn import DQN
from simulator.simulator import multiple_run
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import random


class HeatMapTrainer():

    def __init__(self):
        self.params_folder = "./models/"
        self.params_file = "model_save_943148.h5"
        self.estimator_type = "conv"
        self.nb_simu = 2000
        self.n_aircraft = 1
        self.eps = 0.01
        self.valid_actions = ["Straight", "Small Left", "Left", "Small Right", "Right"]
        self.speed = 1
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

        self.return_instead_of_reward = False
        self.training_experience = []

    def run(self):
        start = timeit.default_timer()
        print "Simu Start Heatmap"
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
            update_experience=True)

        print "landing rate {}".format(batch_landing_rate)
        print "eval score {}".format(eval_score)
        print "valid episodes {}".format(valid_episodes)

        states = [exp[0] for exp in self.training_experience]
        last_frame = [v[-1] for v in states]
        acft_id = (2.0-self.input_mean)/self.input_std
        pos_index = [np.argwhere(v == acft_id) for v in last_frame]
        acft_index = []
        for v in pos_index:
            for ind in v:
                acft_index.append(ind)
        max_val = [self.policy.max_value(s) for s in states]
        data = pd.DataFrame({"acft_index": acft_index, "q_value": max_val})
        data["acft_index"] = data["acft_index"].apply(tuple)
        mean_value = data.groupby("acft_index", as_index=True).mean()
        x = [v[0] + 1 for v in mean_value.index.values]
        y = [v[1] + 1 for v in mean_value.index.values]
        z = mean_value["q_value"]

        plt.style.use('ggplot')
        plt.rcParams['axes.facecolor'] = 'w'
        plt.rcParams['grid.color'] = 'lightgrey'
        plt.gca().set_aspect('equal')
        plt.yscale('linear')
        plt.scatter(x, y, c=z, cmap=plt.cm.jet, marker="s", s=13)
        plt.savefig("heatmap.png", dpi=450)
        stop = timeit.default_timer()
        print "training duration {}".format(stop-start)

    def experience_update(self, transitions):
        # Update the experience replay list
        # (Initialize the transitions' priority values)
        if transitions:
            self.training_experience += transitions


if __name__ == '__main__':
    trainer = HeatMapTrainer()
    trainer.run()
