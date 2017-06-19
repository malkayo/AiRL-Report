#!/usr/bin/python

from simulator.environment import Environment
from simulator.simulator import multiple_run, run_simu
from policy.dqn import DQN
from experience.metric_display import plot_metrics, save_training_history
from experience.experience_sampling import sample_records, update_priority
import timeit
import random
import numpy as np


class Trainer:
    def __init__(self):
        # Experience replay parameters
        self.initial_runs = 15000
        self.max_exp_size = 1000000
        self.nb_simu = 50000000

        # Policy parameters
        self.policy = None
        self.estimator_type = "conv"
        self.params_file = None
        self.params_folder = "./models/"
        self.learning_rate = 1E-4
        self.select_size = 64  # minibatch size 32
        self.update_rate = 16  # 4 in DQN paper
        self.weight_update_rate = 50000  # 10,000 simu step according to literature on DQN
        self.model_save_rate = 5000

        self.graph_rate = 500
        self.keep_all_files = True  # keep all model .h5 and dashboard files
        self.n_eval = 100

        self.n_aircraft = 3
        self.airspace_width = 1500  # Airspace width (meters)
        self.airspace_height = 1500  # Airspace height (meters)
        self.valid_actions = ["Straight", "Small Left", "Left",
                              "Small Right", "Right"]
        # number of frames the agent sees before choosing its next action
        self.frame_buffer_size = 3
        self.wp_heading_bool = True
        self.random_wp_position_bool = False
        # indicate if simulator will return discounted return instead of
        # immediate rewards for the experience transitions
        self.adaptive_reward = True
        self.return_instead_of_reward = True

        self.gamma = .95
        self.exploration_steps = 40000.  # 1M simu step in DQN
        self.init_eps = 1
        self.final_eps = 0.05

        self.update_step = 0
        self.old_policy_update_step = - self.update_rate
        self.old_target_model_update_step = 0

        self.eval_scores = []
        self.eval_landing_rate = []
        self.max_eval_landing_rate = 0

        self.training_scores_value = []
        self.training_scores_ind = []

        self.model_train_loss_values = []
        self.model_train_loss_ind = []

        self.training_experience = []
        self.validation_experience = []

        self.seed = 1234  # random generators seed
        np.random.seed(self.seed)  # for numpy
        random.seed(self.seed)  # for python random library

        # Input/Reward Normalization
        self.input_mean = 2.13E-3
        self.input_std = 7.25E-2
        self.reward_mean = 0.
        self.reward_std = 1.

        # Prioritized Replay
        self.priority_sampling = True
        self.priority = np.ones(0)
        self.priority_sum = 0
        self.pr_alpha = 0.5
        self.pr_beta = 0.5
        self.max_td_error = 10.

    def run(self):
        start = timeit.default_timer()

        # Initialize the environment
        env = Environment(self.airspace_width, self.airspace_height)
        state_dimension = (self.frame_buffer_size, ) + \
            env.get_state_dimension()

        # ----------------------------------------------------------------------
        # Run a batch of simulations to get initial experience
        # ----------------------------------------------------------------------
        print "!!!! Batch Initialization"

        # Set up policy used for the batch initialization
        self.policy = DQN(self.estimator_type, self.valid_actions,
                          state_dimension, self.init_eps,
                          self.params_file, self.params_folder,
                          self.input_mean, self.input_std)
        # Run several simulation and store the outcome
        land_rate, batch_score, valid_episodes = multiple_run(
            self, self.initial_runs, self.n_aircraft, self.gamma,
            self.frame_buffer_size, update_model=False, mverbose=False,
            wp_heading_bool=self.wp_heading_bool, random_wp_position_bool=self.random_wp_position_bool)
        print "initial exp len {}".format(len(self.training_experience))
        print "landing rate {}".format(land_rate)

        # Compute the reward normalization from the inital experience
        reward = np.array([e[2] for e in self.training_experience])
        self.reward_mean = np.mean(reward)
        self.reward_std = np.std(reward)
        print "reward mean {}".format(self.reward_mean)
        print "reward std {}".format(self.reward_std)

        # Initialize policy for training
        self.policy = DQN(
            model_type=self.estimator_type, valid_actions=self.valid_actions,
            state_dimension=state_dimension, eps=self.init_eps,
            params_file=self.params_file, params_folder=self.params_folder,
            input_mean=self.input_mean, input_std=self.input_std,
            w_init="he_normal", learning_rate=self.learning_rate)

        # ----------------------------------------------------------------------
        # Run a lot of simulations and update the policy
        # The model update occur within the simulation at every decision
        # step for the agents (call main_train.update())
        # ----------------------------------------------------------------------
        print "Start training!"
        for n in range(self.nb_simu):
            print "!!!! Simulation {}".format(n + 1)
            self.eps = max(
                (self.init_eps - n / self.exploration_steps), self.final_eps)
            self.policy.set_eps(self.eps)
            reward, landing_count, _, _ = run_simu(
                self, self.n_aircraft, self.gamma, self.frame_buffer_size,
                verbose=False, update_model=True,
                wp_heading_bool=self.wp_heading_bool,
                random_wp_position_bool=self.random_wp_position_bool)
            self.training_scores_value.append(reward)
            self.training_scores_ind.append(self.update_step)

            # Plot Metrics
            if n % self.graph_rate == 0:
                plot_metrics(self)
                print "reward mean {}".format(self.reward_mean)
                print "reward std {}".format(self.reward_std)

            # Save model weights
            if n % self.model_save_rate == 0:
                model_file = self.params_folder + "model_save"
                if self.keep_all_files:
                    model_file += "_{}".format(self.update_step)
                self.policy.save_model(model_file)

            # Keep the experience history data sets to a reasonable size
            train_exp_len = len(self.training_experience)
            if train_exp_len > self.max_exp_size:
                # delete the oldest experience
                cut_idx = train_exp_len - self.max_exp_size
                self.training_experience = self.training_experience[cut_idx:]
                self.priority_sum -= sum(self.priority[:cut_idx])
                self.priority = self.priority[cut_idx:]

        save_training_history(self)
        stop = timeit.default_timer()
        print "training duration {}".format(stop - start)

    def experience_update(self, transitions):
        # Update the experience replay list
        # (Initialize the transitions' priority values)
        if transitions:
            self.training_experience += transitions
            if self.priority_sampling:
                # Initialize priority with max_td_error
                print "transition len {}".format(len(transitions))
                print "priority len {}".format(len(self.priority))
                transitions_priority = np.repeat(
                    self.max_td_error, len(transitions))**self.pr_alpha
                self.priority_sum += np.sum(transitions_priority)
                self.priority = np.append(self.priority, transitions_priority)

    def policy_update(self):
        exp_len = len(self.training_experience)
        if exp_len < self.select_size:
            return
        # Perform a policy update
        if (self.update_step - self.old_policy_update_step) >= self.update_rate:
            self.old_policy_update_step = self.update_step
            # sample experience with high self.priority more often
            # self.priority is high if it is new experience or if TD-error is
            # high
            r = sample_records(trainer, self.priority_sampling)
            sel_idx, is_weights, selected_priority_sum = r

            if self.return_instead_of_reward:
                target_action_values = self.policy.return_Target(
                    self, sel_idx)
            else:
                target_action_values = self.policy.Q_Target(self, sel_idx)

            select_states = np.array(
                [self.training_experience[i][0] for i in sel_idx], dtype=np.int16)
            select_actions = [self.training_experience[i][1] for i in sel_idx]

            # Perform model update
            model_train_loss = self.policy.minibatch_step(
                select_states, select_actions, target_action_values, is_weights,
                verbose=False)

            self.model_train_loss_values.append(model_train_loss[0])

            self.model_train_loss_ind.append(self.update_step)
            print "Model Train Loss {}".format(model_train_loss)

            # Update td-error of the sampled experience
            if self.priority_sampling:
                update_priority(trainer, select_states, select_actions,
                                target_action_values, selected_priority_sum,
                                sel_idx)

        # Update the weights of the fixed policy
        if (self.update_step - self.old_target_model_update_step) >= self.weight_update_rate:
            self.old_target_model_update_step = self.update_step
            print "target model update step {}".format(self.update_step)
            self.policy.update_target_model()

            # Also update the reward mean/std if required
            if self.adaptive_reward:
                latest_rewards = np.array(
                    [e[2] for e in self.training_experience[-10000:]])
                latest_mean = np.mean(latest_rewards)
                latest_std = np.std(latest_rewards)
                self.reward_mean = 0.95 * self.reward_mean + 0.05 * latest_mean
                self.reward_std = 0.95 * self.reward_std + 0.05 * latest_std

    def increment_update_step(self, increment):
        self.update_step += increment


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
