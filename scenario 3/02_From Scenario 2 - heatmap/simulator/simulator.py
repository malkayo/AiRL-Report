import os
import sys
import random
import numpy as np
import pandas as pd
from environment import Environment
from copy import copy
np.set_printoptions(threshold=np.inf, linewidth=200)


class Simulator(object):
    MAX_T_COUNT = 2000  # Max episode duration

    def __init__(self, env, frame_buffer_size, trainer=None,
                 update_experience=True):

        self.trainer = trainer
        self.update_experience = update_experience
        self.env = env
        self.quit = False
        self.action_frame_period = 12  # number of frames between each actions_string
        self.simu_history = {}  # state/action/rewards
        self.frame_buffers = {}  # latest frames for each aircraft
        self.frame_buffer_size = frame_buffer_size
        self.empty_frame_buffer = np.zeros(
            (frame_buffer_size,) + self.env.get_state_dimension())
        self.cumu_reward = {}  # Cumulative reward of the previous frames
        self.landing_count = 0  # How many aircraft have landed

        # Store individual aircraft experience, frames and reward
        for aircraft in self.env.aircrafts:
            self.simu_history[aircraft.call_sign] = {
                "state": [], "action": [], "reward": []}
            # frames between decisions
            self.frame_buffers[aircraft.call_sign] = []
            # cumulative reward between decisions
            self.cumu_reward[aircraft.call_sign] = 0

    def run(self, update_model=False):
        self.quit = False
        t_count = 0
        total_reward = 0  # Total undiscounted reward of the simulation
        final_aircrafts = []  # List of aircraft that have reached a final state
        while True:  # until env.done
            # Store frame_buffer_size frames (env state grid) for each aircraft
            # Between each decision times
            for aircraft in self.env.aircrafts:
                if t_count % int(self.action_frame_period / self.frame_buffer_size) != 0:
                    continue
                call_sign = aircraft.call_sign
                # The representation of the (same) environment is different
                # for each aircraft because it highlights the position of the
                # focus aircraft
                state_grid = self.env.get_state_grid(aircraft)
                if not self.frame_buffers[call_sign]:
                    self.frame_buffers[call_sign] = [
                        copy(state_grid)] * self.frame_buffer_size
                self.frame_buffers[call_sign].pop(0)  # remove the oldest frame
                self.frame_buffers[call_sign].append(copy(state_grid))

            # Periodically choose an action
            if t_count % self.action_frame_period == 0:
                if update_model:
                    self.trainer.increment_update_step(len(self.env.aircrafts))

                # Choose an action depending current states
                # Apply epsilon greedy policy
                policy_inputs = np.array(
                    [np.array(self.frame_buffers[ac.call_sign])
                     for ac in self.env.aircrafts],
                    dtype=np.int8)
                policy_inputs = self.trainer.policy.get_formatted_states(
                    policy_inputs)
                actions = self.trainer.policy.eps_policy_batch(policy_inputs)
                # actions_string = "-".join(actions)
                # df = pd.DataFrame({"action_string": [actions_string]})
                # df.to_csv("actions.csv", mode="a", index=False, header=False)
                for i, aircraft in enumerate(self.env.aircrafts):
                    call_sign = aircraft.call_sign
                    action = actions[i]
                    clearance = self._get_clearance(aircraft, action)
                    self.env.set_clearance(aircraft, clearance)

                    # Save the current state, the action picked and the reward
                    # observed at this step (actually related to the previous
                    # step)
                    self.save_transition(
                        call_sign, self.frame_buffers[call_sign], action,
                        self.cumu_reward[call_sign])
                    self.cumu_reward[call_sign] = 0  # reset the reward count

            # Move all aircraft following their clearances
            rewards, final_aircrafts, ld_count = self.env.step(t_count)

            total_reward += np.sum(np.array([r for k,
                                             r in rewards.iteritems()]))
            self.landing_count += ld_count

            # Store intermediate time step rewards (between actions)
            for aircraft in self.env.aircrafts:
                call_sign = aircraft.call_sign
                self.cumu_reward[call_sign] += rewards[call_sign]
            # Save transition if final state is reached
            for call_sign in final_aircrafts:
                # Save the reward of the final transition
                # There are no final state or action per se because the agent
                # did not have the chance to experience them
                self.cumu_reward[call_sign] += rewards[call_sign]
                self.save_transition(
                    call_sign, reward=self.cumu_reward[call_sign],
                    state=self.empty_frame_buffer, action=None)

            if self.update_experience:
                self.trainer.experience_update(
                    self.get_history(final_aircrafts))

            if t_count % self.action_frame_period == 0:
                # Perform a policy update
                if update_model:
                    self.trainer.policy_update()

            if self.quit or self.env.done:
                valid_episode = len(self.simu_history) > 1
                # df = pd.DataFrame({"duration": [t_count],
                #                    "land_count": [ld_count],
                #                    "reward": [total_reward]})
                # df.to_csv("durations.csv", mode="a", index=False, header=False)
                return total_reward, self.landing_count, valid_episode, t_count

            if t_count > self.MAX_T_COUNT:
                print "Episode too long"
                for aircraft in self.env.aircrafts:
                    call_sign = aircraft.call_sign
                    self.save_transition(
                        call_sign, reward=self.cumu_reward[call_sign],
                        state=self.empty_frame_buffer, action=None)
                if self.update_experience:
                    self.trainer.experience_update(self.get_history(
                        [ac.call_sign for ac in self.env.aircrafts]))

                valid_episode = 0
                # df = pd.DataFrame({"duration": [t_count], "land_count": [ld_count]})
                # df.to_csv("durations.csv", mode="a", index=False, header=False)
                return total_reward, self.landing_count, valid_episode, t_count
            t_count += 1

    def save_transition(self, call_sign, state, action, reward):
        self.simu_history[call_sign]["state"].append(copy(state))
        self.simu_history[call_sign]["action"].append(copy(action))
        self.simu_history[call_sign]["reward"].append(copy(reward))

    def get_history(self, call_signs):
        if call_signs:
            result = []
            for call_sign in call_signs:
                history_len = len(self.simu_history[call_sign]["state"])
                if history_len > 1:
                    all_states = self.trainer.policy.get_formatted_states(
                        np.array(self.simu_history[call_sign]["state"]))
                    print "all states shape {}".format(all_states.shape)
                    states = copy(all_states[:-1])
                    actions = copy(self.simu_history[call_sign]["action"][:-1])
                    rewards = copy(self.simu_history[call_sign]["reward"][1:])
                    next_states = copy(all_states[1:])
                    final_state = (history_len - 1) * [False]
                    final_state[-1] = True
                    if self.trainer and self.trainer.return_instead_of_reward:
                        returns = self._get_return(rewards)
                        result += zip(states, actions, returns,
                                      next_states, final_state)
                    else:
                        result += zip(states, actions, rewards,
                                      next_states, final_state)
            # df = pd.DataFrame({"state": [len(states)], "actions": [len(actions)],
            #                    "next_states": [len(next_states)],
            #                    "reward": [len(rewards)], "result": [len(result)]})
            # df.to_csv("reward.csv", mode="a", index=False, header=False)
            print "history done"
            return result

    def _get_return(self, rewards):
        nb_reward = len(rewards)
        discounted_rewards = np.zeros((nb_reward, nb_reward))
        for i in range(nb_reward):
            for j in range(i, nb_reward):
                discounted_rewards[i, j] = rewards[
                    j] * self.trainer.gamma**(j - i)
        returns = np.sum(discounted_rewards, axis=1).tolist()
        return returns

    def _get_clearance(self, aircraft, action):
        previous_heading = aircraft.heading

        # delta is time required for the aircraft to reach the desired heading
        # we add stochasticity to the heading clearance to avoid being stuck in loops
        # sigma = 5% so we do over 10% too often (<5%)
        delta = np.random.normal(1, 0.05) * \
            self.env.delta_t * self.action_frame_period
        full_heading_delta = aircraft.turn_rate * delta

        if action == "Straight":
            new_clearance = previous_heading
        elif action == "Left":
            new_clearance = previous_heading - full_heading_delta
        elif action == "Right":
            new_clearance = previous_heading + full_heading_delta
        elif action == "Small Left":
            new_clearance = previous_heading - full_heading_delta / 4.
        elif action == "Small Right":
            new_clearance = previous_heading + full_heading_delta / 4.
        new_clearance %= 360
        return new_clearance


def run_simu(trainer, n_aircraft, gamma, frame_buffer_size,
             verbose=False, update_model=False, update_experience=True,
             wp_heading_bool=False, random_wp_position_bool=False):
    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    print "!!!NEW SIMULATION!!!"
    # Create environment and populate with waypoint and aircraft
    env = Environment(trainer.airspace_width, trainer.airspace_height)

    if random_wp_position_bool:
        waypoint_x = (env.width / 4) + np.random.uniform() * env.width / 2
        waypoint_y = (env.height / 4) + np.random.uniform() * env.height / 2
    else:
        waypoint_x = env.width / 2
        waypoint_y = env.height / 2

    # if wp_heading_bool:
    #     wp_heading = random.choice([90, 180, 270, 360])
    # else:
    #     wp_heading = None

    # Fixed waypoint
    wp_heading = 180

    env.create_waypoint(np.array([waypoint_x, waypoint_y, 0]), "WP1", "blue",
                        wp_heading)
    for i in range(n_aircraft):
        env.create_aircraft("RL00{}".format(i))

    # Run Simulation
    simulator = Simulator(env, frame_buffer_size, trainer, update_experience)
    total_reward, landing_count, valid_episode, duration = simulator.run(
        update_model)

    if not verbose:
        sys.stdout = old_stdout

    return total_reward, landing_count, valid_episode, duration


def multiple_run(trainer, nb_simu, n_aircraft, gamma, frame_buffer_size,
                 update_model=False, mverbose=False, update_experience=True,
                 wp_heading_bool=False, random_wp_position_bool=False):
    batch_landing_rate = 0.  # Average proportion of aircraft that successfully land per simulation
    total_landing = 0  # number of successful landing
    batch_score = 0.  # average simulation score (total reward)
    valid_simu_count = 0.  # Number of simulation with a least one transition
    total_duration = 0

    for n in range(nb_simu):
        print "!!!!Simulation {}".format(n + 1)
        # flag that determines if the simulation had at least one valid
        # transition
        valid_episode = False
        total_reward, landing_count, valid_episode, duration = run_simu(
            trainer, n_aircraft, gamma, frame_buffer_size,
            verbose=mverbose, update_model=update_model,
            update_experience=update_experience,
            wp_heading_bool=wp_heading_bool,
            random_wp_position_bool=random_wp_position_bool)

        if valid_episode:
            total_duration += duration
            valid_simu_count += 1
            batch_score += total_reward
            total_landing += landing_count
    print "total duration {}".format(total_duration)
    if valid_simu_count > 0:
        batch_score /= valid_simu_count  # average score
        batch_landing_rate = total_landing / \
            (valid_simu_count * n_aircraft)  # % of successfull episodes
    return batch_landing_rate, batch_score, valid_simu_count
