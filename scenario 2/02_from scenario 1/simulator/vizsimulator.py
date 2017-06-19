import os
import sys
import random
import numpy as np
import pandas as pd
from environment import Environment
from simulator import Simulator
import pygame
import time
from copy import copy


class VisualSimulator(Simulator):

    def __init__(self, env, frame_buffer_size, trainer=None, frame_delay=0, update_delay=0,
                 render_bool=True):
        Simulator.__init__(self, env, frame_buffer_size, trainer, update_experience=False)

        self.current_time = 0.
        self.last_updated = 0.
        self.start_time = 0.
        self.update_delay = update_delay
        self.frame_delay = frame_delay
        self.new_action = {}

        self.render_bool = render_bool
        if self.render_bool:
            self.width = 600
            self.height = 600
            self.position_scale = self.width / self.env.width

            self.bg_color = (224, 224, 224)  # Dim grey background

            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))

            self.arcrft_sprite_size = (40, 40)
            self.aircraft_sprites = {}
            self.circle_sprites = {}

            for acrft in self.env.aircrafts:
                pic_path = "icons/airplane-{}.png".format(acrft.color)
                self.aircraft_sprites[acrft] = pygame.transform.smoothscale(
                    pygame.image.load(pic_path), self.arcrft_sprite_size)

            self.wp_sprite_size = (40, 40)
            self.waypoint_sprites = {}
            self.waypoint_pos = {}
            for (k, wp) in self.env.waypoints.items():
                if wp.heading:
                    pic_path = "icons/arrow-{}.png".format(wp.color)
                else:
                    pic_path = "icons/flag-{}.png".format(wp.color)
                self.waypoint_sprites[wp] = pygame.transform.smoothscale(
                    pygame.image.load(pic_path), self.wp_sprite_size)
                if wp.heading:
                    self.waypoint_sprites[wp] = pygame.transform.rotate(
                        self.waypoint_sprites[wp], -wp.heading)

                self.waypoint_pos[wp] = (wp.xyz_pos[0] * self.position_scale,
                                         self.height - wp.xyz_pos[1] * self.position_scale - self.wp_sprite_size[1] / 2)

            self.font_size = 16
            self.font = pygame.font.SysFont('Consolas', self.font_size)
            pygame.display.set_caption('AiRL Simulator')

    def run(self):
        self.quit = False
        self.last_updated = - self.update_delay
        self.start_time = time.time()
        t_count = 0
        cumu_reward = 0  # Cumulative reward of the last frames
        total_reward = 0  # Total undiscounted reward of the simulation
        while True:
            self.current_time = time.time() - self.start_time
            try:
                # Handle events
                if self.render_bool:
                    for event in pygame.event.get():

                        if event.type == pygame.QUIT:
                            self.quit = True
                        elif event.type == pygame.KEYDOWN:
                            if event.unicode == u' ':
                                print "!!!!!! Skip to Next Simulation"
                                self.quit = True
                            elif event.unicode == u'q':
                                raise self.VizSimulatorQuit(
                                    "User input forced simulator interruption")

                # Update environment
                if (self.current_time - self.last_updated) >= self.update_delay:
                    # Store frames
                    for aircraft in self.env.aircrafts:
                        if t_count % int(self.action_frame_period / self.frame_buffer_size) != 0:
                            continue
                        call_sign = aircraft.call_sign
                        state_grid = self.env.get_state_grid(aircraft)
                        if not self.frame_buffers[call_sign]:
                            self.frame_buffers[call_sign] = [
                                copy(state_grid)] * self.frame_buffer_size
                        self.frame_buffers[call_sign].pop(
                            0)  # remove he oldest frame
                        self.frame_buffers[call_sign].append(copy(state_grid))
                    # Periodically take action
                    if t_count % self.action_frame_period == 0:
                        # Choose an action depending current states
                        # Apply epsilon greedy policy
                        policy_inputs = np.array(
                            [np.array(self.frame_buffers[ac.call_sign])
                             for ac in self.env.aircrafts],
                            dtype=np.int8)
                        policy_inputs = self.trainer.policy.get_formatted_states(
                            policy_inputs)
                        self.new_action = self.trainer.policy.eps_policy_batch(
                            policy_inputs)
                        print "actions {}".format(self.new_action)
                        for i, aircraft in enumerate(self.env.aircrafts):
                            call_sign = aircraft.call_sign
                            action = self.new_action[i]
                            clearance = self._get_clearance(aircraft, action)
                            print "action {}".format(action)
                            print "clearance {}".format(clearance)
                            self.env.set_clearance(aircraft, clearance)

                            # Save the current state, the action picked and the reward
                            # observed at this step (actually related to the
                            # previous step)
                            # reset the reward count
                            self.cumu_reward[call_sign] = 0

                    # Move all aircraft following their clearances
                    rewards, final_aircrafts, ld_count = self.env.step(t_count)
                    total_reward += np.sum(
                        np.array([r for k, r in rewards.iteritems()]))

                    # Store reward
                    for aircraft in self.env.aircrafts:
                        call_sign = aircraft.call_sign
                        cumu_reward += rewards[call_sign]

                    t_count += 1

                # Render and sleep
                if self.render_bool:
                    self.render(t_count, total_reward)
                    pygame.time.wait(self.frame_delay)
            except KeyboardInterrupt:
                self.quit = True
            finally:
                if t_count > self.MAX_T_COUNT:
                    print "Episode Too Long"
                    return total_reward, self.simu_history
                if self.quit or self.env.done:
                    print "Episode duration: {}".format(t_count)
                    pygame.time.wait(500)
                    df = pd.DataFrame({"duration": [t_count],
                                       "land_count": [ld_count],
                                       "reward": [total_reward]})
                    df.to_csv("durations.csv", mode="a", index=False, header=False)

                    return total_reward, self.simu_history

        if self.quit:
            return total_reward, self.simu_history

    def render(self, t_count, total_reward):
        # Clear screen
        self.screen.fill(self.bg_color)
        # Draw Waypoints
        for (k, wp) in self.env.waypoints.items():
            self.screen.blit(self.waypoint_sprites[wp], self.waypoint_pos[wp])

        black = (0, 0, 0)
        margin = 5

        nb_aircraft = len(self.env.aircrafts)
        

        # Draw Action Text Box
        if self.new_action and nb_aircraft > 0:
            text_box_height = margin + nb_aircraft * (self.font_size + margin)
            self.text_box = pygame.draw.rect(
                self.screen, (black),
                (400, 600 - margin - text_box_height, 200 - margin, text_box_height), 2)

            for i, aircraft in enumerate(self.env.aircrafts):
                call_sign = aircraft.call_sign
                action = self.new_action[i]
                text = "{}: {}".format(call_sign, action)
                self.screen.blit(self.font.render(
                    text, True, black),
                    (400 + margin, 600 - text_box_height + i * margin + (i * self.font_size)))

        # Draw Time Text Box
        time_text = "Time: {}s".format(t_count * self.env.delta_t)
        self.screen.blit(self.font.render(time_text, True, black),
                         (2 * margin, 600 - (2 * margin + self.font_size)))
        text_box_height = 2 * margin + self.font_size
        self.text_box = pygame.draw.rect(
            self.screen, (black),
            (margin, 600 - margin - text_box_height, 130, text_box_height), 2)

        # Draw Total Score Text Box
        time_text = "Total Score: {0:.1f}".format(total_reward)
        self.screen.blit(self.font.render(time_text, True, black),
                         (2 * margin, 2 * margin))
        text_box_height = 2 * margin + self.font_size
        self.text_box = pygame.draw.rect(
            self.screen, (black),
            (margin, margin, 180, text_box_height), 2)

        # Draw Aircraft
        for aircraft in self.env.aircrafts:
            # Convert coordinates into pygame coordinates (lower-left => top
            # left)
            agent_pos = (aircraft.xyz_pos[0] * self.position_scale,
                         (self.height - aircraft.xyz_pos[1] * self.position_scale))
            rotated_sprite = pygame.transform.rotate(
                self.aircraft_sprites[aircraft], -aircraft.heading)
            left_coord = agent_pos[0] - self.arcrft_sprite_size[0] / 2
            top_coord = agent_pos[1] - self.arcrft_sprite_size[1] / 2

            rectangle = pygame.rect.Rect(left_coord, top_coord,
                                         self.arcrft_sprite_size[0],
                                         self.arcrft_sprite_size[1])

            self.screen.blit(rotated_sprite, rectangle)
            # self.screen.blit(self.circle_sprites[aircraft], rectangle)

        pygame.display.flip()

    class VizSimulatorQuit(Exception):
        pass


def run_simu(trainer, n_aircraft, frame_buffer_size, verbose=True,
             render_bool=True, frame_delay=0, update_delay=0.01,
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

    if wp_heading_bool:
        wp_heading = random.choice([90, 180, 270, 360])
    else:
        wp_heading = None

    env.create_waypoint(np.array([waypoint_x, waypoint_y, 0]), "WP1", "blue",
                        wp_heading)
    for i in range(n_aircraft):
        env.create_aircraft("RL00{}".format(i))

    if render_bool:
        simulator = VisualSimulator(env, frame_buffer_size, trainer,
                                    frame_delay=frame_delay,
                                    update_delay=update_delay)
    else:
        # Run the simulator without the pygame visualization
        simulator = Simulator(env, frame_buffer_size, trainer)

    simulator.run()

    if not verbose:
        sys.stdout = old_stdout
    return
