#!/usr/bin/python
""" Aircraft that evolve in the environment (airspace)"""

import math
import numpy as np


class Aircraft(object):

    def __init__(self, env, call_sign="RL999", waypoint=None):
        # Environement properties
        self.env = env
        self.color = 'black'
        self.call_sign = call_sign
        self.waypoint = waypoint

        # Aircraft Performance
        self.turn_rate = 20  # deg/s
        self.climb_rate = 1000  # ft/s
        self.speed = 150  # knots

        # Initial State
        self.xyz_pos = np.zeros(3)
        self.heading = 0
        self.heading_clearance = None
        self.altitude_clearance = None
        self.clearance_change = False
        self.clearance_change_diff = 0

        print "!!! Aircraft {} Initialized".format(self.call_sign)

    def set_heading_clearance(self, heading_clearance):
        self.heading_clearance = heading_clearance

    def update(self):
        move_success = self.move(self.env.delta_t)
        return move_success

    def move(self, delta_t):
        """
        :param delta_t: time elapsed since the last update
        :return: bool if aircraft did not exit airspace
        """

        # ========== HEADING UPDATE ==========
        # Change heading to toward heading_clearance
        if self.heading_clearance:
            self.heading_clearance %= 360  # make sure heading_clearance in [0,360[
            if self.heading == self.heading_clearance:
                pass

            elif self.heading_clearance > self.heading:
                heading_diff = self.heading_clearance - self.heading
                # potential heading change since last move (during delta_t):
                heading_delta = self.turn_rate * delta_t

                if heading_diff <= 180:
                    if heading_delta >= heading_diff:
                        self.heading = self.heading_clearance  # do not overshoot
                    else:
                        self.heading += heading_delta  # turn right
                elif heading_diff > 180:
                    heading_diff = 360 - heading_diff
                    if heading_delta >= heading_diff:
                        self.heading = heading_delta  # do not overshoot
                    else:
                        self.heading -= heading_delta  # turn left

            elif self.heading_clearance < self.heading:
                heading_diff = self.heading - self.heading_clearance
                # potential heading change since last move (during delta_t):
                heading_delta = self.turn_rate * delta_t
                if heading_diff <= 180:
                    if heading_delta >= heading_diff:
                        self.heading = self.heading_clearance  # do not overshoot
                    else:
                        self.heading -= heading_delta  # turn left
                elif heading_diff > 180:
                    heading_diff = 360 - heading_diff
                    if heading_delta >= heading_diff:
                        self.heading = heading_delta  # do not overshoot
                    else:
                        self.heading += heading_delta  # turn right

        # Heading may have become <0 or >360 during operations above
        # Make self.heading in [0;360]
        self.heading %= 360

        # ========== ALTITUDE UPDATE ==========
        # Change altitude toward altitude clearance
        if self.altitude_clearance:
            # if invalid altitude_clearance
            # do not change altitude
            if (self.altitude_clearance < self.env.MIN_ALTITUDE) or \
               (self.altitude_clearance > self.env.MAX_ALTITUDE):
                self.altitude_clearance = self.xyz_pos[2]

            altitude_diff = self.altitude_clearance - self.xyz_pos[2]

            # potential altitude change since last move (during delta_t):
            altitude_delta = self.climb_rate * delta_t

            if self.altitude_clearance == self.xyz_pos[2]:
                pass
            elif altitude_diff > 0:
                if altitude_delta > altitude_diff:
                    self.xyz_pos[2] = self.altitude_clearance  # do not overshoot
                else:
                    self.xyz_pos[2] += altitude_delta  # climb
            elif altitude_diff < 0:
                if altitude_delta < altitude_diff:
                    self.xyz_pos[2] = self.altitude_clearance  # do not overshoot
                else:
                    self.xyz_pos[2] -= altitude_delta  # descend

        # ========== HORIZONTAL POSITION UPDATE ==========
        # Update (x,y) position according to heading and speed
        # Puts 0 deg heading toward the top of the screen
        heading_rad = math.radians((90 - self.heading) % 360)
        # update x
        self.xyz_pos[0] += math.cos(heading_rad) * self.speed * delta_t * self.env.KT_TO_MPS
        # update y
        self.xyz_pos[1] += math.sin(heading_rad) * self.speed * delta_t * self.env.KT_TO_MPS

        # Check if the new xyz and heading are valid
        success = self.check_valid_state()

        return success

    def check_valid_state(self):
        # Check if the aircraft is still in the airpace boundaries
        valid_state = True
        x, y, z = self.xyz_pos
        if x < 0 or x > self.env.width:
            print "!!! Aircraft Error - {} Invalid x value: {} ({}, " \
                  "{})!!!".format(self.call_sign, x, x, y)
            valid_state = False
        if y < 0 or y > self.env.height + 1:
            print "!!! Aircraft Error - {} Invalid y value: {} ({}, {}" \
                  ")!!!".format(self.call_sign, y, x, y)
            valid_state = False
        if z < self.env.MIN_ALTITUDE or z > self.env.MAX_ALTITUDE:
            print "!!! Aircraft Error - {} Invalid z value: {} !!!".format(
                self.call_sign, z)
            valid_state = False

        if self.heading < 0 or self.heading > 360:
            print "!!! Aircraft Error - {} Invalid heading value: {} " \
                  "!!!".format(self.call_sign, self.heading)
            valid_state = False

        return valid_state

    def get_state(self):
        return {'xyz_pos': self.xyz_pos, 'heading': self.heading}

    def get_dist_to_wp(self, wp_xyz):
        return np.linalg.norm(self.xyz_pos - wp_xyz)
