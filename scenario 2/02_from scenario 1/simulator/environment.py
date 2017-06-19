#!/usr/bin/python
""" Environment where the aircraft and the waypoints interact"""

import math
import numpy as np
import random
from copy import copy
from aircraft import Aircraft


class Environment(object):
    """
    Contains the aircraft and implements the airspace rules
    """

    aircraft_safety_dist = 100  # minimum distance between aircraft else crash
    # radius from the waypoint used to determine whether the aircraft reached
    # the wp
    wp_proximity_threshold = 100
    angular_margin = 20
    MAX_ALTITUDE = 50000  # maximum airspace altitude (ft)
    MIN_ALTITUDE = 0  # minimum airspace altitude (ft)
    done = False  # Major event terminates the simulation
    delta_t = 0.1  # Position refresh time interval (seconds)
    KT_TO_MPS = 0.5144  # Knot to m/s conversion

    # Environment state grid mesh size should be small enough so
    # aicraft appear on a different grid cell (if it moves along one grid axis)
    # at each frame perceived by the agent
    # for instance if aircraft.speed = 150 kt or 77.2 m/s
    # and the frame rate is 3 Hz then the mesh_size should be < 25.7 m
    mesh_size = 25

    def __init__(self, width, height):
        self.width = width + 0.  # Airspace width (meters)
        self.height = height + 0.  # Airspace height (meters)
        self.grid_width = int(self.width / self.mesh_size)
        self.grid_height = int(self.height / self.mesh_size)
        self.state_grid = np.zeros(
            (self.grid_width, self.grid_height), dtype=np.int8)

        # numpy array of xyz positions
        self.aircrafts = []
        self.waypoints = {}

    def reset(self):
        self.state_grid = np.zeros(
            (self.grid_width, self.grid_height), dtype=np.int8)
        self.aircrafts = []
        self.waypoints = {}
        self.done = False

    class CallSignCollision(Exception):
        pass

    def create_aircraft(self, call_sign):
        """
        Create an aircraft and assign its initial position, add to env.aircrafts
        and update the state grid
        :param call_sign: aircraft call sign
        :return: the initialized aircraft
        """
        existing_call_signs = [ac.call_sign for ac in self.aircrafts]
        if call_sign in existing_call_signs:
            raise self.CallSignCollision("Call signs must be unique")

        acrft = Aircraft(self, call_sign)
        acrft.call_sign = call_sign

        acrft.xyz_pos[:2], acrft.heading = self._init_side_pos_heading(
            3*self.aircraft_safety_dist)

        wp = random.choice(self.waypoints.values())
        acrft.waypoint = wp
        acrft.color = wp.color

        self.aircrafts.append(acrft)

        self.update_state_grid(acrft)

        return acrft

    def _init_side_pos_heading(self, init_buffer):
        # try new positions until no conflict with other aircraft
        # Valid heading initialization
        valid_headings = range(0, 360)
        conflict = True
        while conflict:
            conflict = False
            position = []
            heading = 0
            headings = []
            self.angular_margin = 20
            spatial_margin = 225
            # side of the airspace where aircraft is initialized
            side = random.choice(["left", "right", "top", "down"])

            if side == "left":
                position = [0, random.randrange(
                    0 + spatial_margin, self.height - spatial_margin)]
                headings = [head for head in valid_headings if (
                    0 + self.angular_margin) < head < (180 - self.angular_margin)]

                heading = random.choice(headings)
            elif side == "right":
                position = [self.width, random.randrange(
                    0 + spatial_margin, self.height - spatial_margin)]
                headings = [head for head in valid_headings if (
                    180 + self.angular_margin) < head < (360 - self.angular_margin)]
                heading = random.choice(headings)
            elif side == "top":
                position = [random.randrange(
                    0 + spatial_margin, self.width - spatial_margin), self.height]
                headings = [head for head in valid_headings if (
                    90 + self.angular_margin) < head < (270 - self.angular_margin)]
                heading = random.choice(headings)
            elif side == "down":
                position = [random.randrange(
                    0 + spatial_margin, self.width - spatial_margin), 0]
                headings = [head for head in valid_headings if (0 < head < (90 - self.angular_margin)) or
                            ((270 + self.angular_margin) < head < 360)]
                heading = random.choice(headings)
            else:
                print "!!!!! ERROR _init_side_pos_heading ERROR !!!!!"

            for ac in self.aircrafts:
                other_aircraft_pos = ac.xyz_pos[:2]
                dist = np.linalg.norm(other_aircraft_pos - position)
                if dist <= init_buffer:  # aircraft too close to another, retry
                    conflict = True
        print "Initial position {} heading {}".format(position, heading)
        return position, heading

    def remove_aircraft(self, removed_acrft, prev_coord=None):
        """
        Remove aircraft from env.aircrafts and env.state_grid
        and delete the aircraft object
        :param removed_acrft
        :param prev_coord
        :return:
        """
        print "Remove {}".format(removed_acrft.call_sign)
        self._aircraft_grid_remove(removed_acrft, prev_coord)

        self.aircrafts.remove(removed_acrft)
        del removed_acrft

    def set_clearance(self, aircraft, clearance):
        selected_aircraft = aircraft
        previous_clearance = selected_aircraft.heading_clearance

        if previous_clearance:
            diff = clearance - previous_clearance
            diff = np.abs((diff + 180) % 360 - 180)
        else:
            diff = 0
        selected_aircraft.clearance_change_diff = np.abs(diff)
        selected_aircraft.clearance_change = (clearance != previous_clearance)
        # df = pd.DataFrame({"diff": [diff]})
        # df.to_csv("diff.csv", mode="a", index=False, header=False)
        selected_aircraft.set_heading_clearance(clearance)

    def _heading_from_vec(self, vec):
        return np.angle(vec[0] + vec[1] * 1j, deg=True)

    def step(self, t_count):
        """
        Move aircraft according to heading and check if any crash
        :param t_count: number of time steps since start of simulation
        """
        rewards = {}  # total reward for all aircraft
        # aircraft that reached end of simu (landing/exit/crash)
        final_aircrafts = []
        landing_count = 0  # number of aircraft that landed during this step
        # Update the position of aircraft one time step forward
        for acrft in copy(self.aircrafts):
            # we may remove aircraft of env.aircrafts during this loop
            # this is why we need to iterate on a copy of the list
            call_sign = acrft.call_sign
            # save the previous grid coordinates
            prev_coord = list(
                (acrft.xyz_pos[:2, ] / self.mesh_size).astype(int) - 1)
            move_success = acrft.update()
            if move_success:  # aircraft did not exit airspace
                self.update_state_grid(acrft, prev_coord)
                landed, r = self.move_reward(acrft)
                rewards[call_sign] = r
                if landed:
                    print "{} Successful landing!!!".format(acrft.call_sign)
                    final_aircrafts.append(call_sign)
                    self.remove_aircraft(acrft, prev_coord)
                    landing_count += 1
            else:
                rewards[call_sign] = -10
                final_aircrafts.append(call_sign)
                self.remove_aircraft(acrft, prev_coord)

        if len(self.aircrafts) == 0:
            print "No More Aircraft"
            self.done = True
            return rewards, final_aircrafts, landing_count

        # Check if a crash happened
        crash_list = self._crash_check()
        # If crash, end the simulation for all aircraft
        if crash_list:
            print "CRASH!!!!!!!!!"
            for acrft in copy(self.aircrafts):
                call_sign = acrft.call_sign
                if call_sign in crash_list:
                    print "remove {}".format(call_sign)
                    rewards[call_sign] = -10
                    final_aircrafts.append(call_sign)
                    self.remove_aircraft(acrft, prev_coord)

        return rewards, final_aircrafts, landing_count

    def move_reward(self, acrft):
        landed = False
        wp = acrft.waypoint
        dist_to_wp = acrft.get_dist_to_wp(wp.xyz_pos)
        close_to_wp = dist_to_wp < self.wp_proximity_threshold
        if close_to_wp:
            if wp.heading:
                angular_diff = acrft.heading - wp.heading
                angular_diff = abs((angular_diff + 180) % 360 - 180)
                if angular_diff < self.angular_margin:
                    landed = True
            else:
                landed = True

        if landed:
            reward = 10
        else:
            reward = 0
        if acrft.clearance_change:
            reward += -.05 * acrft.clearance_change_diff / \
                30.  # penalize frequent large clearance changes
            acrft.clearance_change = False
        return landed, reward

    def _crash_check(self):
        """
        Check if the distance between aircraft > aircraft_safety_dist
        :return: whether some aircraft crashed into each other (True if crash)
        """
        n_aircraft = len(self.aircrafts)
        crash_list = []
        for i in range(n_aircraft):
            pos_i = self.aircrafts[i].xyz_pos[:2]
            for j in range(i + 1, n_aircraft):
                pos_j = self.aircrafts[j].xyz_pos[:2]
                dist = np.linalg.norm(pos_i - pos_j)
                if dist < self.aircraft_safety_dist:
                    crash_list.append(self.aircrafts[i].call_sign)
                    crash_list.append(self.aircrafts[j].call_sign)
        # Remove duplicates
        crash_list = list(set(crash_list))
        return crash_list

    # (grid_width, grid_height) matrix representing the position of the elements
    # of the environment
    # Pos 0: aircraft current heading value
    # Pos 1: waypoint indicator (1 if present)

    def get_state_dimension(self):
        return self.grid_width, self.grid_height

    def update_state_grid(self, acrft, prev_coord=None):
        if prev_coord:  # remove the previous position
            self.state_grid[prev_coord[0], prev_coord[1]] = 0

        x_coord, y_coord = (acrft.xyz_pos[:2, ] / self.mesh_size).astype(
            int) - 1
        self.state_grid[x_coord, y_coord] = 3
        return None

    def _aircraft_grid_remove(self, acrft, prev_coord=None):
        if prev_coord:
            x_coord, y_coord = prev_coord
        else:
            x_coord, y_coord = (
                acrft.xyz_pos[:2, ] / self.mesh_size).astype(int) - 1
        self.state_grid[x_coord, y_coord] = 0
        return None

    def _waypoint_grid_update(self, wp):
        x_coord, y_coord = (wp.xyz_pos[:2, ] / self.mesh_size).astype(int) - 1
        if wp.heading:
            rad_heading = math.radians(wp.heading)
            x_direction = int(round(math.cos(rad_heading)))
            y_direction = int(round(math.sin(rad_heading)))
            self.state_grid[x_coord - x_direction, y_coord - y_direction] = -1  # waypoint entrance indicator
            self.state_grid[x_coord + x_direction, y_coord + y_direction] = -3  # waypoint exit indicator
            self.state_grid[x_coord, y_coord] = -2  # waypoint middle indicator
        else:
            self.state_grid[x_coord, y_coord] = 1  # waypoint indicator
        return None

    def get_state_grid(self, acrft):
        # return the state grid from the point of view of the agent
        # replace aircraft position with value "2" then revert to "3"
        x_coord, y_coord = (acrft.xyz_pos[:2, ] / self.mesh_size).astype(
            int) - 1
        self.state_grid[x_coord, y_coord] = 2
        result = copy(self.state_grid)
        self.state_grid[x_coord, y_coord] = 3
        return result

    def remove_all_aircraft(self):
        for acrft in self.aircrafts:
            self.remove_aircraft(acrft)
        del self.aircrafts

    class Waypoint:
        def __init__(self, wp_xyz, wp_name, color="black", heading=None):
            self.name = wp_name
            self.color = color
            self.xyz_pos = wp_xyz
            self.heading = heading

    def create_waypoint(self, wp_xyz, wp_name, color, heading=None):
        wp = self.Waypoint(wp_xyz, wp_name, color, heading)
        self.waypoints[wp.name] = wp
        self._waypoint_grid_update(wp)
        return wp
