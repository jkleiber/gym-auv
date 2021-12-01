"""
AUV dynamical system implemented by Justin Kleiber
"""

from math import cos, fmod
import gym
from gym import spaces

import numpy as np

class AUVControlEnv(gym.Env):
    """
    Description:
        An AUV starts underwater and moves through the water at a constant
        speed. The vehicle starts at level flight at a depth and the goal
        is to control the vehicle to some target yaw angle and depth.

    Observation:
        Type: Box(5)
        Num     Observation     Min                 Max
        0       Pitch           -0.349 (-20 deg)    0.349 (20 deg)
        1       Pitch Rate      -Inf                Inf
        2       Yaw Error       -3.14 (-180 deg)    3.14 (180 deg)
        3       Yaw Rate        -Inf                Inf
        4       Depth Error     -15                 15

    Actions:
        Type: Box(2)
        Num     Action              Min                 Max
        0       Elevator Angle      -0.349 (20 deg)     0.349 (20 deg)
        1       Rudder Angle        -0.349 (20 deg)     0.349 (20 deg)

    Reward:
        Reward is R = -1*yaw_error^2 - 1*depth_error^2 for each step. Thus being
        closer to the target yaw and depth yields more reward.

    Starting State:
        Depth is randomly chosen between [1, 10) meters
        Yaw is randomly chosen between [0, 2*pi) radians
        Pitch starts at 0 radians
        Pitch and yaw rate start at 0 radians / sec

    Episode Termination:
        Pitch exceeds +/- 20 degrees.
        Depth <= 0 meters
        Depth >= 15 meters

    Solved Requirements:
        TBD
    """
    def __init__(self):
        # Vehicle parameters
        self.mass = 80
        self.buoy = 0.002

        # Simulation parameters
        self.dt = 0.1
        self.target_speed = 2 # m/s

        # Reward parameters
        self.early_stop_penalty = -10000

        # Environment boundaries
        self.pitch_limit = 0.349
        self.yaw_error_limit = np.math.pi
        self.depth_limit = 15
        self.yaw_max = 2*np.math.pi
        self.yaw_min = 0

        # Define the observation space
        self.obs_limits = [
            self.pitch_limit,
            np.finfo(np.float32).max,
            self.yaw_error_limit,
            np.finfo(np.float32).max,
            self.depth_limit
        ]
        bound = np.array(self.obs_limits, dtype=np.float32)
        self.observation_space = spaces.Box(-bound, bound, dtype=np.float32)

        # Define action space
        self.max_fin_angle = 0.349
        self.action_space = spaces.Box(
            low = -1*self.max_fin_angle, high = self.max_fin_angle, shape = (2,), dtype = np.float32
            )

        # Define the goal state to be 90 deg yaw at 5m depth
        self.goal = np.array([
            0,
            0,
            1.57,
            0,
            5
        ], dtype=np.float32)

        # Set the random seed for the environment
        self.set_random_seed()

        # Define the state dynamics
        self.full_auv_state = None
        self.auv_state = None
        self.error_state = None

    def set_random_seed(self, seed=None):
        self.gen_random = np.random.default_rng(seed)

    def reset(self):
        # Random initial state
        self.full_auv_state = np.array([
            0,
            0,
            0,
            0,
            self.gen_random.uniform(low = 0, high = 2*np.math.pi, size=1),
            0,
            self.gen_random.uniform(low = 1, high = 10, size=1)
        ], dtype=np.float32)

        # Update the AUV state and error state
        self.update_auv_state()

        return self.error_state

    def step(self, action):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        # Compute the state update equations
        self.full_auv_state = self.dynamics(action)

        # State variables
        depth = self.full_auv_state[6]
        pitch = self.full_auv_state[1]

        # Update the AUV state and error state
        self.update_auv_state()

        # Compute the reward
        reward = self.reward_fn()

        # Is the simulation done?
        done = False

        # Done if the vehicle re-surfaces or violates the max depth
        if depth <= 0 or depth > self.depth_limit:
            done = True
        # Done if the pitch exceeds the limits
        elif pitch > self.pitch_limit or pitch < -self.pitch_limit:
            done = True

        # If the simulation ends early, then the reward is changed to a large penalty
        if done:
            reward = self.early_stop_penalty

        return self.error_state, reward, done, {}

    def render(self):
        pass

    def reward_fn(self):
        return -1 * np.sqrt(self.error_state.dot(self.error_state))

    def set_goal(self, goal):
        self.goal = goal

    def dynamics(self, u):
        # State vector
        # full_auv_state: [heave; pitch; pitch rate; sway; yaw; yaw rate; depth]
        x = self.full_auv_state[0:6]
        depth = self.full_auv_state[6]

        # Linear System for Pitch and Yaw
        # x states are [heave; pitch; pitch rate; sway; yaw; yaw rate]
        A = np.array(
            [
            [  0.96, 0, 0.03,      0, 0,     0],
            [-0.004, 1,  0.1,      0, 0,     0],
            [ -0.09, 0,    1,      0, 0,     0],
            [     0, 0,    0,      1, 0, 0.017],
            [     0, 0,    0, -0.007, 1,   0.1],
            [     0, 0,    0,  -0.14, 0,   0.9]
        ], dtype=np.float32)

        B = np.array(
            [
            [0.007,      0],
            [0.001,      0],
            [0.026,      0],
            [    0,  0.007],
            [    0, -0.001],
            [    0, -0.036]
        ], dtype=np.float32)

        # Linear model update
        xp1 = (A@x) + (B@u)

        # Update Depth (nonlinear)
        depth = depth - self.buoy*self.mass*self.dt + (x[0]*np.cos(x[1]) - self.target_speed*np.sin(x[1]))*self.dt

        # Return the full dynamics
        new_state = np.append(xp1,depth)
        return new_state

    def update_auv_state(self):
        # update the open AI gym auv state from the full state
        self.auv_state = self.full_auv_state[[1,2,4,5,6]]

        # Get the error state
        self.error_state = self.goal - self.auv_state

        # Wrap the error for yaw
        self.error_state[2] = self.wrap_yaw_error(
                                self.wrap_angle(self.goal[2]),
                                self.wrap_angle(self.auv_state[2])
                                )

    def wrap_angle(self, angle):
        assert self.yaw_max > self.yaw_min
        return fmod(angle, self.yaw_max - self.yaw_min)

    def wrap_yaw_error(self, goal_yaw, yaw):
        # It is assumed that yaw and goal yaw are in the range [0, 2pi) radians
        # Find the yaw error
        yaw_error = goal_yaw - yaw

        # Wrap if the error exceeds +/- pi by finding the shorter path around
        # the circle
        if yaw_error > np.math.pi:
            return (goal_yaw - self.yaw_max) + (self.yaw_min - yaw)
        elif yaw_error < -np.math.pi:
            return (goal_yaw - self.yaw_min) + (self.yaw_max - yaw)

        return yaw_error