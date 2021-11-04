"""
AUV dynamical system implemented by Justin Kleiber
"""

from math import fmod
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
        Type: IDK

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
        Episode length is greater than 150

    Solved Requirements:
        TBD
    """
    def __init__(self):
        # Vehicle parameters
        self.mass = 80
        self.buoy = 0.002

        # Environment boundaries
        self.pitch_limit = 0.349
        self.yaw_error_limit = np.math.pi
        self.depth_error_limit = 15
        self.yaw_max = 2*np.math.pi
        self.yaw_min = 0

        # Define the observation space
        self.obs_limits = [
            self.pitch_limit,
            np.finfo(np.float32).max,
            self.yaw_error_limit,
            np.finfo(np.float32).max,
            self.depth_error_limit
        ]
        bound = np.array(self.obs_limits, dtype=np.float32)
        self.observation_space = spaces.Box(-bound, bound, dtype=np.float32)

        # Define the goal state to be 90 deg yaw at 5m depth
        self.goal = np.array([
            0,
            0,
            1.57,
            0,
            5
        ], dtype=np.float32)

        # Define the state dynamics
        self.auv_state = None
        self.error_state = None

    def set_random_seed(self, seed=None):
        self.gen_random = np.random.default_rng(seed)

    def reset(self):
        # Random initial state
        self.auv_state = np.array([
            0,
            0,
            self.gen_random.uniform(low = 0, high = 2*np.math.pi, size=1),
            0,
            self.gen_random.uniform(low = 1, high = 10, size=1)
        ], dtype=np.float32)

        # Get the error state
        self.error_state = self.goal - self.auv_state

        # Wrap the error for yaw
        self.error_state[3] = self.wrap_yaw_error(
                                self.wrap_angle(self.goal[3]), 
                                self.wrap_angle(self.auv_state[3])
                                )

        return self.error_state

    def step(self, action):

        pass

    def render(self):
        pass

    def reward_fn(self):
        return 0

    def set_goal(self, goal):
        pass

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