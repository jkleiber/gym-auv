"""
AUV dynamical system implemented by Justin Kleiber
"""

from math import cos, fmod, radians
import gym
from gym import spaces

import numpy as np

class AUVDepthControlGameEnv(gym.Env):
    """
    Description:
        An AUV starts underwater and moves through the water at a constant
        speed. The vehicle starts at level flight at a depth and the goal
        is to control the vehicle to some target yaw angle and depth.

    Observation:
        Type: Box(3)
        Num     Observation     Min                 Max
        0       Pitch           (-30 deg)           (30 deg)
        1       Pitch Rate      -Inf                Inf
        2       Depth Error     -15                 15

    Actions:
        Type: Box(1)
        Num     Action              Min                 Max
        0       Elevator Angle      -0.349 (20 deg)     0.349 (20 deg)

    Reward:
        Reward is the negative squared error for each step. Thus being
        closer to the target yaw and depth yields more reward.

        If the vehicle exits the constraints early, then there is a large
        penalty.

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
        self.mass = 100
        self.buoy = 0.001

        # Simulation parameters
        self.dt = 0.1 # sec
        self.target_speed = 2 # m/s

        # Reward parameters
        self.early_stop_penalty = 0 #-10000

        # Environment boundaries
        self.pitch_limit = radians(30)
        self.yaw_error_limit = np.math.pi
        self.depth_limit = 2
        self.yaw_max = 2*np.math.pi
        self.yaw_min = 0

        # Define the observation space
        self.obs_limits = [
            self.pitch_limit,
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

        # Define the goal state to be at 7m depth
        self.goal = np.array([
            0,
            0,
            1
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
            self.gen_random.uniform(low = -self.pitch_limit/4.0, high = self.pitch_limit/4.0, size=1),
            0,
            self.gen_random.uniform(low = 0.5, high = 1.5, size=1)
        ], dtype=np.float32)

        # Update the AUV state and error state
        self.update_auv_state()

        return self.error_state

    def step(self, action):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        # Compute the state update equations
        # full_auv_state: [heave; pitch; pitch rate; depth]
        self.full_auv_state = self.dynamics(action)

        # State variables
        depth = self.full_auv_state[3]
        pitch = self.full_auv_state[1]

        # Update the AUV state and error state
        self.update_auv_state()

        # Compute the reward
        reward = self.reward_fn()

        # Is the simulation done?
        done = False

        # Done if the vehicle re-surfaces or violates the max depth
        if depth <= 0 or depth > self.depth_limit:
            print(f"Depth limit violated: {depth} meters")
            done = True
        # Done if the pitch exceeds the limits
        elif pitch > self.pitch_limit or pitch < -self.pitch_limit:
            print(f"Pitch limit violated: {pitch} radians")
            done = True

        # If the simulation ends early, then the reward is changed to a large penalty
        if done:
            reward = -10000

        return self.error_state, reward, done, {}

    def render(self):
        pass

    def reward_fn(self):
        Q = np.array([
            [1,   0,   0],
            [0, 0.1,   0],
            [0,   0, 100]
        ])
        return -1 * self.error_state.dot(Q.dot(self.error_state))

    def set_goal(self, goal):
        self.goal = goal

    def dynamics(self, u):
        # State vector
        # full_auv_state: [heave; pitch; pitch rate; depth]
        x = self.full_auv_state[0:3, None]
        depth = self.full_auv_state[3]

        # Linear System for Pitch and Yaw
        # x states are [heave; pitch; pitch rate]
        A = np.array([
            [  0.95, 0, 0.03],
            [-0.004, 1,  0.1],
            [ -0.09, 0,    1]
        ], dtype=np.float32)

        B = np.array(
            [
            [0.007],
            [0.001],
            [0.026]
        ], dtype=np.float32)

        # Limit control
        u = self.clamp(u, -self.max_fin_angle, self.max_fin_angle)

        # Linear model update
        xp1 = (A@x) + (B*u)

        # Update Depth (nonlinear)
        depth = depth - self.buoy*self.mass*self.dt + (x[0]*np.cos(x[1]) - self.target_speed*np.sin(x[1]))*self.dt

        # Return the full dynamics
        new_state = np.append(xp1,depth)
        return new_state

    def update_auv_state(self):
        # update the open AI gym auv state from the full state
        self.auv_state = self.full_auv_state[[1,2,3]]

        # Get the error state
        self.error_state = self.auv_state - self.goal

    def clamp(self, x, x_min, x_max):
        return max(min(x, x_max), x_min)