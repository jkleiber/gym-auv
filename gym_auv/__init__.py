from gym.envs.registration import register

register(
    id='AUVControl-v0',
    entry_point='gym_auv.envs:AUVControlEnv',
)