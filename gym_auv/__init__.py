from gym.envs.registration import register

register(
    id='AUVControl-v0',
    entry_point='gym_auv.envs:AUVControlEnv',
)

register(
    id='AUVDepthControl-v0',
    entry_point='gym_auv.envs:AUVDepthControlEnv',
)

register(
    id='AUVDepthControlGame-v0',
    entry_point='gym_auv.envs:AUVDepthControlGameEnv',
)