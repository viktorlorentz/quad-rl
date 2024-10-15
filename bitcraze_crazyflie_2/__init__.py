from gym.envs.registration import register

register(
    id='DroneEnv-v0',
    entry_point='bitcraze_crazyflie_2.envs:DroneEnv',
)
