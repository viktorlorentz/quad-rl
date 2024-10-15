import unittest
import gym
import bitcraze_crazyflie_2

class TestDroneEnv(unittest.TestCase):
    def test_environment(self):
        env = gym.make('DroneEnv-v0')
        obs = env.reset()
        self.assertEqual(len(obs), 13)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.close()

if __name__ == '__main__':
    unittest.main()
