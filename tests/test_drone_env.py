import unittest
import gym
import numpy as np
import bitcraze_crazyflie_2

class TestDroneEnv(unittest.TestCase):
    def test_environment(self):
        env = gym.make('DroneEnv-v0')
        
        # Unpack the observation and info from reset
        obs, info = env.reset()
        self.assertEqual(len(obs), 13)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (13,))
        self.assertEqual(obs.dtype, np.float32)
        
        action = env.action_space.sample()
        self.assertTrue(env.action_space.contains(action))
        
        # Unpack the step return values
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(len(obs), 13)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (13,))
        self.assertEqual(obs.dtype, np.float32)
        
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        
        env.close()

if __name__ == '__main__':
    unittest.main()
