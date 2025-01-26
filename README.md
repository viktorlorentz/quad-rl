# Bitcraze Crazyflie 2 Gym Environment

A custom OpenAI Gym environment for the Bitcraze Crazyflie 2 drone using MuJoCo. This environment allows you to train reinforcement learning agents to control a simulated drone using the PPO algorithm provided by Stable Baselines3.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/  TODO
```

### 2. Create and Activate the Conda Environment

```bash
conda create -n quad-rl python=3.10
conda activate quad-rl
```

Alternative with micromamba:
  
  ```bash
  micromamba create -n quad-rl python=3.10
  micromamba activate quad-rl
  ```

### 3. Install the Package and Dependencies


Install the package in editable mode and the required dependencies
```bash
pip install -e .
```

# Login to Weight & Biases
```bash
wandb login
```


## Usage

### Training the Agent

Run the training script to train the PPO agent:

```bash
train-drone
```
On headless servers, you have to set mujoco headless
```bash
export MUJOCO_GL=egl
```

**Notes:**

- The training script uses Stable Baselines3's PPO implementation.
- Models and logs are saved in the `models/` directory.
- You can adjust hyperparameters in the `train_agent.py` script if needed.

### Evaluating the Trained Agent

After training, you can evaluate the agent:

```bash
evaluate-drone
```

This will run the trained agent in the environment and render its behavior.

## Docker

To build the Docker image, navigate to the project root directory and run:

```bash
docker build -t quad-rl-image .
```

To run the Docker container with the Wandb api key, use the following command:

```bash
docker run -it -e WANDB_API_KEY=your_api_key_here quad-rl-image
```

Replace `your_api_key_here` with your actual Weights & Biases API key.



### Monitoring Training with TensorBoard

To monitor training progress, start TensorBoard:

```bash
tensorboard --logdir=models/
```

Open the provided URL (usually `http://localhost:6006`) in your web browser to view training metrics such as rewards and losses.

## Important Notes

- **MuJoCo License:** Ensure you have a valid MuJoCo license and that MuJoCo is properly installed.
- **Dependencies:** All necessary dependencies are listed in `requirements.txt`. Install them using:

  ```bash
  pip install -r requirements.txt
  ```

- **Python Version:** The project is designed for Python 3.8, as specified in the Conda environment creation command.
- **Environment Registration:** The custom environment `DroneEnv-v0` is registered upon importing the `bitcraze_crazyflie_2` package.

## Customizing the Environment

- **Observation Space:** The environment provides observations including position, orientation, linear velocity, and angular velocity.
- **Action Space:** The action space consists of the thrust inputs for the four motors.
- **Reward Function:** The default reward function penalizes the Euclidean distance from the target position (hovering at 1.0 meter altitude). You can modify the reward function in `drone_env.py` to suit your needs.
- **Simulation Parameters:** Adjust `self.simulation_steps` in `drone_env.py` to control the number of simulation steps per environment step.

## Testing the Environment

You can run unit tests to verify that the environment is working correctly:

```bash
python -m unittest discover tests
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact Information

For any questions or support, please contact:

- **Author:** Viktor Lorentz
- **Email:** lorentz@campus.tu-berlin.de