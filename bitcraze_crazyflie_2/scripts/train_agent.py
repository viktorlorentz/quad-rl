import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


def main():
    env_id = 'DroneEnv-v0'

    num_envs = 32  # Adjusted number of environments
    n_steps = 1024  # Increased n_steps
    total_timesteps_per_update = num_envs * n_steps  # 64 * 1024 = 65,536
    batch_size = 256  # Should be a factor of total_timesteps_per_update

    env = make_vec_env(env_id, n_envs=num_envs, vec_env_cls=SubprocVecEnv, monitor_dir="./logs")
    eval_env = make_vec_env(env_id, n_envs=1, vec_env_cls=SubprocVecEnv, seed=0)


    # Directory to save models and logs
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'ppo_drone')

    # Clear previous checkpoint models
    for file in os.listdir(models_dir):
        if file.startswith('ppo_drone_'):
            os.remove(os.path.join(models_dir, file))



    # Adjust checkpoint and eval frequencies
    checkpoint_callback = CheckpointCallback(
        save_freq=5 * total_timesteps_per_update,  # Save every 5 updates
        save_path=models_dir,
        name_prefix='ppo_drone'
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=models_dir,
        eval_freq=n_steps,  # Evaluate every update
        deterministic=True,
        render=False
    )

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[128,128], vf=[128,128])
    )

    model = PPO(
        'MlpPolicy',
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=3e-4,  
        device='auto',
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs",

    )
    #Defaults:
        # policy: Union[str, Type[ActorCriticPolicy]],
        # env: Union[GymEnv, str],
        # learning_rate: Union[float, Schedule] = 3e-4,
        # n_steps: int = 2048,
        # batch_size: int = 64,
        # n_epochs: int = 10,
        # gamma: float = 0.99,
        # gae_lambda: float = 0.95,
        # clip_range: Union[float, Schedule] = 0.2,
        # clip_range_vf: Union[None, float, Schedule] = None,
        # normalize_advantage: bool = True,
        # ent_coef: float = 0.0,
        # vf_coef: float = 0.5,
        # max_grad_norm: float = 0.5,
        # use_sde: bool = False,
        # sde_sample_freq: int = -1,
        # rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        # rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        # target_kl: Optional[float] = None,
        # stats_window_size: int = 100,
        # tensorboard_log: Optional[str] = None,
        # policy_kwargs: Optional[Dict[str, Any]] = None,
        # verbose: int = 0,
        # seed: Optional[int] = None,
        # device: Union[th.device, str] = "auto",

    time_steps = 30_000_000
    model.learn(
        total_timesteps=time_steps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    model.save(model_path)
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
