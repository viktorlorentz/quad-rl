from setuptools import setup, find_packages

setup(
    name='bitcraze_crazyflie_2',
    version='0.1.0',
    description='A custom Gym environment for the Bitcraze Crazyflie 2 drone using MuJoCo.',
    author='',
    author_email='',
    url='https://github.com/',
    license='MIT',
    packages=find_packages(where='.'),
    install_requires=[
        'gymnasium[other]==1.0.0',
        'mujoco==3.2.3',
        'stable-baselines3[extra]==2.4.0',
        'wandb==0.19.0',
        'scipy==1.10.1',
        'imageio==2.35.1',
        'tensorboard==2.14.0',
        'numpy>=1.25',
        'tqdm==4.67.1',
        'rich==13.9.4',
    ],
    package_data={
        'bitcraze_crazyflie_2': ['assets/*'],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'train-drone=bitcraze_crazyflie_2.scripts.train_agent:main',
            'evaluate-drone=bitcraze_crazyflie_2.scripts.evaluate_agent:main',
        ],
    },
    python_requires='>=3.8',
)
