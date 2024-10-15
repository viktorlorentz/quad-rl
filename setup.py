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
        'gymnasium>=0.29',
        'numpy>=1.18',
        'stable-baselines3>=1.7',
        'mujoco>=2.2',
        'tensorboard>=2'
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
