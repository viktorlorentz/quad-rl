from setuptools import setup, find_packages

setup(
    name='bitcraze_crazyflie_2',
    version='0.1.0',
    description='A custom Gym environment for the Bitcraze Crazyflie 2 drone using MuJoCo.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/bitcraze_crazyflie_2',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'gym>=0.17.2',
        'numpy>=1.18.0',
        'mujoco-py>=2.0.2.13',
        'stable-baselines3>=1.0',
        'mujoco>=2.2.0'
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
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
)
