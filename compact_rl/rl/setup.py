from setuptools import setup, find_packages

setup(
    name='rl_src',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        'tensorflow==2.15.0',
        'numpy',
        'matplotlib',
        'tf_agents==0.19.0'

    ],
    description='A reinforcement learning project for experiments with Storm models.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)