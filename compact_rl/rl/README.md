# Implementation of RL
 Implementation of reinforcement learning for Storm models using the VecStorm implementation.
 
## Installation
 Install paynt.
 Then use following commands:
 ```shell
   $ source ../prerequisites/venv/bin/activate
   $ pip install libclang
   $ pip install tensorflow==2.15
   $ pip install tf_agents
   $ pip install tqdm dill matplotlib pandas seaborn
 ```

## Used Framework and Sources
 The implementation is primarily based on TensorFlow Agents framework, which implements many important blocks of this project as reinforcement learning algorithms, TF environment interface, policy drivers etc. We also took some inspiration and in case of ./environment/pomdp_builder, we took the code from repository: [Shielding](https://github.com/stevencarrau/safe_RL_POMDPs).

## Examples of Run
 To run some selected model (intercept):
 ```shell
   $ source ../prerequisites/venv/bin/activate
   $ python3 experiment_runner.py --model-condition intercept
 ```

 To run experiments for multiple models from some_directory:
 ```shell
   $ source ../prerequisites/venv/bin/activate
   $ python3 experiment_runner.py --path-to-models some_directory
 ```
 
 If you want to design your own experiments, you should start with:
 ```shell
   $ source ../prerequisites/venv/bin/activate
   $ python3 experiment_runner.py --help
 ```

