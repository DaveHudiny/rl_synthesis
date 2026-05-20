import os
import time

import numpy as np

import paynt.parser.sketch

from compact_rl.vec_storm import StormVecEnv


import logging
logging.basicConfig(level=logging.INFO)


def benchmark(env):
    for l in range(15):
        r = 2000
        n = 2**l

        if n > 1000:
            r //= 2**(l-9)

        env.set_num_envs(n)

        start = time.time()
        env.reset()
        env.step(np.array([2]*n))
        for i in range(r):
            env.step(np.array([0]*n))
            env.step(np.array([5]*n))
            env.step(np.array([3]*n))
            env.step(np.array([0]*n))
            env.step(np.array([0]*n))
            env.step(np.array([0]*n))


        end = time.time()

        print(f"{n=}", end=":\t")
        print(f"Time: {end-start:.3f}", end=",\t")
        print(f"Steps s^-1: { r / (end-start):.3f}", end=",\t")
        print(f"EnvSteps s^-1: {n * r / (end-start):.3f}")


if __name__ == '__main__':
    def load_pomdp(env_path):
        env_path = os.path.abspath(env_path)
        sketch_path = os.path.join(env_path, "sketch.templ")
        properties_path = os.path.join(env_path, "sketch.props")    
        quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
        return quotient.pomdp
    
    model_path = "/opt/learning/synthesis/rl_src/models_large/network-5-10-8"
    # model_path = "/opt/learning/synthesis/rl_src/models/network-3-8-20"
    # model_path = "/opt/learning/synthesis/rl_src/models/refuel-10"
    pomdp = load_pomdp(model_path)

    def get_scalarized_reward(rewards, rewards_types):
        last_reward = rewards_types[-1]
        # return 100*rewards["refuels"] + 10*rewards["costs"] + rewards["steps"] + 1000*rewards[last_reward]
        return rewards[last_reward]

    env = StormVecEnv(pomdp, get_scalarized_reward)
    benchmark(env)
