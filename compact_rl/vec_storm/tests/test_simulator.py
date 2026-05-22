import os
import tempfile
import paynt.parser.sketch
import numpy as np
from pathlib import Path

from pytest import approx

from compact_rl.vec_storm import StormVecEnv
from compact_rl.vec_storm.simulator import Simulator


AVOID_DET = Path(os.path.abspath(__file__)).parent / 'models/det_avoid'
AVOID_RAND = Path(os.path.abspath(__file__)).parent / 'models/rand_avoid'

def load_pomdp(env_path):
    env_path = os.path.abspath(env_path)
    sketch_path = os.path.join(env_path, "sketch.templ")
    properties_path = os.path.join(env_path, "sketch.props")    
    quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
    return quotient.pomdp


def test_enable_random_init_changes_hash():
    """
        Test that the hash of the simulator changes after enabling random_init.
    """
    pomdp = load_pomdp(AVOID_DET)
    env = StormVecEnv(pomdp, lambda x, y: 0, num_envs=1)
    hash_before = hash(env.simulator)
    env.enable_random_init()
    hash_after = hash(env.simulator)
    assert hash_before != hash_after


def test_disable_random_init_changes_hash():
    """
        Test that the hash of the simulator changes after disabling random_init.
    """
    pomdp = load_pomdp(AVOID_DET)
    env = StormVecEnv(pomdp, lambda x, y: 0, num_envs=1)
    env.enable_random_init()
    hash_before = hash(env.simulator)
    env.disable_random_init()
    hash_after = hash(env.simulator)
    assert hash_before != hash_after


def test_save_load():
    """
        Test that the environment can be saved and loaded.
    """
    pomdp = load_pomdp(AVOID_DET)

    def get_scalarized_reward(rewards, rewards_types):
        return (rewards["costs"] * 0) + 42

    env = StormVecEnv(pomdp, get_scalarized_reward, num_envs=1)

    with tempfile.NamedTemporaryFile() as tmp:
        env.save(tmp.name)
        env2 = StormVecEnv.load(tmp.name)
    
    env2.reset()
    res = env2.step(np.array([0]))
    assert res[1] == approx(42)


def test_get_free_id():
    """
        Test that the simulator IDs are correctly incremented on `get_free_id` calls.
    """
    id1 = Simulator.get_free_id()
    id2 = Simulator.get_free_id()
    assert id1 != id2


def test_envs_have_different_ids():
    """
        Test that the environment IDs are correctly incremented on creation.
    """
    pomdp1 = load_pomdp(AVOID_RAND)
    pomdp2 = load_pomdp(AVOID_DET)
    env1 = StormVecEnv(pomdp1, lambda x, y: 0, num_envs=1)
    env2 = StormVecEnv(pomdp2, lambda x, y: 0, num_envs=1)
    assert env1.simulator.id != env2.simulator.id


def test_envs_have_different_ids_after_load():
    """
        Test that the environment IDs are correctly incremented after loading.
    """
    pomdp = load_pomdp(AVOID_RAND)
    env1 = StormVecEnv(pomdp, lambda x, y: 0, num_envs=1)
    with tempfile.NamedTemporaryFile() as tmp:
        env1.save(tmp.name)
        env2 = StormVecEnv(pomdp, lambda x, y: 0, num_envs=1)
        env3 = StormVecEnv.load(tmp.name)
    assert env1.simulator.id + 1 == env2.simulator.id
    assert env2.simulator.id + 1 == env3.simulator.id


def _get_cost_reward(rewards, rewards_types):
    return rewards["costs"]


def _obs_to_dict(env, obs):
    return {
        key: val for key, val in zip(env.get_observation_labels(), obs)
    }


def test_reset():
    """
        Test that the environment resets correctly when random_init is disabled.
    """
    env = StormVecEnv(load_pomdp(AVOID_DET), _get_cost_reward, num_envs=1)
    obs, act_mask, metalabels = env.reset()
    o = _obs_to_dict(env, obs[0])
    correct_o = {
        "amdone": 0, "cangoeast": 1, "cangonorth": 0, "cangosouth": 1, "cangowest": 0, "hascrash": 0, "start": 0, "x": 0, "y": 0
    }
    assert o == approx(correct_o)


def _test_trajectory(env, actions, expected_observations, expected_rewards, expected_dones, expected_labels):
    obs, act_mask, metalabels = env.reset()
    o = _obs_to_dict(env, obs[0])
    # check that the initial observation is correct
    for key, val in expected_observations.items():
        assert o[key] == approx(val[0])

    # check steps
    for i, action in enumerate(actions):
        assert(act_mask[0][action])

        obs, rew, done, trunc, act_mask, labels = env.step(np.array([action]))
        o = _obs_to_dict(env, obs[0])
        for key, val in expected_observations.items():
            assert o[key] == approx(val[i+1])
        assert rew[0] == approx(expected_rewards[i])
        assert np.all(done[0] == expected_dones[i])
        assert np.all(labels[0] == expected_labels[i])


def test_crash():
    """
        Test that the environment is done after crashing.
    """
    env = StormVecEnv(load_pomdp(AVOID_DET), _get_cost_reward, num_envs=1, metalabels={"avoid": ["traps"], "reach": ["goal"]})
    actions = [2, 0, 3]
    expected_observations = {
        "x": [0, 0, 1, 0],
        "y": [0, 0, 0, 0],
        "hascrash": [0, 0, 0, 0],
        "start": [0, 1, 1, 0],
    }
    expected_rewards = [0, 1, 1]
    expected_dones = [False, False, True]
    expected_labels =np.array([
        [False, False],
        [False, False],
        [True, False],
    ])
    
    _test_trajectory(env, actions, expected_observations, expected_rewards, expected_dones, expected_labels)


def test_goal():
    """
        Test that the environment is done after reaching the goal.
    """
    env = StormVecEnv(load_pomdp(AVOID_DET), _get_cost_reward, num_envs=1, metalabels={"avoid": ["traps"], "reach": ["goal"]})
    actions = [2, 0, 0, 3, 3, 4, 3, 0, 1, 0, 3]
    expected_observations = {
        "x": [0, 0, 1, 2, 2, 2, 1, 1, 2, 2, 3, 0],
        "y": [0, 0, 0, 0, 1, 2, 2, 3, 3, 2, 2, 0],
        "hascrash": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "start": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    }
    expected_rewards = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    expected_dones = [False, False, False, False, False, False, False, False, False, False, True]
    expected_labels =np.array([
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, True],
    ])
    
    _test_trajectory(env, actions, expected_observations, expected_rewards, expected_dones, expected_labels)


def test_truncation():
    """
        Test that the environment is truncated after `max_steps` steps.
    """
    env = StormVecEnv(load_pomdp(AVOID_DET), _get_cost_reward, num_envs=1, metalabels={"avoid": ["traps"], "reach": ["goal"]}, max_steps=3)
    actions = [2, 0, 0]
    expected_observations = {
        "x": [0, 0, 1, 0],
        "y": [0, 0, 0, 0],
        "hascrash": [0, 0, 0, 0],
        "start": [0, 1, 1, 0],
    }
    expected_rewards = [0, 1, 1]
    expected_dones = [False, False, True]
    expected_labels =np.array([
        [False, False],
        [False, False],
        [False, False],
    ])
    
    _test_trajectory(env, actions, expected_observations, expected_rewards, expected_dones, expected_labels)


def test_random_steps():
    """
        Test dynamics of a randomised environment.
    """
    num_envs = 100000
    env = StormVecEnv(load_pomdp(AVOID_RAND), _get_cost_reward, num_envs=num_envs, metalabels={"avoid": ["traps"], "reach": ["goal"]})
    env.reset()
    env.step(np.array([2]*num_envs))

    obs = _obs_to_dict(env, env.step(np.array([0]*num_envs))[0].T)
    assert (obs['x'] == 1).sum() + (obs['x'] == 2).sum() == num_envs
    assert (obs['x'] == 1).mean() == approx(0.7, abs=0.01)
    assert (obs['x'] == 2).mean() == approx(0.3, abs=0.01)

    obs = _obs_to_dict(env, env.step(np.array([0]*num_envs))[0].T)
    assert (obs['x'] == 2).sum() + (obs['x'] == 3).sum() + (obs['x'] == 4).sum() == num_envs
    assert (obs['x'] == 2).mean() == approx(0.49, abs=0.01)
    assert (obs['x'] == 3).mean() == approx(0.42, abs=0.01)
    assert (obs['x'] == 4).mean() == approx(0.09, abs=0.01)


def test_random_after_det():
    """
        Test dynamics of one environment after another is already loaded (test JIT compilation).
    """
    num_envs = 100000
    env = StormVecEnv(load_pomdp(AVOID_DET), _get_cost_reward, num_envs=num_envs, metalabels={"avoid": ["traps"], "reach": ["goal"]})
    env = StormVecEnv(load_pomdp(AVOID_RAND), _get_cost_reward, num_envs=num_envs, metalabels={"avoid": ["traps"], "reach": ["goal"]})
    env.reset()
    env.step(np.array([2]*num_envs))

    obs = _obs_to_dict(env, env.step(np.array([0]*num_envs))[0].T)
    assert (obs['x'] == 1).sum() + (obs['x'] == 2).sum() == num_envs
    assert (obs['x'] == 1).mean() == approx(0.7, abs=0.01)
    assert (obs['x'] == 2).mean() == approx(0.3, abs=0.01)

    obs = _obs_to_dict(env, env.step(np.array([0]*num_envs))[0].T)
    assert (obs['x'] == 2).sum() + (obs['x'] == 3).sum() + (obs['x'] == 4).sum() == num_envs
    assert (obs['x'] == 2).mean() == approx(0.49, abs=0.01)
    assert (obs['x'] == 3).mean() == approx(0.42, abs=0.01)
    assert (obs['x'] == 4).mean() == approx(0.09, abs=0.01)


def test_toggle_random_init():
    """
        Test whether `random_init` can be toggled on and off (test JIT compilation).
    """
    num_envs = 100000
    env = StormVecEnv(load_pomdp(AVOID_DET), _get_cost_reward, num_envs=num_envs)
    env.reset()
    assert np.all(env.simulator_states.vertices == 0)
    env.enable_random_init()
    env.reset()
    assert np.any(env.simulator_states.vertices != 0)
    assert np.any(env.simulator_states.vertices.mean() == approx((env.nr_states-1) / 2, abs=0.1))

    env.disable_random_init()
    env.reset()
    assert np.all(env.simulator_states.vertices == 0)
    env.enable_random_init()
    env.step(np.array([2]*num_envs))
    env.step(np.array([0]*num_envs))
    env.step(np.array([3]*num_envs))
    assert np.all(env.simulator_states.vertices.mean() == approx((env.nr_states-1) / 2, abs=0.1))

    env.disable_random_init()
    env.reset()
    assert np.all(env.simulator_states.vertices == 0)
    env.step(np.array([2]*num_envs))
    env.step(np.array([0]*num_envs))
    env.step(np.array([3]*num_envs))
    assert np.all(env.simulator_states.vertices == 0)

def test_set_states():
    """
        Test that the simulator can set states correctly.
    """
    pomdp = load_pomdp(AVOID_DET)
    env = StormVecEnv(pomdp, _get_cost_reward, num_envs=1)
    env.reset()
    env.set_states(np.array([0]))
    assert np.all(env.simulator_states.vertices == 0)
    env.set_states(np.array([1]))
    assert np.all(env.simulator_states.vertices == 1)
    env.set_num_envs(2)
    env.set_states(np.array([0, 1]))
    assert np.all(env.simulator_states.vertices == np.array([0, 1]))
    env.set_states(np.array([1, 0]))
    assert np.all(env.simulator_states.vertices == np.array([1, 0]))
    assert env.reset()[0].shape == (2, 1)
    env.set_states(np.array([0, 1]))
    assert env.step(np.array([0, 1]))[0].shape == (2, 1)

def test_change_num_envs():
    """
        Test that the number of environments can be changed.
    """
    env = StormVecEnv(load_pomdp(AVOID_DET), _get_cost_reward, num_envs=1)
    env.reset()
    env.step(np.array([2]))
    env.step(np.array([2]))
    env.set_num_envs(2)
    env.reset()
    env.step(np.array([2, 2]))
