"""Train a learned risk-budget function via PPO on `RiskBudgetTrainingEnv`: a fixed,
already-loaded policy is treated as the thing being shielded, and the RL action is the
risk-budget allocation itself (see risk_budget_training_env.py's docstring for how the
per-step reward `-D` realizes the intervention-cost objective
    J_gamma(b; pi) = E_tau[sum_t gamma^t * D(pi(tau(t)), shield_b(pi, tau(t)))]
via time_step.discount=gamma and letting PPO/GAE's standard discounted-return machinery
handle the rest).
"""
import argparse
import os

import numpy as np
import stormpy
import tensorflow as tf
from keras.optimizers import Adam
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from compact_rl.robust_rl.robust_rl_tools import load_sketch
from compact_rl.rl.agents.recurrent_ppo_agent import Recurrent_PPO_agent
from compact_rl.rl.agents.tf_agents_modif import ppo_agent
from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec
from compact_rl.rl.environment.tf_py_environment import TFPyEnvironment
from compact_rl.rl.shielding.model_info import ModelInfo
from compact_rl.rl.shielding.risk_budget_networks import RiskBudgetActorNetwork, RiskBudgetValueNetwork
from compact_rl.rl.shielding.risk_budget_training_env import RiskBudgetTrainingEnv
from compact_rl.rl.tests.general_test_tools import init_args


def build_model_info(model, bad_state="bad"):
    assert model.nr_states == model.nr_observations
    components = stormpy.SparseModelComponents(transition_matrix=model.transition_matrix,
                                                reward_models=model.reward_models, state_labeling=model.labeling)
    components.choice_labeling = model.choice_labeling
    if model.has_state_valuations():
        components.state_valuations = model.state_valuations
    if model.has_choice_origins():
        components.choice_origins = model.choice_origins
    mdp = stormpy.storage.SparseMdp(components)
    min_result = stormpy.model_checking(mdp, stormpy.parse_properties(f'Pmin=? [ F "{bad_state}" ]')[0])
    max_result = stormpy.model_checking(mdp, stormpy.parse_properties(f'Pmax=? [ F "{bad_state}" ]')[0])
    observation_to_state = [None] * model.nr_observations
    for state in range(model.nr_states):
        observation_to_state[model.get_observation(state)] = state
    return ModelInfo(model=model, observation_to_state=observation_to_state, bad_state=bad_state,
                      vmin=min_result.get_values(), vmax=max_result.get_values())


def build_shielded_policy(environment, tf_env, args, agent_folder):
    """Loads the fixed policy being shielded, in stochastic mode with real (unmasked)
    logits exposed - the same setup used throughout this session's shielding.py runs."""
    agent = Recurrent_PPO_agent(environment=environment, tf_environment=tf_env, args=args, load=True,
                                 agent_folder=agent_folder)
    policy = agent.get_policy(False, True)
    policy.set_greedy(False)
    policy.set_policy_masker()
    policy.set_return_real_logits(True)
    return policy


def build_agent(train_env, tf_train_env, learning_rate=8.6e-4, num_epochs=3, normalize_rewards=True,
                 entropy_regularization=0.0, actor_net=None, value_net=None):
    state_feature_dim = train_env.observation_spec()["state_features"].shape[0]
    if actor_net is None:
        actor_net = RiskBudgetActorNetwork(tf_train_env.observation_spec(), train_env.max_pairs,
                                            train_env.max_actions, state_feature_dim)
    if value_net is None:
        value_net = RiskBudgetValueNetwork(tf_train_env.observation_spec(), train_env.max_actions, state_feature_dim)
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.99, beta_2=0.99, weight_decay=0.0001)
    train_step_counter = tf.Variable(0)
    agent = ppo_agent.PPOAgent(
        tf_train_env.time_step_spec(),
        tf_train_env.action_spec(),
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=num_epochs,
        train_step_counter=train_step_counter,
        # Not the repo's usual args.discount_factor: RiskBudgetTrainingEnv already emits
        # time_step.discount=gamma per non-terminal step (0 on reset) to directly realize
        # J_gamma's gamma^t weighting. PPOAgent multiplies discount_factor into
        # time_step.discount for the return/advantage computation (tf_agents_modif/
        # ppo_agent.py:652-653), so leaving this at the usual <1 default would
        # double-discount; 1.0 makes it a pure passthrough of the env's own discount.
        discount_factor=1.0,
        use_gae=True,
        lambda_value=0.95,
        # Not the repo's usual 0.02: entropy of a Normal grows without bound as its stddev
        # grows (unlike a bounded/categorical action space, where entropy saturates), so any
        # positive entropy bonus here gives the optimizer a standing incentive to inflate std
        # indefinitely - confirmed empirically (std climbed monotonically every iteration,
        # 0.35 -> 0.39 over 80 iterations, alongside mean intervention cost getting WORSE, not
        # better). 0.0 is also upstream PPOAgent's own default (Schulman 2017 used none).
        entropy_regularization=entropy_regularization,
        # Must stay off: the observation dict includes pair_mask, a bool tensor.
        # TensorNormalizer normalizes and casts back to spec.dtype across the WHOLE nest
        # with no per-leaf opt-out - casting a normalized float back to bool would make
        # padding slots spuriously read as valid almost everywhere.
        normalize_observations=False,
        normalize_rewards=normalize_rewards,
        importance_ratio_clipping=0.2,
    )
    agent.initialize()
    return agent


def collect_and_train(agent, tf_train_env, num_iterations, trajectory_num_steps, num_envs, log_every=1):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec, batch_size=num_envs, max_length=trajectory_num_steps + 1)
    # Matches this repo's own driver convention (father_agent.py's init_collector_driver):
    # DynamicStepDriver is given a PyTFEagerPolicy-wrapped policy even though it's the
    # TF-native driver, since tf_train_env wraps a vectorized (batch_time_steps=False)
    # simulator-backed PyEnvironment rather than a standard single-lane TF environment.
    eager_policy = py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True, batch_time_steps=False)
    driver = DynamicStepDriver(
        tf_train_env, eager_policy, observers=[replay_buffer.add_batch],
        num_steps=num_envs * trajectory_num_steps)

    tf_train_env.reset()
    losses = []
    for i in range(num_iterations):
        driver.run()
        experience = replay_buffer.gather_all()
        loss = agent.train(experience).loss.numpy()
        losses.append(loss)
        replay_buffer.clear()
        if i % log_every == 0:
            rewards = experience.reward.numpy()
            std = tf.nn.softplus(agent._actor_net._std_bias).numpy()
            print(f"iter {i}: loss={loss:.4f} mean_reward={rewards.mean():.6f} "
                  f"(mean D={-rewards.mean():.6f}) std={std:.4f}")
    return losses


def save_agent(agent, path):
    checkpoint = tf.train.Checkpoint(agent=agent)
    manager = tf.train.CheckpointManager(checkpoint, path, max_to_keep=5)
    manager.save()


def load_agent(agent, path):
    checkpoint = tf.train.Checkpoint(agent=agent)
    manager = tf.train.CheckpointManager(checkpoint, path, max_to_keep=5)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Loaded risk-budget agent from checkpoint: {manager.latest_checkpoint}")
    else:
        print(f"No checkpoint found at {path}, starting from scratch.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path")
    parser.add_argument("--load-agent", required=True, help="agent_folder under trained_agents/, e.g. trained_agents/test-corridor/greedy-iter-100")
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-environments", type=int, default=16)
    parser.add_argument("--trajectory-num-steps", type=int, default=32)
    parser.add_argument("--num-iterations", type=int, default=50)
    parser.add_argument("--episode-length", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=8.6e-4)
    parser.add_argument("--save-agent", default=None, help="path to save the trained risk-budget PPO agent checkpoint")
    parser.add_argument("--use-clamp", action="store_true", default=False,
                         help="train against the older clamp-to-vmin-safe correction instead of the (default) L1-closest-allowed projection.")
    parser.add_argument("--num-epochs", type=int, default=3, help="PPO epochs per collected batch.")
    parser.add_argument("--entropy-regularization", type=float, default=0.0)
    parser.add_argument("--normalize-rewards", dest="normalize_rewards", action="store_true", default=True)
    parser.add_argument("--no-normalize-rewards", dest="normalize_rewards", action="store_false")
    parser.add_argument("--goal-rew", type=float, default=100.0)
    parser.add_argument("--fail-rew", type=float, default=-100.0)
    args_cli = parser.parse_args()

    prism_path = os.path.join(args_cli.project_path, "sketch.templ")
    properties_path = os.path.join(args_cli.project_path, "sketch.props")
    args = init_args(prism_path=prism_path, properties_path=properties_path, use_rnn_less=True,
                      max_steps=args_cli.episode_length, seed=None, prefer_stochastic=True)
    sketch = load_sketch(project_path=args_cli.project_path)
    model = sketch.pomdp
    args.num_environments = args_cli.num_environments

    environment = EnvironmentWrapperVec(model, args, num_envs=args_cli.num_environments, enforce_compilation=True,
                                         goal_value=args_cli.goal_rew, antigoal_value=args_cli.fail_rew)
    model_info = build_model_info(model)

    tf_env = TFPyEnvironment(environment)
    policy = build_shielded_policy(environment, tf_env, args, args_cli.load_agent)

    train_env = RiskBudgetTrainingEnv(environment=environment, policy=policy, model_info=model_info,
                                       actions=environment.action_keywords, nu=args_cli.nu, gamma=args_cli.gamma,
                                       use_l1_projection=not args_cli.use_clamp)
    tf_train_env = TFPyEnvironment(train_env)

    agent = build_agent(train_env, tf_train_env, learning_rate=args_cli.learning_rate, num_epochs=args_cli.num_epochs,
                        normalize_rewards=args_cli.normalize_rewards,
                        entropy_regularization=args_cli.entropy_regularization)
    losses = collect_and_train(agent, tf_train_env, args_cli.num_iterations, args_cli.trajectory_num_steps,
                                args_cli.num_environments)

    if args_cli.save_agent is not None:
        save_agent(agent, args_cli.save_agent)
        print(f"Saved risk-budget agent to {args_cli.save_agent}")


if __name__ == "__main__":
    main()
