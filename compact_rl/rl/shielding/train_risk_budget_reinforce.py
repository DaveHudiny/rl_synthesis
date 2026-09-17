"""Train a learned risk-budget function via plain REINFORCE (Monte-Carlo policy gradient,
no critic/value network) on `RiskBudgetTrainingEnv`, as a simpler alternative to the
PPO+critic approach in `train_risk_budget_shield.py` (see that module's docstring for the
shared environment/reward semantics: J_gamma via time_step.discount=gamma per step).

Single network (the actor only, `RiskBudgetActorNetwork` - unchanged from the PPO version).
Each iteration collects a fixed-length window of steps across all lanes (not fixed episode
count - lanes auto-reset independently mid-window), computes the exact Monte-Carlo
discounted return-to-go G_t per step via a single backward scan over the whole window
(correct across multiple episode boundaries within one lane's window, since
RiskBudgetTrainingEnv's own discount is exactly 0 at a reset, which zeroes the backward
recursion exactly at episode boundaries - no per-lane bookkeeping needed for this part).
Steps belonging to the trailing, not-yet-confirmed-complete episode in each lane (no later
reset observed within the window, so its true full return is unknown) are excluded from the
training batch, mirroring the same "exclude dangling episodes" principle used for the
discounted-cost evaluation metric in shields.py.

Subtracts a single non-learned scalar baseline (the batch's own mean return over valid
steps) and takes ONE gradient step per batch - no PPO-style clipped-ratio multi-epoch reuse.
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from tf_agents.trajectories import time_step as ts

from compact_rl.robust_rl.robust_rl_tools import load_sketch
from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec
from compact_rl.rl.environment.tf_py_environment import TFPyEnvironment
from compact_rl.rl.shielding.risk_budget_networks import RiskBudgetActorNetwork
from compact_rl.rl.shielding.risk_budget_training_env import RiskBudgetTrainingEnv
from compact_rl.rl.shielding.train_risk_budget_shield import build_model_info, build_shielded_policy
from compact_rl.rl.tests.general_test_tools import init_args


def build_actor(train_env, tf_train_env, learning_rate=8.6e-4, actor_net=None):
    state_feature_dim = train_env.observation_spec()["state_features"].shape[0]
    if actor_net is None:
        actor_net = RiskBudgetActorNetwork(tf_train_env.observation_spec(), train_env.max_pairs,
                                            train_env.max_actions, state_feature_dim)
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.99, beta_2=0.99, weight_decay=0.0001)
    return actor_net, optimizer


def collect_window(actor_net, tf_train_env, policy_state, window_steps):
    """Steps the batched environment window_steps times using actor_net's own stochastic
    distribution, recording everything needed to recompute log_probs and returns afterward.
    Returns stacked [num_envs, window_steps, ...] tensors/arrays plus the final policy_state
    (carried into the next window, matching how a driver's policy_state normally persists
    across .run() calls - the LSTM encoder resets internally wherever step_type==FIRST
    regardless of what's passed in, so a stale state for an ended episode is harmless)."""
    observations_list = []
    actions_list = []
    step_types_list = []
    rewards_list = []
    discounts_list = []
    next_step_types_list = []

    time_step = tf_train_env.current_time_step()
    initial_policy_state = policy_state
    for _ in range(window_steps):
        dist, policy_state = actor_net(time_step.observation, time_step.step_type, policy_state, training=False)
        action = dist.sample()
        observations_list.append(time_step.observation)
        actions_list.append(action)
        step_types_list.append(time_step.step_type)

        next_time_step = tf_train_env.step(action)
        rewards_list.append(next_time_step.reward)
        discounts_list.append(next_time_step.discount)
        next_step_types_list.append(next_time_step.step_type)
        time_step = next_time_step

    observations = {k: tf.stack([o[k] for o in observations_list], axis=1) for k in observations_list[0]}
    actions = tf.stack(actions_list, axis=1)
    step_types = tf.stack(step_types_list, axis=1)
    rewards = tf.stack(rewards_list, axis=1).numpy()
    discounts = tf.stack(discounts_list, axis=1).numpy()
    next_step_types = tf.stack(next_step_types_list, axis=1).numpy()

    return observations, actions, step_types, rewards, discounts, next_step_types, initial_policy_state, policy_state


def compute_returns_and_mask(rewards, discounts, next_step_types):
    """rewards/discounts/next_step_types: [num_envs, window_steps] numpy arrays.
    Returns (returns, valid_mask), both [num_envs, window_steps]."""
    num_envs, window_steps = rewards.shape
    returns = np.zeros_like(rewards)
    running = np.zeros(num_envs, dtype=np.float32)
    for t in reversed(range(window_steps)):
        running = rewards[:, t] + discounts[:, t] * running
        returns[:, t] = running

    # A step is "valid" (its return is exact, not truncated) iff its episode is confirmed to
    # have ended by some later step within this window - i.e. there's a step_type==FIRST at
    # or after it. The trailing run with no such later reset is the one still-in-progress
    # episode per lane whose true remaining cost we haven't observed yet.
    is_episode_end = next_step_types == ts.StepType.FIRST
    valid_mask = np.flip(np.maximum.accumulate(np.flip(is_episode_end, axis=1), axis=1), axis=1)
    return returns, valid_mask


def compute_analytic_baseline(observations):
    """Model-based, zero-learning per-step baseline signal: the same (unnormalized) weight
    CapacityRiskBudget's own formula assigns to each reachable (action, next-state) pair -
    action_prob_per_slot * pair_prob * (pair_vmax - pair_vmin) - summed over reachable pairs.
    This is a function of the current state and the fixed policy's OWN current proposal only
    (already-known quantities from the transition model, not of the risk-budget action chosen
    this step), so it's a valid control-variate signal: fitting a regression coefficient
    against it and subtracting that from the return cannot bias the policy gradient (it doesn't
    depend on the action), while being correlated with the actual future cost for the same
    reason CapacityRiskBudget itself outperforms a uniform allocation - a branch with a large
    (vmax-vmin) gap under a likely action is where cost tends to concentrate.

    Returns an array shaped like `returns` ([num_envs, window_steps])."""
    mask = observations["pair_mask"].numpy()
    pair_action_onehot = observations["pair_action_onehot"].numpy()
    action_distribution = observations["action_distribution"].numpy()
    pair_prob = observations["pair_prob"].numpy()
    pair_vmin = observations["pair_vmin"].numpy()
    pair_vmax = observations["pair_vmax"].numpy()

    action_prob_per_slot = np.sum(pair_action_onehot * action_distribution[..., np.newaxis, :], axis=-1)
    weight = action_prob_per_slot * pair_prob * (pair_vmax - pair_vmin)
    weight = np.where(mask, weight, 0.0)
    return weight.sum(axis=-1)


def train_step(actor_net, optimizer, observations, step_types, actions, returns, valid_mask, initial_policy_state,
               use_analytic_baseline=False):
    valid = valid_mask.astype(bool)
    n_valid = max(int(valid.sum()), 1)
    r_mean = float(np.sum(returns * valid_mask) / n_valid)
    c_star = 0.0

    if use_analytic_baseline and valid.sum() > 1:
        analytic = compute_analytic_baseline(observations)
        a_valid = analytic[valid]
        a_mean = float(a_valid.mean())
        a_centered = analytic - a_mean
        var_a = float(np.mean((a_centered[valid]) ** 2))
        if var_a > 1e-8:
            cov_ra = float(np.mean((returns[valid] - r_mean) * a_centered[valid]))
            c_star = cov_ra / var_a
        advantage_full = (returns - r_mean) - c_star * a_centered
    else:
        advantage_full = returns - r_mean

    advantage = tf.constant(advantage_full, dtype=tf.float32)
    mask = tf.constant(valid_mask, dtype=tf.float32)

    with tf.GradientTape() as tape:
        dist, _ = actor_net(observations, step_types, initial_policy_state, training=True)
        log_probs = dist.log_prob(actions)
        loss = -tf.reduce_sum(advantage * log_probs * mask) / tf.reduce_sum(mask)
    grads = tape.gradient(loss, actor_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, actor_net.trainable_variables))
    return float(loss.numpy()), r_mean, c_star


def save_actor(actor_net, path):
    checkpoint = tf.train.Checkpoint(actor_net=actor_net)
    manager = tf.train.CheckpointManager(checkpoint, path, max_to_keep=5)
    manager.save()


def load_actor(actor_net, path):
    checkpoint = tf.train.Checkpoint(actor_net=actor_net)
    manager = tf.train.CheckpointManager(checkpoint, path, max_to_keep=5)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Loaded actor from checkpoint: {manager.latest_checkpoint}")
    else:
        print(f"No checkpoint found at {path}, starting from scratch.")


def collect_and_train(actor_net, optimizer, tf_train_env, num_iterations, window_steps, num_envs, log_every=1,
                       use_analytic_baseline=False):
    tf_train_env.reset()
    policy_state = actor_net.get_initial_state(batch_size=num_envs)
    losses = []
    for i in range(num_iterations):
        observations, actions, step_types, rewards, discounts, next_step_types, initial_policy_state, policy_state = (
            collect_window(actor_net, tf_train_env, policy_state, window_steps))
        returns, valid_mask = compute_returns_and_mask(rewards, discounts, next_step_types)
        loss, baseline, c_star = train_step(actor_net, optimizer, observations, step_types, actions, returns,
                                             valid_mask, initial_policy_state,
                                             use_analytic_baseline=use_analytic_baseline)
        losses.append(loss)
        if i % log_every == 0:
            mean_d = -float(np.sum(rewards * valid_mask) / max(np.sum(valid_mask), 1))
            frac_valid = float(valid_mask.mean())
            print(f"iter {i}: loss={loss:.4f} mean_D={mean_d:.6f} baseline_return={baseline:.4f} "
                  f"c_star={c_star:.4f} frac_valid_steps={frac_valid:.3f}")
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path")
    parser.add_argument("--load-agent", required=True)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-environments", type=int, default=16)
    parser.add_argument("--window-steps", type=int, default=128,
                         help="steps collected per lane per iteration - should be several times the episode length so most lanes complete multiple episodes.")
    parser.add_argument("--num-iterations", type=int, default=50)
    parser.add_argument("--episode-length", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=8.6e-4)
    parser.add_argument("--save-agent", default=None)
    parser.add_argument("--goal-rew", type=float, default=100.0)
    parser.add_argument("--fail-rew", type=float, default=-100.0)
    parser.add_argument("--use-clamp", action="store_true", default=False,
                         help="train against the older clamp-to-vmin-safe correction instead of the (default) L1-closest-allowed projection.")
    parser.add_argument("--use-analytic-baseline", action="store_true", default=False,
                         help="use a model-based control-variate baseline (CapacityRiskBudget's own unnormalized weight sum) instead of the plain batch-mean-return baseline.")
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

    actor_net, optimizer = build_actor(train_env, tf_train_env, learning_rate=args_cli.learning_rate)
    collect_and_train(actor_net, optimizer, tf_train_env, args_cli.num_iterations, args_cli.window_steps,
                       args_cli.num_environments, use_analytic_baseline=args_cli.use_analytic_baseline)

    if args_cli.save_agent is not None:
        save_actor(actor_net, args_cli.save_agent)
        print(f"Saved actor to {args_cli.save_agent}")


if __name__ == "__main__":
    main()
