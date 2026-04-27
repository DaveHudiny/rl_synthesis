#!/usr/bin/env python3
"""
Probabilistic Shielding Demo
=============================

Demonstrates shielding for MDPs as described in:
  "Shields to Guarantee Probabilistic Safety in MDPs"
  
Run inside Docker:

  docker build -t cav26-shielding . && \\
  docker run -v "$PWD/results:/app/results" --rm -it cav26-shielding \\
      /bin/bash -c "python3 shielding-demo.py \\
          models/shielding/test-corridor/sketch.templ \\
          --shield-type online --nu 0.1"
"""

import os
import pickle

import click
import numpy as np
import stormpy
import stormpy.simulator

from rl_src.shielding.model_info import ModelInfo
from rl_src.shielding.shield_processor import ShieldData
import rl_src.shielding.shields as shields


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_builder_options() -> stormpy.BuilderOptions:
    opts = stormpy.BuilderOptions()
    opts.set_build_state_valuations(True)
    opts.set_build_with_choice_origins(True)
    opts.set_build_all_labels(True)
    opts.set_build_choice_labels(True)
    opts.set_add_overlapping_guards_label(True)
    opts.set_build_observation_valuations(True)
    opts.set_build_all_reward_models(True)
    return opts


def build_model_info(model: stormpy.storage.SparsePomdp) -> ModelInfo:
    """Compute vmin/vmax via model checking and wrap into ModelInfo."""
    # The shield needs an MDP view of the model (POMDP with all observations = states).
    components = stormpy.SparseModelComponents(
        transition_matrix=model.transition_matrix,
        reward_models=model.reward_models,
        state_labeling=model.labeling,
    )
    components.choice_labeling = model.choice_labeling
    if model.has_state_valuations():
        components.state_valuations = model.state_valuations
    if model.has_choice_origins():
        components.choice_origins = model.choice_origins

    mdp = stormpy.storage.SparseMdp(components)

    vmin = stormpy.model_checking(
        mdp, stormpy.parse_properties("Pmin=? [ F \"bad\" ]")[0]
    ).get_values()
    vmax = stormpy.model_checking(
        mdp, stormpy.parse_properties("Pmax=? [ F \"bad\" ]")[0]
    ).get_values()

    s0 = mdp.initial_states[0]
    print(f"  Model: {model.nr_states} states, {len(model.choice_labeling.get_labels())} actions")
    print(f"  vmin(s0) = {vmin[s0]:.4f}   vmax(s0) = {vmax[s0]:.4f}")

    observation_to_state = [None] * model.nr_observations
    for state in range(model.nr_states):
        observation_to_state[model.get_observation(state)] = state

    return ModelInfo(
        model=model,
        observation_to_state=observation_to_state,
        bad_state="bad",
        vmin=vmin,
        vmax=vmax,
    )


# Helper: saving a shield
def save_shield(shield, path) -> None:
    """Pickle the learned data of a self-constructing shield to *path*."""
    data = ShieldData(
        actions=shield.actions,
        original_model_nr_states=shield.model_info.model.nr_states,
        observation_to_state=shield.model_info.observation_to_state,
        memory=shield.memory,
        rounding_precision=shield.rounding_precision,
        initial_node=shield.initial_node,
        current_action_distributions=shield.current_action_distributions,
    )
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Shield saved to {path}")

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_episodes(
    shield: shields.Shield,
    model: stormpy.storage.SparsePomdp,
    model_info: ModelInfo,
    actions: list,
    n_episodes: int,
    episode_length: int = 20,
) -> int:
    """Simulate *n_episodes* with a uniform random policy shaped by *shield*.

    The shield's ``correct()`` call signature is::

        shielded_dist = shield.correct(last_action, current_state, agent_dist, reset)

    where
      - ``last_action``   is the global action index played in the previous step
                          (``None`` at the first step of an episode),
      - ``current_state`` is the model state index,
      - ``agent_dist``    is a list of floats of length == nr available actions,
                          summing to 1.0 (the agent's proposed distribution),
      - ``reset``         is ``True`` only at the first step of each episode.

    Returns the number of episodes that reached a bad state.
    """
    simulator = stormpy.simulator.create_simulator(model)
    bad_episodes = 0

    for _ in range(n_episodes):
        current_obs, _, _ = simulator.restart()
        current_state = model_info.observation_to_state[current_obs]
        last_action = None   # no previous action at episode start

        for step in range(episode_length):
            row_start = model.transition_matrix.get_row_group_start(current_state)
            nr_actions = model.transition_matrix.get_row_group_end(current_state) - row_start

            # Agent proposes a uniform distribution (replace with your RL policy here).
            agent_dist = [1.0 / nr_actions] * nr_actions

            # Shield corrects the distribution.
            # Pass reset=True only at the very first step of each episode.
            shielded_dist = shield.correct(
                last_action, current_state, agent_dist, reset=(step == 0)
            )

            # Sample a local action index from the shielded distribution.
            local_action = np.random.choice(len(shielded_dist), p=shielded_dist)

            # Get global action index from local action index and model's choice labeling.
            last_action = actions.index(
                model.choice_labeling.get_labels_of_choice(row_start + local_action).pop()
            )

            next_obs, _, labels = simulator.step(local_action)
            current_state = model_info.observation_to_state[next_obs]

            if "bad" in labels:
                bad_episodes += 1
                break
            if "goal" in labels:
                break  # success; end episode early

    return bad_episodes


# ---------------------------------------------------------------------------
# Load for self-constructing shields
# ---------------------------------------------------------------------------

def load_shield_as_static(
    path: str,
    model_info: ModelInfo,
    actions: list,
    nu: float,
) -> shields.SelfConstructingShield:
    """Load pickled shield data into a *static* SelfConstructingShield.

    The base SelfConstructingShield class enforces learned distributions
    but never adds new ones, making it safe to use for deployment.

    Usage::

        # First, initialize the SelfConstructingShield object for the model.
        static_shield = load_shield_as_static("results/offline-shield.pickle",
                                              model_info, actions, nu=0.1)
        # Then use it normally:
        shielded_dist = static_shield.correct(last_action, state, agent_dist, reset)
    """
    with open(path, "rb") as f:
        data: ShieldData = pickle.load(f)

    # Initialize with the same model; the base class will not update.
    static = shields.SelfConstructingShield(
        model_info=model_info, actions=actions, nu=nu, memory=data.memory
    )
    static.rounding_precision = data.rounding_precision

    # Overwrite the empty initial node with the learned tree.
    static.initial_node = data.initial_node
    # Overwrite the per-state allowed distributions (needed for memory > 0).
    static.current_action_distributions = data.current_action_distributions
    if data.memory > 0:
        static.load_matrix_vector_from_current_distributions()

    print(f"  Loaded ← {path}  (tree nodes: {static.initial_node.number_of_tree_nodes()})")
    return static


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SHIELD_CHOICES = ["safe", "pessimistic", "optimistic", "delta", "online", "offline"]


@click.command()
@click.argument(
    "model_path",
    type=click.Path(exists=True),
    default="models/shielding/test-corridor/sketch.templ",
)
@click.option(
    "--shield-type",
    type=click.Choice(SHIELD_CHOICES),
    default="safe",
    show_default=True,
    help="Shielding method to use.",
)
@click.option(
    "--nu",
    type=float,
    default=0.05,
    show_default=True,
    help="Safety threshold ν: acceptable probability of reaching bad states.",
)
@click.option(
    "--construction-episodes",
    type=int,
    default=100,
    show_default=True,
    help="Episodes used to construct an offline shield.",
)
@click.option(
    "--eval-episodes",
    type=int,
    default=1000,
    show_default=True,
    help="Episodes used for the evaluation.",
)
@click.option(
    "--episode-length",
    type=int,
    default=20,
    show_default=True,
    help="Maximum steps per episode.",
)
def main(model_path, shield_type, nu, construction_episodes, eval_episodes, episode_length):
    """Probabilistic shielding demo.

    MODEL_PATH is a PRISM sketch.templ for a fully observable POMDP (MDP)
    with a 'bad' label marking unsafe states.

    \b
    Examples:
      python3 shielding-demo.py models/shielding/test-corridor/sketch.templ \\
          --shield-type safe --nu 0.1
      python3 shielding-demo.py models/shielding/test-corridor/sketch.templ \\
          --shield-type online --nu 0.1 --construction-episodes 1000
      python3 shielding-demo.py models/shielding/test-corridor/sketch.templ \\
          --shield-type offline --nu 0.1
    """
    print(f"\n{'='*62}")
    print(f"  Probabilistic Shielding Demo")
    print(f"  model  : {model_path}")
    print(f"  shield : {shield_type}   ν = {nu}")
    print(f"{'='*62}\n")

    # ------------------------------------------------------------------
    # 1. Load model and compute vmin / vmax
    # ------------------------------------------------------------------
    print("Loading model and running model checking...")
    prism = stormpy.parse_prism_program(model_path)
    model = stormpy.build_sparse_model_with_options(prism, get_builder_options())

    assert model.nr_states == model.nr_observations, (
        "Shielding currently requires a fully observable POMDP (MDP): "
        "nr_states must equal nr_observations."
    )
    assert model.initial_states is not None and len(model.initial_states) == 1, (
        "Only single-initial-state models are supported."
    )

    model_info = build_model_info(model)

    # Global list of action labels (order is fixed for the life of the shield).
    actions = list(model.choice_labeling.get_labels())

    # ------------------------------------------------------------------
    # 2. Create the requested shield
    # ------------------------------------------------------------------

    # Factory for all shields except 'offline' (which has its own workflow below).
    SHIELD_FACTORIES = {
        "safe":        lambda: shields.StandardShield(model_info=model_info, actions=actions),
        "pessimistic": lambda: shields.PessimisticShield(model_info=model_info, actions=actions, nu=nu),
        "optimistic":  lambda: shields.OptimisticShield(model_info=model_info, actions=actions, nu=nu),
        "delta":       lambda: shields.DeltaShield(model_info=model_info, actions=actions, delta=nu),
        "online":      lambda: shields.SelfConstructingShieldOnline(model_info=model_info, actions=actions, nu=nu),
    }

    if shield_type == "offline":
        # Offline workflow: construct from traces → save → load as static.
        #
        # You can use any sampling strategy here.
        # The loaded static SelfConstructingShield (base class) is safe (S+).

        # Phase 1: construct
        print(f"\n[Phase 1/3] Constructing offline shield ({construction_episodes} episodes)...")
        offline_shield = shields.SelfConstructingShieldOffline(
            model_info=model_info, actions=actions, nu=nu
        )
        offline_shield.rounding_precision = 6
        run_episodes(offline_shield, model, model_info, actions, construction_episodes, episode_length)
        print(f"  Tree nodes: {offline_shield.initial_node.number_of_tree_nodes()}  "
              f"non-optimal distributions added: {offline_shield.added_nonoptimal_actions}")

        # Phase 2: save
        os.makedirs("results", exist_ok=True)
        shield_path = "results/offline-shield.pickle"
        print(f"\n[Phase 2/3] Saving shield...")
        save_shield(offline_shield, shield_path)

        # Phase 3: load as static
        print(f"\n[Phase 3/3] Loading as static shield...")
        shield = load_shield_as_static(shield_path, model_info, actions, nu)

    else:
        shield = SHIELD_FACTORIES[shield_type]()

    shield.rounding_precision = 6

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    print(f"\nEvaluating over {eval_episodes} episodes (max {episode_length} steps each)...")
    bad = run_episodes(shield, model, model_info, actions, eval_episodes, episode_length)

    print(f"\n{'='*62}")
    print(f"  shield={shield_type}   ν={nu}")
    print(f"  Bad episodes    : {bad}/{eval_episodes}")
    print(f"  Shield calls    : {shield.shield_calls}")
    print(f"  Blocked actions : {shield.blocked_actions}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
