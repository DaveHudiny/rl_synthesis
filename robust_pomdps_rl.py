
from compact_rl.robust_rl.robust_rl_trainer import initialize_extractor
from compact_rl.robust_rl.robust_rl_tools import parse_args
from compact_rl.robust_rl.robust_rl_tools import assignment_to_pomdp, load_sketch
import paynt.cli

import os
import cProfile

import paynt.utils
import paynt.utils.timer
from paynt.quotient.pomdp_family import PomdpFamilyQuotient

from compact_rl.rl.tests.general_test_tools import init_args

from paynt.rl_extension.robust_rl.family_quotient_numpy import FamilyQuotientNumpy

import random
import numpy as np
import tensorflow as tf

import logging

logger = logging.getLogger(__name__)

def set_global_seeds(seed):
    """Set the global random seeds for reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    args_cmd = parse_args()

    paynt.utils.timer.GlobalTimer.start()

    profiling = True
    if profiling:
        pr = cProfile.Profile()
        pr.enable()

    paynt.cli.setup_logger()

    project_path = args_cmd.project_path
    pomdp_sketch : PomdpFamilyQuotient = load_sketch(project_path)


    num_samples_learn = 4001
    if "avoid-large" in project_path:
        nr_additional_pomdps = 10
    else:
        nr_additional_pomdps = 5

    # This can be useful for extraction and some other stuff.
   

    prism_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")

    # Here, you can change the main parameters of the training etc.
    # Batched_vec_storm is used to run multiple different POMDPs in parallel. If you want to always run a single POMDP (e.g. worst-case), set it to False.
    # Masked_training is used to train the agent with masking, i.e. the agent will be forbidden to take illegal actions.
    args_emulated = init_args(
        prism_path=prism_path, properties_path=properties_path, batched_vec_storm=True, masked_training=False)
    args_emulated.width_of_lstm = args_cmd.lstm_width
    args_emulated.batched_vec_storm = args_cmd.batched_vec_storm
    args_emulated.extraction_type = args_cmd.extraction_method
    args_emulated.model_name = project_path.split("/")[-1]
    args_emulated.max_steps = 601
    args_emulated.geometric_batched_vec_storm = args_cmd.geometric_batched_vec_storm
    args_emulated.without_extraction = args_cmd.without_extraction
    args_emulated.periodic_restarts = args_cmd.periodic_restarts
    args_emulated.noisy_observations = args_cmd.noisy_observations
    args_emulated.single_pomdp_experiment = args_cmd.single_pomdp_setting
    args_emulated.seed = args_cmd.seed
    args_emulated.with_gru = args_cmd.with_gru

    set_global_seeds(args_emulated.seed)

    # If family has more than one POMDP, we need to use FamilyQuotientNumpy for robust RL.
    if len(list(pomdp_sketch.family.all_combinations())) > 1: # Robust version of the loop
        family_quotient_numpy = FamilyQuotientNumpy(pomdp_sketch)
        if args_emulated.single_pomdp_experiment:
            nr_additional_pomdps = 0

        hole_assignment = pomdp_sketch.family.pick_random()
        pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, hole_assignment)
        # Print number of members of family
        logger.info("Number of POMDPs in family: %d", len(list(pomdp_sketch.family.all_combinations())))

        extractor = initialize_extractor(
            pomdp_sketch, args_emulated, family_quotient_numpy)

        agent = extractor.generate_agent(pomdp, args_emulated)
        extractor.extraction_loop(pomdp_sketch, project_path=project_path,
                                nr_initial_pomdps=nr_additional_pomdps, num_samples_learn=num_samples_learn)
    else: # Non-robust version of the loop, i.e., family has only one POMDP. This is because we do not want to repeatedly extract policy for the same POMDP.
        family_quotient_numpy = FamilyQuotientNumpy(pomdp_sketch)
        hole_assignment = pomdp_sketch.family.pick_random() # There is only one anyway
        pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, hole_assignment)
        extractor = initialize_extractor(
            pomdp_sketch, args_emulated, family_quotient_numpy)
        agent = extractor.generate_agent(pomdp, args_emulated)
        extractor.train_and_extract_single_pomdp(pomdp_sketch, nr_iterations=401, num_samples_learn=num_samples_learn, args=args_emulated, project_path=project_path)





main()
