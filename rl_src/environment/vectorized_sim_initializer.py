# This file is used to initialize the vectorized simulation environment

import vec_storm
import os
import sys
import pickle as pkl
import re

from rl_src.environment.batched_vec_storm import BatchedVecStorm

import logging
logger = logging.getLogger(__name__)

storm_vec_env_constructor = vec_storm.StormVecEnv

from rl_src.tools.args_emulator import ArgsEmulator


class SimulatorInitializer:
    @staticmethod
    def load_and_store_simulator(stormpy_model = None, 
                                 get_scalarized_reward : callable = None, 
                                 num_envs : int = 1, max_steps : int = 400, 
                                 metalabels : dict = {"goals": "goal"}, 
                                 model_path : str = ".models/mba/", 
                                 compiled_models_path : str ="compiled_models_vec_storm",
                                 enforce_recompilation : bool = False,
                                 obs_evaluator = None,
                                 quotient_state_valuations = None,
                                 observation_to_actions = None,
                                 batched_vec_storm = False,
                                 args : ArgsEmulator = None) -> vec_storm.StormVecEnv | BatchedVecStorm:
        """ Load the simulator for the environment. If the model was not compiled previously, the model is compiled from scratch and saved. Otherwise, the model is loaded from the file.

        Args:
            stormpy_model (stormpy.model.Model): The stormpy model (POMDP).
            get_scalarized_reward (function): The function to get the scalarized reward.
            num_envs (int): The number of environments.
            max_steps (int): The maximum number of steps.
            metalabels (list): The metalabels of goal states.
            model_path (str): The path to the model. For loading and saving the model.
            compiled_models_path (str): The path to the compiled models.
            enforce_recompilation (bool): If True, the model is recompiled even if it was compiled before.
            obs_evaluator (): An object, that contains valuations for the observations from the quotient instance
            quotient_state_valuations (dict): The valuations of the quotient states.
            observation_to_actions (dict): The mapping from observations to actions.
            batched_vec_storm (bool): If True, use the BatchedVecStorm simulator instead of the StormVecEnv.

        Returns:
            VectorizedSimulator: The vectorized simulator for the environment.
        """
        if not os.path.exists(compiled_models_path):
            os.makedirs(compiled_models_path)
        name = SimulatorInitializer.get_name_from_path(model_path)
        
        assert (batched_vec_storm and enforce_recompilation) or not batched_vec_storm, "Batched vectorized storm can only be used with enforced recompilation.\
            Set enforce_recompilation=True if you want to use BatchedVecStorm."
        
        global storm_vec_env_constructor
        if batched_vec_storm:
            storm_vec_env_constructor = BatchedVecStorm

        if enforce_recompilation or "unknown" in name:
            logger.info(f"Compiling model {name}...")
            simulator = storm_vec_env_constructor(
                stormpy_model, get_scalarized_reward, num_envs=num_envs, max_steps=max_steps, metalabels=metalabels,
                obs_evaluator=obs_evaluator,
                quotient_state_valuations=quotient_state_valuations,
                observation_to_actions=observation_to_actions, seed=args.seed if args is not None else int.from_bytes(os.urandom(4), "big"))
            if batched_vec_storm:
                simulator.set_args(args)
            return simulator
        
        simulator = SimulatorInitializer.try_load_simulator_by_name_from_pickle(
            name, compiled_models_path)
        if simulator is None:
            logger.info(f"Compiling model {name}...")
            simulator = storm_vec_env_constructor(
                stormpy_model, get_scalarized_reward, num_envs=num_envs, max_steps=max_steps, metalabels=metalabels,
                obs_evaluator=obs_evaluator,
                quotient_state_valuations=quotient_state_valuations,
                observation_to_actions=observation_to_actions, seed=args.seed if args is not None else int.from_bytes(os.urandom(4), "big"))
            simulator.save(f"{compiled_models_path}/{name}.pkl")
        return simulator

    @staticmethod
    def get_name_from_path(model_path):
        """ Get the name of the model from the model_path. The model path looks like /path/to/model/name/sketch.templ

        Args:
            path (str): The path to the model.

        Returns:
            str: The name of the model.
        """
        if model_path is None:
            return None
        return re.search(r"([^/]+)\/sketch.templ", model_path).group(1)

    @staticmethod
    def try_load_simulator_by_name_from_pickle(name, path_to_compiled_models):
        """ Try to load the simulator by name.

        Args:
            name (str): The name of the model.
            path_to_compiled_models (str): The path to the compiled models.

        Returns:
            VectorizedSimulator: The simulator.
        """
        if not os.path.exists(f"{path_to_compiled_models}/{name}.pkl") or name is None:
            logger.info(
                f"Model {name} not found in {path_to_compiled_models}. The model will be compiled.")
            return None
        else:
            logger.info(
                f"Model {name} found in {path_to_compiled_models}. The model will be loaded")
            try:
                return vec_storm.StormVecEnv.load(f"{path_to_compiled_models}/{name}.pkl")
            except Exception as e:
                logger.error(
                    f"Error while loading model {name} from {path_to_compiled_models}: {e}")
                return None
