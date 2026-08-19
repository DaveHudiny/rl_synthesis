import pickle
from collections import defaultdict
from tf_agents.trajectories import StepType
from aalpy.learning_algs import run_RPNI
from aalpy.automata import MooreMachine,MealyMachine



def can_be_run_mealy(fsc : MealyMachine ,trajectory):
    state = fsc.initial_state
    for obs,act in trajectory: # assuming input completeness
        if obs not in state.transitions:
            return False
        if state.output_fun[obs] != act:
            return False
        state = state.transitions[obs]
    return True

def can_be_run_onfsm(fsc : MealyMachine ,trajectory):
    state = fsc.initial_state
    non_det = 0
    for obs,act in trajectory: # assuming input completeness
        if obs not in state.transitions:
            return non_det,False
        transitions = state.transitions[obs]
        if len(transitions) > 1:
            non_det += 1
            # print(f"Nr. trans: {len(transitions)}")
        found = False
        for transition in transitions:
            if transition[0] == act:
                state = transition[1]
                found = True
                break
        if not found:
            return non_det, False

    return non_det,True


def can_be_run_moore(fsc : MooreMachine ,trajectory):
    obs_seq, final_act = trajectory
    state = fsc.initial_state
    for obs in obs_seq: # assuming input completeness
        if obs not in state.transitions:
            return False
        state = state.transitions[obs]
    return state.output == final_act

def create_mealy_learn_traj(trajectories):
    mealy_trajs = []
    for traj in trajectories:
        for k in range(len(traj)):
            mealy_traj_i = [i[0] if type(i) == list else i for i,o in traj[:k+1]]
            mealy_traj_o = traj[k][1]
            mealy_trajs.append((mealy_traj_i,mealy_traj_o))
    return mealy_trajs

def eval_model(fsc,trajectories,mealy, onfsm):
    acc = 0
    non_det_count = 0
    for trajectory in trajectories:
        if onfsm:
            non_det,accepted = can_be_run_onfsm(fsc,trajectory)
            if accepted:
                acc += 1
            non_det_count += non_det
        if not onfsm and mealy and can_be_run_mealy(fsc,trajectory):
            acc += 1
        if not onfsm and not mealy and can_be_run_moore(fsc,trajectory):
            acc += 1
    print(f"Accuracy: {acc/len(trajectories)}, accepted {acc} out {len(trajectories)}")
    if onfsm:
       print(f"Non-det.: {non_det_count/sum(map(len,trajectories))}, "
             f"non-det {non_det_count} out {sum(map(len,trajectories))}")

def split_long_traj(long_trajectory,mealy):
    short_trajectories = []
    short_trajectory = None

    last_obs = None
    for (action, obs,step_type) in long_trajectory:
        if step_type == int(StepType.FIRST):
            short_trajectory = []
            if mealy:
                last_obs = str(obs)
            else:
                short_trajectory.append(str(obs))
        elif step_type == int(StepType.MID):
            if mealy:
                short_trajectory.append((last_obs,action))
                last_obs = str(obs)
            else:
                short_trajectory.append(str(obs))
        elif step_type == int(StepType.LAST):
            if mealy:
                short_trajectory.append((last_obs,action))
                short_trajectories.append(short_trajectory)
            else:
                short_trajectory.append(str(obs))
                short_trajectories.append((tuple(short_trajectory),action))
        else:
            raise Exception("Unknown step type")
    return short_trajectories


def learn_automaton(model_name,mealy,n_envs,onfsm):
    with open(f"aut_learn_data_{model_name}.pkl", "rb") as fp:
        aut_learn_data = pickle.load(fp)
    if aut_learn_data is None:
        exit(1)
    all_trajectories = create_trajectories(aut_learn_data, mealy, n_envs)
    print(f"All trajectories: {len(all_trajectories)}")
    max_traj = 100_000
    n_traj = int(min(len(all_trajectories), max_traj) * 1)
    # n_traj = 25
    learn_trajectories = all_trajectories[:n_traj]
    eval_trajectories = all_trajectories[n_traj:max_traj]

    if onfsm:
        from aalpy.learning_algs import run_k_tails
        model = run_k_tails(learn_trajectories, "mealy", k=4, input_completeness=None, print_info=True)
    else:
        if mealy:
            learn_trajectories = create_mealy_learn_traj(learn_trajectories)
            model = run_RPNI(learn_trajectories, 'mealy', algorithm='gsm',
                             input_completeness=None, print_info=True)
        else:
            model = run_RPNI(learn_trajectories, 'moore', algorithm='gsm',
                             input_completeness=None, print_info=True)
    model.make_input_complete()
    model.save(f"{model_name}_fsc.dot")
    return model,eval_trajectories


def create_trajectories(aut_learn_data, mealy, n_envs, integer_obs = True):
    init_time_step = aut_learn_data[0]
    long_trajectories = defaultdict(list)
    integer_obs_str = "integer" if integer_obs else "observation"
    for ne in range(n_envs):
        action = None
        if integer_obs:
            obs = int(init_time_step.observation[integer_obs_str][ne].numpy())
        else:
            obs = list(init_time_step.observation[integer_obs_str][ne].numpy())

        step_type = int(init_time_step.step_type[ne])
        # for some reason the initial step type is MID
        long_trajectories[ne].append((action, obs, StepType.FIRST))
    for i, (policy_step, time_step) in enumerate(aut_learn_data[1:]):
        for ne in range(n_envs):
            action = int(policy_step.action[ne])
            if integer_obs:
                obs = int(time_step.observation[integer_obs_str][ne].numpy())
            else:
                obs = list(time_step.observation[integer_obs_str][ne].numpy())
            step_type = int(time_step.step_type[ne])
            long_trajectories[ne].append((action, obs, step_type))
    all_trajectories = []
    for ne in range(n_envs):
        trajectories = split_long_traj(long_trajectories[ne], mealy)
        all_trajectories.extend(trajectories)
    return all_trajectories


if __name__ == "__main__":
    # observation_labels = ['amdone', 'dx', 'dy', 'hasleft', 'seedx', 'seedy', 'start', 'turn'] # this is for intercept
    mealy = True
    # split up data from envs
    n_envs = 16
    model_name = "geo-2-8"
    onfsm = True
    model,eval_trajectories = learn_automaton(model_name,mealy,n_envs,onfsm)

    eval_model(model,eval_trajectories,mealy,onfsm=onfsm)