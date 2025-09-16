# %%
from imports import *
from information_conditions import Information_Conditions
from base_ecopg import *
from helper_functions import *
from simulation_and_results_functions import *

# %%
mae_ecopg_only_action = create_mae_ecopg_for_given_mode_POstratAC("social")
mae_ecopg_both_state_and_action = create_mae_ecopg_for_given_mode_POstratAC("complete")

# %%
def generate_random_initial_conditions_around_point(mae, test_point, num_samples, perturbation_size, mode):
    random_initial_conditions = lhs_sampling(mae.Q, num_samples, mae.N)
    # if mode == 'complete' or  mode == 'ecological':
    #     random_initial_conditions_around_point = [(add_random_degraded_state_policy(test_point) + perturbation_size*sample)/(1 + perturbation_size) for sample in random_initial_conditions]
    # if mode ==  'none' or  mode == 'social':
    random_initial_conditions_around_point = [(test_point + perturbation_size*sample)/(1 + perturbation_size) for sample in random_initial_conditions]
    return random_initial_conditions_around_point

# %%
def check_for_stability(test_point, mae, mode):
    

    random_initial_conditions_around_point = generate_random_initial_conditions_around_point(mae, test_point , 5, 0.05, mode)
    results_list = []
    for initial_condition in random_initial_conditions_around_point:
        result = run_simulation_for_initial_condition(
                mae = mae, 
                initial_condition = initial_condition,
                mode = mode,
            )  
        results_list.append(result)

    new_final_points = [result['final_point'] for result in results_list]

    if mode == 'complete' or  mode == 'ecological':
    #     new_final_points = [new_final_point[:, 1::2,:] for new_final_point in new_final_points]
        test_point = test_point[:, 1::2,:]


    checking_if_same = [np.linalg.norm(new_final_point.flatten() - (test_point.flatten()), ord = 2) < 0.1 for new_final_point in new_final_points]

    if np.all(checking_if_same) == True:
        stability =  True
    else:
        stability = False
    
    return stability


# %%
def add_degraded_state_policies_both_state_and_action(strategy):
        
        strategy_propserous_and_degraded_state = strategy.copy()

        for i in [0, 2, 4, 6]:
            x = np.random.rand()
            strategy_propserous_and_degraded_state.insert(i, x)
        
        return strategy_propserous_and_degraded_state


def add_degraded_state_policies_only_state(strategy):
        
        strategy_propserous_and_degraded_state = strategy.copy()
        x = np.random.rand()
        strategy_propserous_and_degraded_state.insert(0, x)
        
        return strategy_propserous_and_degraded_state


def create_determinstic_strategies_set_for_both_players(mode):
    '''creates deterministic strategy sets for given mode '''

    mae_ecopg_for_evaluating_no_of_states = create_mae_ecopg_for_given_mode_POstratAC(mode)
    number_of_states = mae_ecopg_for_evaluating_no_of_states.Q
    
    if mode == 'none' or mode == 'social':
        determinstic_strategy_itertools = itertools.product([1 , 0], repeat = number_of_states)
        determinisic_strategy_lists = list(list(strat) for strat in determinstic_strategy_itertools)
        all_determinstic_strategy_dictionary_full = {str(np.round(strat)):strat for strat in determinisic_strategy_lists}

    else:
        number_of_prosperous_states = int(number_of_states/2)
        determinstic_strategy_itertools = itertools.product([1 , 0], repeat = number_of_prosperous_states)
        determinisic_strategy_lists = list(list(strat) for strat in determinstic_strategy_itertools)
        all_determinstic_strategy_dictionary_only_prosperous = {str(np.round(strat)):strat for strat in determinisic_strategy_lists}

        if mode == 'complete':
             all_determinstic_strategy_dictionary_full = {key: add_degraded_state_policies_both_state_and_action(value) for key, value in all_determinstic_strategy_dictionary_only_prosperous.items()}
        elif mode == 'ecological':
             all_determinstic_strategy_dictionary_full = {key: add_degraded_state_policies_only_state(value) for key, value in all_determinstic_strategy_dictionary_only_prosperous.items()}


    strategy_set_p1 = all_determinstic_strategy_dictionary_full
    strategy_set_p2 = all_determinstic_strategy_dictionary_full

    return strategy_set_p1, strategy_set_p2


# %%
def create_policy_from_strategy(agent_1_strategy, agent_2_strategy):


    agent_1_strategy = [[x, 1-x] for x in agent_1_strategy]
    agent_2_strategy = [[x, 1-x] for x in agent_2_strategy]

    policy = np.array([agent_1_strategy, agent_2_strategy])

    return policy
# %% 

strategy_set_p1_temp = {'[1, 0, 0, 0]': [1, 0, 0, 0]}
# %%
if __name__ == '__main__':

    # filepath = os.path.join("..", "..", "..", filename)
    # df = pd.read_excel("./Code/Data/Local_Stability_Analysis/stable_policies_local_stability_analysis.xlsx")
    df = pd.read_csv("Code/Data/Local_Stability_Analysis/stable_policies_local_stability_analysis.csv")
    modes = ['ecological']
    
    print(df.head())

    for mode in modes: 
        
        discount_factor_list = [0.995]
        m_value = -4
        
        for discount_factor in discount_factor_list:

            # --- Step 1: Check if row already exists ---
            sub = df[(df["mode"] == mode) & (df["discount_factor"] == discount_factor) & (df["m_value"] == m_value)]

            if not sub.empty:
                print("Result already exists:")
                print(sub)
            else:
                print("Result does not aleady exist, computing now...")
                    
                strategy_set_p1, strategy_set_p2 = create_determinstic_strategies_set_for_both_players(mode)
                strategy_combinations = itertools.combinations_with_replacement(strategy_set_p1.values(),2)
                mae = create_mae_ecopg_for_given_mode_POstratAC_expanded(mode, m = m_value, discount_factor = discount_factor)

                # print(list(strategy_combinations))
                policies =  np.array([create_policy_from_strategy(p1_strategy, p2_strategy) for p1_strategy, p2_strategy in strategy_combinations])
                check_for_stability_mode = partial(check_for_stability, mae = mae, mode = mode)

                stable_policies_list = []

                with Pool() as pool:
                            results = pool.map(check_for_stability_mode, policies)
                            print("==", mode, "===")
                            print('m =', m_value, "discount factor =", discount_factor)
                            if mode == 'complete' or mode == 'ecological':
                                policies_for_print = np.array([policy[:, 1::2,0] for policy in policies])
                            else:
                                policies_for_print = np.array([policy[:, :,0] for policy in policies])

                            stable_policies = policies_for_print[results]

                            rows = [{"discount_factor": discount_factor,"m_value": m_value, "policy": stable_policies.tolist(), "mode" : mode}]
                            
                            stable_policies_list.extend(rows)
                
                updated_df = pd.concat([df, pd.DataFrame(stable_policies_list)], ignore_index=True)
                # updated_df = pd.DataFrame(stable_policies_list, ignore_index=True)
                updated_df.to_csv("./Code/Data/Local_Stability_Analysis/stable_policies_local_stability_analysis.csv", index=False)

