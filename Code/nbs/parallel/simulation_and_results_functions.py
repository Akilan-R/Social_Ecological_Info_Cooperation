# %%
from imports import *
from helper_functions import *
from information_conditions import Information_Conditions
# %%
def run_simulation_for_initial_condition(initial_condition, mae,
                                         make_degraded_state_cooperation_probablity_zero_at_end, make_degraded_state_obsdist_zero_at_end):


    xtraj, fixedpointreached = mae.trajectory(initial_condition, Tmax= 100000, tolerance=1e-5)
    # xtraj, fixedpointreached = mae.trajectory(initial_condition, Tmax=50000, tolerance=1e-25)
    # if fixedpointreached == False:
    #     print("Warning: Fixed point not reached within 50000 iterations", np.round(initial_condition,3), mode)
        
    final_point = xtraj[-1]
    obsdist = mae.obsdist(final_point)

   
    if make_degraded_state_cooperation_probablity_zero_at_end:  final_point = make_degraded_state_cooperation_probablity_zero(final_point, mae.env.Oset[0])
    if make_degraded_state_obsdist_zero_at_end: obsdist = exclude_degraded_states_from_obsdist(obsdist, mae.env.Oset[0])

    #IMP! Only excluded degraded states from obsdist and final point at the end of simulation for reporting values.
        #  No interference while simulations are running.
    

    avg_coop_across_states_each_agent = get_average_cooperativeness(
        policy=final_point, 
        obsdist= obsdist
    ) #we're only considiering agent i

    avg_coop_across_states_across_agents = np.mean(avg_coop_across_states_each_agent)  #average cooperation between agents. Alternatively, one could just take the cooperatoin of the first agent.
    time_to_reach = xtraj.shape[0]

    del xtraj

    results_dict = {
        'avg_coop': avg_coop_across_states_across_agents,
        'time_to_reach': time_to_reach,
        # 'xtraj': xtraj,            #xtraj is large, so we don't want to store it esp when parallel processing is odne
        'final_point': final_point,
         'obsdist': obsdist,
    }

    return results_dict


# %%


def run_simulation_across_conditions(mae, mode, num_samples, 
                                     exclude_degraded_state_for_average_cooperation):
    """
    Runs Monte Carlo simulations across multiple initial conditions.

    Parameters:
        mae: The POstratAC instance (learning agent).
        information_condition_instance: The instance of Information_Conditions.
        num_samples (int): Number of initial conditions to sample.
        initial_cooperation_in_degraded_state (int): If 0, cooperation in degraded state is set to zero;
                                                     if 1, it is set to one; otherwise, no changes.
        include_degraded_state_for_average_cooperation (bool): Whether to include the degraded state in average cooperation.

    Returns:
        list: A list of (average cooperation, time-to-reach) tuples.
    """
    result_tuple_list = []
    initial_conditions_list = lhs_sampling(mae.Q, num_samples, mae.N)

    for initial_condition in initial_conditions_list:
        result = run_simulation_for_initial_condition(
            mae, mode,
             exclude_degraded_state_for_average_cooperation,
             initial_condition
        )
        result_tuple_list.append(result)

    return result_tuple_list


# %%
def run_simulation_across_conditions_parallel(mae, num_samples, make_degraded_state_cooperation_probablity_zero_at_end = True, make_degraded_state_obsdist_zero_at_end = True):

    result_tuple_list = []

    initial_conditions_list = lhs_sampling(mae.Q, num_samples, mae.N)
    run_simulation_for_initial_condition_partial = partial(run_simulation_for_initial_condition, mae = mae, make_degraded_state_cooperation_probablity_zero_at_end = make_degraded_state_cooperation_probablity_zero_at_end, make_degraded_state_obsdist_zero_at_end = make_degraded_state_obsdist_zero_at_end)
    
    with Pool() as pool:
        result_tuple_list = pool.map(run_simulation_for_initial_condition_partial, initial_conditions_list)
    

    return result_tuple_list


# %%
def get_cooperation_time_summary(result_tuple_list):

    avg_coop_time_pairs = [(result["avg_coop"], result["time_to_reach"]) for result in result_tuple_list]
    df = pd.DataFrame(avg_coop_time_pairs, columns=["AverageCooperation", "TimeToReach"])
    total_count = len(df)

    # Overall average cooperation
    average_cooperation = round(df['AverageCooperation'].mean(), 3)

    # Classification
    df['Classification'] = df['AverageCooperation'].apply(
        lambda x: "Defection" if x < 0.1 else "Cooperation" if x > 0.9 else "Mixed"
        # lambda x: "Defection" if x < 0.4 else "Cooperation" if x > 0.6 else "Mixed"

    )

    # Basin of attraction statistics
    basin_of_attraction_size = df.groupby('Classification')['TimeToReach'].agg(
        MedianTimetoReach='median',
        Percentage=lambda x: round((len(x) / total_count) * 100, 1)
    ).reset_index()

    return average_cooperation, basin_of_attraction_size


# %%
def get_results_average_cooperation(result_tuple_list):
    """
    Computes summary statistics on cooperation outcomes.

    Parameters:
        avg_coop_time_pairs (list of tuples): Each tuple is (average_cooperation, time_to_reach).

    Returns:
        tuple: (average_cooperation, basin_of_attraction_size DataFrame)
    """
    avg_coop = [avg_coop  for avg_coop, * _rest in result_tuple_list]


    # Overall average cooperation
    average_cooperation = round(np.mean(avg_coop), 3)

    return average_cooperation


# %%
def get_results_only_cooperation_basin_of_attraction_size(result_tuple_list):
    """
    Computes summary statistics on cooperation outcomes.

    Parameters:
        avg_coop_time_pairs (list of tuples): Each tuple is (average_cooperation, time_to_reach).

    Returns:
        tuple: (average_cooperation, basin_of_attraction_size DataFrame)
    """
    avg_coop_time_pairs = [(result["avg_coop"], result["time_to_reach"]) for result in result_tuple_list]


    df = pd.DataFrame(avg_coop_time_pairs, columns=["AverageCooperation", "TimeToReach"])
    total_count = len(df)

    # Overall average cooperation
    average_cooperation = round(df['AverageCooperation'].mean(), 3)

    # Classification
    df['Classification'] = df['AverageCooperation'].apply(
        lambda x: "Defection" if x < 0.1 else "Cooperation" if x > 0.9 else "Mixed"
        # lambda x: "Defection" if x < 0.4 else "Cooperation" if x > 0.6 else "Mixed"
    )
    
    basin_stats = df['Classification'].value_counts(normalize = True)*100

 
    return round(basin_stats.get('Cooperation', 0), 1)

# %%
def make_plots_ecopgg(ecopg, mode, num_samples):

    information_condition_instance = Information_Conditions(ecopg, mode=mode)
    num_observed_states = len(information_condition_instance.Oset[0])
    x, y = ([0], list(range(num_observed_states)), [0]), ([1], list(range(num_observed_states)), [0])
    mae = POstratAC_eps(env=information_condition_instance, learning_rates=0.1, discount_factors=0.98)
    initial_conditions_list = lhs_sampling(mae.Q, num_samples, mae.N)

    ax = fp.plot_strategy_flow(
        mae,
        x, y, flowarrow_points=np.linspace(0.01, 0.99, 9), NrRandom=32,
        conds=np.array(information_condition_instance.Oset)[0, :num_observed_states],
        # col = 'blac'
    )

    for initial_condition in initial_conditions_list:
        xtraj, fixedpointreached = mae.trajectory(initial_condition, Tmax=100000, tolerance=1e-5)
        final_point = xtraj[-1]

        fp.plot_trajectories([xtraj], x, y, cols=['grey'], lss = "--", axes = ax)

        for plot_index, (x_indices,y_indicies) in enumerate(zip(it.product(*x), it.product(*y))):
            ax[plot_index].scatter(final_point[x_indices],final_point[y_indicies], color = 'red')

# %%
def get_unique_list_of_final_points(results_tuple_list):

    final_points = [results["final_point"] for results in results_tuple_list]
    unique_final_points = get_unique_arrays(final_points)
    return unique_final_points



# %%
def make_plots(information_condition_instance, mae, result_list):

    num_observed_states = len(information_condition_instance.Oset[0])

    x, y = ([0], list(range(num_observed_states)), [0]), ([1], list(range(num_observed_states)), [0])

    ax = fp.plot_strategy_flow(
        mae,
        x, y, flowarrow_points=np.linspace(0.01, 0.99, 9), NrRandom= 128,
        conds=np.array(information_condition_instance.Oset)[0, :num_observed_states],
        # col = 'blac'
    )

    for result in result_list:
        
        xtraj = result["xtraj"]
        final_point = result["final_point"]
        fp.plot_trajectories([xtraj], x, y, cols=['grey'], lss = "--", axes = ax)

        for plot_index, (x_indices,y_indicies) in enumerate(zip(it.product(*x), it.product(*y))):
            ax[plot_index].scatter(final_point[x_indices],final_point[y_indicies], color = 'red')


# %%
def make_plots_only_final_point(information_condition_instance, mae, result_list):

    num_observed_states = len(information_condition_instance.Oset[0])

    x, y = ([0], list(range(num_observed_states)), [0]), ([1], list(range(num_observed_states)), [0])

    ax = fp.plot_strategy_flow(
        mae,
        x, y, flowarrow_points=np.linspace(0.01, 0.99, 9), NrRandom= 128,
        conds=np.array(information_condition_instance.Oset)[0, :num_observed_states],
        # col = 'blac'
    )

    for result in result_list:
        
        final_point = result["final_point"]
        # fp.plot_trajectories([xtraj], x, y, cols=['grey'], lss = "--", axes = ax)

        for plot_index, (x_indices,y_indicies) in enumerate(zip(it.product(*x), it.product(*y))):
            ax[plot_index].scatter(final_point[x_indices],final_point[y_indicies], color = 'red', alpha = 0.7)
        
        plt.show()