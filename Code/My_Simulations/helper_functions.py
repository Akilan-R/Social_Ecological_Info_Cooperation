# %%
from imports import *

# %%
def average_policy_for_given_observation_set(X, O):
    """
    Takes in the strategy wrt to complete state set as input and returns the average strategy wrt to the given observation set. For example Strategty with respect to state set might
    have c, c, g = 0.2 and c, c, p = 0.8. But the observation set will be c, c, if only actions are observed. Then the average strategy for this observation set would be 0.5

    """
    A, S, F = X.shape        # Number of layers, states, and features
    _, _, num_obs = O.shape   # Number of observations

    # Initialize the output matrix with zeros
    output = np.zeros((A, num_obs, F))

    for i in range(A):
        for obs in range(num_obs):
            current_mask = O[i, :, obs]  # Shape: (S,)

            # Select rows from X where mask is 1
            selected_X = X[i][current_mask == 1]  # Shape: (num_selected_states, F)
            if selected_X.size > 0:
                mean_vector = selected_X.mean(axis=0)  # Shape: (F,)
            else:
                mean_vector = np.zeros(F)  # Default to zero vector
            output[i, obs, :] = mean_vector

    return output

 
def generate_action_history_observation_set(stateset, number_of_agents):
    action_histories = [state[:3] for state in stateset]
    unique_action_histories = sorted(list(set(action_histories)))
    Oset = [unique_action_histories.copy() for _ in range(number_of_agents)]
    return Oset


def generate_state_observation_set(stateset, number_of_agents):
    state_histories = [state[4:] for state in stateset]
    unique_state_histories = sorted(list(set(state_histories)))
    Oset = [unique_state_histories.copy() for _ in range(number_of_agents)]
    return Oset






# %%
def lhs_sampling(no_of_states, number_of_samples, agents):
    global global_seed
    number_of_dimensions_for_sampling = agents*no_of_states
    sampler = qmc.LatinHypercube(d= number_of_dimensions_for_sampling, seed = global_seed)

    # Sampling for each agent and stacking similar result lists
    lhs_random_samples_list = sampler.random(number_of_samples)
    result = [np.stack((random_samples, 1 - random_samples), axis=-1) for random_samples in lhs_random_samples_list]
    reshaped_result = [random_samples.reshape(agents,no_of_states,2) for random_samples in result]
    
    # cross_product = [np.stack((x, y), axis=0) for x, y in it.combinations_with_replacement(result, agents)]

    return reshaped_result


# %%
def lhs_sampling_structured(no_of_states, number_of_samples, agents):
    global global_seed
    sampler = qmc.LatinHypercube(d=no_of_states, seed = global_seed)

    # Sampling for each agent and stacking similar result lists
    lhs_random_samples_list = sampler.random(number_of_samples)
    result = [np.stack((random_samples, 1 - random_samples), axis=-1) for random_samples in lhs_random_samples_list]
    cross_product = [np.stack((x, y), axis=0) for x, y in it.combinations_with_replacement(result, agents)]

    return cross_product

# %%
def make_degraded_state_cooperation_probablity_zero(initial_condition, Oset):

    degraded_mask = jnp.array(['g' in label for label in Oset])
    initial_condition[:, degraded_mask, 0] = 0
    initial_condition[:, degraded_mask, 1] = 1

    return initial_condition


# %%
def exclude_degraded_states_from_obsdist(obsdist, Oset):
   
        # Exclude degraded states from the observation distribution

    degraded_mask = jnp.array(['g' in label for label in Oset])
    obsdist = jnp.where(degraded_mask, 0, obsdist)

    # Normalize rows to ensure sum of probabilities is 1
    # row_sums = jnp.sum(obsdist, axis=1, keepdims=True)
    # obsdist_without_degraded_state = jnp.where(row_sums > 0, obsdist / row_sums, obsdist)  # Avoid division by zero

    return obsdist

# %%

def get_average_cooperativeness(policy, obsdist):
        
    policy_cooperation_probabilities = policy[:,:, 0]
    agent_index, state_index = [0, 1]

    average_cooperation_for_each_agent = jnp.einsum(policy_cooperation_probabilities, [agent_index, state_index], obsdist, [agent_index, state_index], [agent_index])
    
    return average_cooperation_for_each_agent

# %%

def get_unique_arrays(list_of_arrays):
    """

    Args:
        arrays (list of np.ndarray): List of NumPy arrays to check for uniqueness.

    Returns:
        list of np.ndarray: A list of unique arrays restored to their original shapes.
    """
    list_of_unique_arrays = [list_of_arrays[0]]  #the first element of array is always uniqyue
    for array in list_of_arrays:
        is_unqiue = True
        for unique_array in list_of_unique_arrays:
            if  np.allclose(array, unique_array, rtol = 0.01) or  np.allclose(array, np.flip(unique_array, axis = 0), rtol = 0.01):
                is_unqiue = False
                break
        if is_unqiue:            
            list_of_unique_arrays.append(array)
    return list_of_unique_arrays


# %% 

# def create_strategy_frequncy_table(list_of_final_points):

#     strategy_shape = list_of_final_points[0].shape
#     total_number_of_strategies = len(list_of_final_points)

#     flattened_strategies = [tuple(point.flatten()) for point in list_of_final_points]
#     unique_strategies, counts = np.unique(flattened_strategies, axis = 0, return_counts=True)
   
#     strategy_frequencies = pd.DataFrame([{'strategy': strategy.reshape(strategy_shape), 'frequency': (count/total_number_of_strategies)*100} for strategy, count in zip(unique_strategies, counts)])
#     strategy_frequencies = strategy_frequencies.sort_values(by='frequency', ascending=False).reset_index(drop=True)

#     return strategy_frequencies

def create_strategy_frequncy_table(results_list_flattened_final_point, strategy_shape):

    total_number_of_strategies = len(results_list_flattened_final_point)
    dataframe =  pd.DataFrame(results_list_flattened_final_point)

    grouped_df = dataframe.groupby('final_point').agg(
        avg_coop =  ('avg_coop', 'mean'),
        avg_obsdist = ('obsdist', lambda x: np.mean(np.stack(x), axis= 0)),
        frequency = ('final_point', lambda x: np.round(len(x)/total_number_of_strategies* 100, 2))
        ).reset_index()
    sorted_df = grouped_df.sort_values(by = 'frequency', ascending=False ).reset_index(drop = True)
    sorted_df['final_point'] = sorted_df['final_point'].apply(lambda x: np.array(x).reshape(strategy_shape)).reset_index(drop = True)
    
    dataframe['final_point'] = dataframe['final_point'].apply(lambda x: np.array(x).reshape(strategy_shape)).reset_index(drop = True)
    
    return sorted_df, dataframe
# %%

all_information_modes = [
        "complete",
        "social",
            "ecological",
            "none"]

# %%

def strategy_to_label(strategy, mode, include_mixed_strategies=False):

    # print(strategy)

    strategy = np.array(strategy)
    if mode == 'social' or mode == 'complete':
   

        Oset = np.array(['c,c', 'c,d', 'd,c', 'd,d'])
        if np.all(strategy >= 0.9):
            classification = 'ALL C'

        elif np.all((strategy[np.isin(Oset, ['c,c', 'd,d'])]) >= 0.99) and np.all((strategy[np.isin(Oset, ['c,d', 'd,c'])]) <= 0.01):
            classification = 'WSLS'

        elif np.all((strategy[np.isin(Oset, ['c,c'])]) >= 0.99) and np.all((strategy[np.isin(Oset, ['c,d', 'd,c', 'd,d'])]) <= 0.01):
            classification = 'GT'
        
        elif np.all(strategy <= 0.1):
            classification = 'ALL D'

        elif np.all((strategy[np.isin(Oset, ['d,d'])]) >= 0.99) and np.all((strategy[np.isin(Oset, ['c,d', 'd,c', 'c,c'])]) <= 0.01):
            classification = 'Inv. GT'


        elif include_mixed_strategies:
            strategy_rounded = np.round(strategy)
            #change everything to equal to 1 or 0
           

            if np.all(strategy_rounded == 1):
                classification = 'Almost ALL C'
            elif np.all(strategy_rounded == 0):
                classification = 'Almost ALL D'
            elif np.all((strategy_rounded[np.isin(Oset, ['c,c', 'd,d'])]) == 1) and np.all((strategy_rounded[np.isin(Oset, ['c,d', 'd,c'])]) == 0):
                classification = 'Almost WSLS'
            elif np.all((strategy_rounded[np.isin(Oset, ['c,c'])]) == 1) and np.all((strategy_rounded[np.isin(Oset, ['c,d', 'd,c', 'd,d'])]) == 0):
                    classification = 'Almost GT'
            elif np.all((strategy_rounded[np.isin(Oset, ['d,d'])]) == 1) and np.all((strategy_rounded[np.isin(Oset, ['c,d', 'd,c', 'c,c'])]) == 0):
                    classification = 'Almost Inv. GT'
        
            else:
                classification = 'other'

        else:
            classification = "other"

    if mode == 'ecological' or mode == 'none':

        if np.all(strategy >= 0.9):
            classification = 'ALL C'
        elif np.all(strategy <= 0.1):
            classification = 'ALL D'

        elif include_mixed_strategies:
            strategy_rounded = np.round(strategy)
            #change everything to equal to 1 or 0
            if np.all(strategy_rounded == 1):
                classification = 'ALL C'
            elif np.all(strategy_rounded == 0):
                classification = 'ALL D'
            else:
                classification = 'other'
        else:
            classification = "other"

    return classification


def classify_final_point(final_point):
    final_point_only_coop = final_point[:,:,0]
    agent_1_strategy, agent_2_strategy = final_point_only_coop[0, :], final_point_only_coop[1, :]
    agent_1_classification, agent_2_classification = classify_strategy(agent_1_strategy), classify_strategy(agent_2_strategy)

    return (agent_1_classification, agent_2_classification)


#%%



#%%
def make_barplots_for_cooperate_basin_size(basin_of_attraction_cooperation_results_each_mode):


        cooperation_basin_size = [(basin_of_attraction_cooperation_results_each_mode[condition]) for condition in all_information_modes]

        print(cooperation_basin_size)

        conditions = [
                "Both Social and Ecological State Information", 
                "Only Social Information", 
                "Only Ecological State Information", 
                "No Information"
            ]

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Information Condition': conditions,
            'Cooperation Basin Size': cooperation_basin_size
        })

        # Define a color palette
        color_map = {
            "Both Social and Ecological State Information": "#4c72b0",  # Muted Blue
            "Only Social Information": "#FFB6C1",  # Muted Pink
            "Only Ecological State Information": "#55a868",  # Muted Green
            "No Information": "#000000"  # Black
        }

        # Create figure
        fig = go.Figure()

        for i, row in plot_df.iterrows():
            condition = row['Information Condition']
            percentage = row['Cooperation Basin Size']
            color = color_map[condition]
            
            if percentage == 0:
                # 1. Actual outline bars for zero values (shown in the plot)
                fig.add_trace(go.Bar(
                    x=[condition], 
                    y=[percentage], 
                    marker=dict(color='rgba(0,0,0,0)', line=dict(color=color, width=4)),
                    text=f"{float(percentage):.1f}%",
                    textposition='outside',
                    textfont=dict(size=15, color='black'),  # Larger, darker percentage text
                    showlegend=False,  # Don't show this in the legend
                    legendgroup=condition  # Group legend with the solid bar
                ))

                # 2. Hidden solid legend bar (only for legend display)
                fig.add_trace(go.Bar(
                    x=[None],  # Invisible bar in the plot
                    y=[None],
                    name=condition,
                    marker=dict(color=color),  # Filled marker for the legend
                    legendgroup=condition  # Matches legend with outline bar
                ))

            else:
                # Normal filled bars
                fig.add_trace(go.Bar(
                    x=[condition], 
                    y=[percentage], 
                    name=condition,
                    marker=dict(color=color),
                    text=f"{float(percentage):.1f}%",
                    textposition='outside',
                    textfont=dict(size=15, color='black')  # Larger, darker percentage text
                ))

        # Update layout for aesthetics
        fig.update_layout(
            yaxis_title="Cooperation Basin Size (%)",
            yaxis=dict(
                range=[0, 100],
                tickfont=dict(size=18)  # Larger, darker y-axis label
            ),
            xaxis=dict(title='', showticklabels=False),
            plot_bgcolor='snow',  # Clean background
            width=500,
            height=675,
            bargap=0,  # Minimize gaps
            legend=dict(
                title="",
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
            font=dict(size=13.5, color='black')
            )
        )

        # Show figure
        fig.show()


