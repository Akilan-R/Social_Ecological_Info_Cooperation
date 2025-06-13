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

def get_average_cooperativeness(policy, obsdist, mode, Oset, exclude_degraded_state_for_average_cooperation):
    
    if exclude_degraded_state_for_average_cooperation:
        if mode == 'only_state_information' or mode == 'both_state_and_action_information':
            obsdist = exclude_degraded_states_from_obsdist(obsdist, Oset)

        
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

def enumerate_strategies_frequency(list_of_final_points):

    strategy_shape = list_of_final_points[0].shape
    total_number_of_strategies = len(list_of_final_points)

    flattened_strategies = [tuple(point.flatten()) for point in list_of_final_points]
    #flatten and make it to tuple, otherwise np.unqiue will do element wise
    unique_strategies, counts = np.unique(flattened_strategies, axis = 0, return_counts=True)
    #axis = 0 so that we are looking for unique rows (Hhere tuples rather than)
    # formatted_strategies = [np.array2string(np.array(strategy).reshape(strategy_shape), max_line_width = np.inf) for strategy in unique_strategies]
    strategy_frequencies = pd.DataFrame([{'strategy': strategy.reshape(strategy_shape), 'frequency': (count/total_number_of_strategies)*100} for strategy, count in zip(unique_strategies, counts)])
    #reshape, then array2string to make it human readable, then dataframe for better visualization
    return strategy_frequencies
    



