# %%
from imports import *
from information_conditions import Information_Conditions
from base_ecopg import BaseEcologicalPublicGood
from helper_functions import *
from simulation_and_results_functions import *

# %%
all_information_modes = [
        'both_state_and_action_information', 
        'only_action_history_information', 
        'only_state_information', 
        'no_information'
    ]

# %%
# Example usage:
if __name__ == "__main__":
    start = time.time()
    basin_of_attraction_cooperation_results_each_mode = compare_conditions_cooperation_basin_size(num_samples=200, degraded_choice=False)
    end = time.time()
    time_elapsed = end - start
    print("Execution time:", timedelta(seconds = time_elapsed))

    print(basin_of_attraction_cooperation_results_each_mode)

    make_barplots_for_cooperate_basin_size(basin_of_attraction_cooperation_results_each_mode)
    

