# %%

# %%
from imports import *
from information_conditions import Information_Conditions
from base_ecopg import BaseEcologicalPublicGood
from helper_functions import *
from simulation_and_results_functions import *


# %%
np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# 1. No information: all defect with some margin of error
# 2. Only State - all coop or all defect (with margin of error)
# 3. Only action history -
# 4. Both ecological and 

# %%
mode = "only_action_history_information"

ecopg = BaseEcologicalPublicGood()
information_condition_instance = Information_Conditions(ecopg, mode= mode)
mae_ecopg = POstratAC_eps(env=information_condition_instance, learning_rates=0.05, discount_factors= 0.98)

if __name__ == "__main__":

    result_list = run_simulation_across_conditions_parallel(
        mae = mae_ecopg, 
        mode = mode,
        num_samples = 25, 
        exclude_degraded_state_for_average_cooperation = False
    )

    final_point_test = result_list[0]['final_point']

    strategy_list_degraded_zero = [np.round(make_degraded_state_cooperation_probablity_zero(result['final_point'], information_condition_instance.Oset[0]),2) for result in result_list]

    list_of_avg_coop = [float(np.round(float(result['avg_coop']),2)) for result in result_list]

    

    unique_strategies_and_frequency = enumerate_strategies_frequency(strategy_list_degraded_zero)
    unique_strategies_and_frequency.to_csv('only_action_unique_strategies_and_frequency.csv', index=False)

