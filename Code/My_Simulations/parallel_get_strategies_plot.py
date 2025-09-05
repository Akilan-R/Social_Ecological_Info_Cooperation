# %%

# %%
from imports import *
from information_conditions import Information_Conditions
from base_ecopg import BaseEcologicalPublicGood
from helper_functions import *
from simulation_and_results_functions import *



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
            num_samples = 2,
            make_degraded_state_cooperation_probablity_zero_at_end= True,
            make_degraded_state_obsdist_zero_at_end= True
        )

    make_plots_only_final_point(information_condition_instance, mae_ecopg, result_list)

