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

if __name__ == "__main__":

    for mode in all_information_modes:
        ecopg = BaseEcologicalPublicGood()
        information_condition_instance = Information_Conditions(ecopg, mode= mode)
        mae_ecopg = POstratAC(env=information_condition_instance, learning_rates=0.05, discount_factors= 0.98)
 
        print("Running simulation with mode:", mode)

        result_list = run_simulation_across_conditions_parallel(
            mae = mae_ecopg, 
            num_samples = 1000,
            make_degraded_state_cooperation_probablity_zero_at_end= True,
            make_degraded_state_obsdist_zero_at_end= True
        )
        strategy_shape = result_list[0]['final_point'].shape
        result_list_with_degraded_zero_and_flattened_list = [
            {'final_point': tuple(np.round(results['final_point'], 1).flatten()),
                'avg_coop': float(np.round(float(results['avg_coop']), 2)),
                 'obsdist': np.round(results['obsdist'],2)
            } 
            for results in result_list]
        
        # list_of_avg_coop = [float(np.round(float(result['avg_coop']),2)) for result in result_list]
        
        unique_strategies_and_frequency_df, original_dataframe = create_strategy_frequncy_table(result_list_with_degraded_zero_and_flattened_list, strategy_shape)

        filename = f"{mode}_unique_strategies_and_frequency.xlsx"
        unique_strategies_and_frequency_df.to_excel(filename, index=False)  

    


