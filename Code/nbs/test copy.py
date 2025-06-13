
import imports.ipynb
from information_conditions.ipynb import Information_Conditions
from base_ecopg.ipynb import BaseEcologicalPublicGood
from helper_functions.ipynb import *
from simulation_and_results_functions.ipynb import *



def compare_conditions_cooperation_basin_size(num_samples= 5, degraded_choice = False, m_value = -6, discount_factor = 0.98, exclude_degraded_state_for_average_cooperation = True , information_modes = all_information_modes):
    """
    Runs simulations for different information conditions and outputs 
    the results for each condition.
    
    Parameters:
        ecopg (EcologicalPublicGood): An instance of the ecological public good model.
        num_samples (int): Number of initial conditions to sample.
        Tmax (int): Maximum time steps for trajectory simulation.
        tolerance (float): Convergence tolerance for fixed point detection.
        
    Returns:
        None (prints the output summaries for each information condition)
    """

    print(locals())
    
    basin_of_attraction_cooperation_results = {}
    
    
    ecopg = BaseEcologicalPublicGood(m = m_value, degraded_choice=degraded_choice)

    for mode in information_modes:
        # Initialize the information condition
        information_condition_instance = Information_Conditions(ecopg, mode=mode)
        mae = POstratAC_eps(env=information_condition_instance, learning_rates=0.01, discount_factors= discount_factor)

        # Data storage

        # print(f"\nMode: {mode}")

        avg_coop_time_pairs = run_simulation_across_conditions(
            mae = mae, 
            mode = mode,
            num_samples = num_samples, 
            exclude_degraded_state_for_average_cooperation = exclude_degraded_state_for_average_cooperation
        )

        cooperation_basin_size = get_results_only_cooperation_basin_of_attraction_size(avg_coop_time_pairs)

        basin_of_attraction_cooperation_results[mode] = cooperation_basin_size


    return basin_of_attraction_cooperation_results
    

# Example usage:
data = compare_conditions_cooperation_basin_size(degraded_choice=False)
