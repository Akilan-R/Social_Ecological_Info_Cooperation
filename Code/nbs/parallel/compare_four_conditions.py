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

def compare_conditions_cooperation_basin_size(num_samples= 250, degraded_choice = False, m_value = -6, discount_factor = 0.98, make_degraded_state_cooperation_probablity_zero_at_end= True,
            make_degraded_state_obsdist_zero_at_end= True , information_modes = all_information_modes):
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
        mae_ecopg = POstratAC_eps(env=information_condition_instance, learning_rates=0.01, discount_factors= discount_factor)

        # Data storage

        # print(f"\nMode: {mode}")

        avg_coop_time_pairs = run_simulation_across_conditions_parallel(
            mae = mae_ecopg, 
            num_samples = num_samples,
            make_degraded_state_cooperation_probablity_zero_at_end = make_degraded_state_cooperation_probablity_zero_at_end,
            make_degraded_state_obsdist_zero_at_end = make_degraded_state_obsdist_zero_at_end
        )

        cooperation_basin_size = get_results_only_cooperation_basin_of_attraction_size(avg_coop_time_pairs)

        basin_of_attraction_cooperation_results[mode] = cooperation_basin_size


    return basin_of_attraction_cooperation_results
    

# Example usage:
if __name__ == "__main__":
    start = time.time()
    data = compare_conditions_cooperation_basin_size(degraded_choice=False)
    end = time.time()
    time_elapsed = end - start
    print("Execution time:", timedelta(seconds = time_elapsed))

    print(data)
    cooperation_basin_size = [(data[condition]) for condition in all_information_modes]


# Debugging output
    print("Extracted Cooperation Basin Size:", cooperation_basin_size)

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





