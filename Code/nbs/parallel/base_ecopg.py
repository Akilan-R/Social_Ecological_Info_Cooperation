# %%
from imports import *
from information_conditions import Information_Conditions

# %%
BaseEcologicalPublicGood = partial(
    EcologicalPublicGood,
    N=2,
    f=1.2,
    c=5,
    m=-6,
    qc=0.02,
    qr=0.0001,
    degraded_choice=False
)

# Baseline instance

# Overriding certain arguments


# %%




# %%
all_information_modes = [
        'both_state_and_action_information', 
        'only_action_history_information', 
        'only_state_information', 
        'no_information'
    ]




# %%

def create_mae_ecopg_for_given_mode_POstratAC(mode):
    ecopg = BaseEcologicalPublicGood()
    information_condition_instance = Information_Conditions(ecopg, mode= mode)
    mae_ecopg = POstratAC(env=information_condition_instance, learning_rates=0.05, discount_factors= 0.98)
    return mae_ecopg


def create_mae_ecopg_for_given_mode_POstratAC_expanded(mode, m = -6, discount_factor = 0.98,  f = 1.2):
    ecopg = BaseEcologicalPublicGood(m = m, f = f)
    information_condition_instance = Information_Conditions(ecopg, mode= mode)
    mae_ecopg = POstratAC(env=information_condition_instance, learning_rates=0.05, discount_factors= discount_factor)
    return mae_ecopg