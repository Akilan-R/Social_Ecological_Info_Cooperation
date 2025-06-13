# %%
from imports import *


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
def create_mae_POstratAC_eps(information_condition_instance = None, discount_factors=0.98):
    return POstratAC_eps(env=information_condition_instance, learning_rates=0.1, discount_factors=discount_factors)



# %%
all_information_modes = [
        'both_state_and_action_information', 
        'only_action_history_information', 
        'only_state_information', 
        'no_information'
    ]


