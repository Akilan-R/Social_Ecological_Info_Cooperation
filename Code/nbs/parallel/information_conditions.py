# %%
from imports import *


# %%
 
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
from pyCRLD.Environments.HistoryEmbedding import HistoryEmbedded

# %%
#
class Information_Conditions(HistoryEmbedded):
    def __init__(self, ecopg , mode):


        super().__init__(ecopg, h=(1, 1, 1))

        self.mode = mode
        self.configure_information_condition()

    def configure_information_condition(self):
        """
        Set the observation mode and configure the observation tensor, Oset, and other properties.
        Modes: 'state', 'action', 'none', 'state+action'
        """
        if self.mode == "only_state_information":
            self._configure_state()
        elif self.mode == "only_action_history_information":
            self._configure_action()
        elif self.mode == "no_information":
            self._configure_none()
        elif self.mode == "both_state_and_action_information":
            self._configure_state_action()
        else:
            raise ValueError("Invalid mode..")
        # self._print_configuration()

        self.Q = self.O.shape[2]

    def _configure_state(self):
        def generate_state_tensor(state_set, observation_set):
            state_tensor = np.zeros((2, len(state_set), len(observation_set)), dtype=int)
            for i in range(2):
                for j, state in enumerate(state_set):
                    for k, observation in enumerate(observation_set):
                        if state.endswith(observation):
                            state_tensor[i, j, k] = 1
            return state_tensor
        
        
        self.Oset = generate_state_observation_set(self.Sset, 2)

        self.O = generate_state_tensor(
            self.Sset,  self.Oset[0])
        

    def _configure_action(self):
        def generate_action_tensor(state_set, action_set):
            action_tensor = np.zeros((2, len(state_set), len(action_set)), dtype=int)
            for i in range(2):  
                for j, state in enumerate(state_set):
                    for k, action in enumerate(action_set):
                        if action[:3] == state[:3]:
                            action_tensor[i, j, k] = 1
            return action_tensor

        self.Oset = generate_action_history_observation_set(self.Sset, self.N)
        self.O = generate_action_tensor(self.Sset, self.Oset[0])

    def _configure_none(self):
        def generate_none_tensor():
            return np.ones((2, 8, 1), dtype=int)

        self.O = generate_none_tensor()
        self.Oset = [['.'], ['.']]

    def _configure_state_action(self):
        # This assumes the default state+action information in `ecopg_with_history`
        pass
          # No modification needed; default setup already uses state+action information.


    def _print_configuration(self):
        print(f"Mode: {self.mode}")
        # print("Observation Tensor:\n", self.O)
        # print("Observation Set:", self.Oset)
        # print("O shape", self.O.shape)
        # print("Q shape", self.Q)
        print("Oset:")

        print("------\n")


# %%



