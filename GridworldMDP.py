import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GridworldMDP:
    def __init__(self, grid_size = 11, gamma = 0.8, randomize=0.0):
        # Define states (positions in the grid)
        self.states = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        # Define actions
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        self.grid_size = grid_size
        self.gamma = gamma
        self.P = self.__build_P()
        self.C = self.__build_C(step_cost = 0.1, target_cost = -2, trap_cost = 2, randomize=randomize)

    # Convert (i, j) to state index
    def state_to_index(self,state):
        return state[0] * self.grid_size + state[1]

    # Convert state index to (i, j)
    def index_to_state(self,index):
        return (index // self.grid_size, index % self.grid_size)


    def __build_P(self, slip_prob = 0.1):
        # Initialize the transition probability matrix
        # Define transition probability matrix P
        P = np.zeros((self.num_states, self.num_actions, self.num_states))

        # Define transitions for each action, including stochastic slip in every direction
        intended_prob = 1.0 - 3 * slip_prob  # Probability of intended action

        # Define transitions for each action
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                current_state = (i, j)
                current_index = self.state_to_index(current_state)
                
                for a, action in enumerate(self.actions):
                    # Intended next state based on action
                    if action == 'Up':
                        next_state = (max(i-1, 0), j)
                    elif action == 'Down':
                        next_state = (min(i+1, self.grid_size-1), j)
                    elif action == 'Left':
                        next_state = (i, max(j-1, 0))
                    elif action == 'Right':
                        next_state = (i, min(j+1, self.grid_size-1))
                    
                    next_index = self.state_to_index(next_state)
                    P[current_index, a, next_index] = intended_prob  # Assign intended probability
                    
                    # Slips in other directions
                    slip_states = []
                    if i > 0: slip_states.append((i-1, j))  # Up slip
                    if i < self.grid_size - 1: slip_states.append((i+1, j))  # Down slip
                    if j > 0: slip_states.append((i, j-1))  # Left slip
                    if j < self.grid_size - 1: slip_states.append((i, j+1))  # Right slip

                    for slip_state in slip_states:
                        slip_index = self.state_to_index(slip_state)
                        P[current_index, a, slip_index] += slip_prob

        P = self.__normalize_transition_probabilities(P)
        self.__validate_transition_probabilities(P)
        return P
    
    def __normalize_transition_probabilities(self, P):
        for state in self.states:
            for a, action in enumerate(self.actions):
                p_from_state_given_a_to_any_state = 0.0
                for next_state in self.states:
                    #sum_probs = sum(self.P[(state, a, next_state)] for next_state in self.states)
                    current_state_index = self.state_to_index(state)
                    next_state_index = self.state_to_index(next_state)
                    p = P[current_state_index,a,next_state_index]
                    p_from_state_given_a_to_any_state += p
                
                # ok - so now we have the sum and we re-scale all
                if p_from_state_given_a_to_any_state <= 1e-12:
                    raise ValueError(f"Transition probabilities for state {state} and action {action} sum to 0. Can't even rescale.")
                
                for next_state in self.states:
                    current_state_index = self.state_to_index(state)
                    next_state_index = self.state_to_index(next_state)
                    p = P[current_state_index,a,next_state_index]
                    p_rescaled = p/p_from_state_given_a_to_any_state 
                    P[current_state_index,a,next_state_index] = p_rescaled
        
        return P

    def __validate_transition_probabilities(self, P):
        for state in self.states:
            for a, action in enumerate(self.actions):
                p_from_state_given_a_to_any_state = 0.0
                for next_state in self.states:
                    #sum_probs = sum(self.P[(state, a, next_state)] for next_state in self.states)
                    current_state_index = self.state_to_index(state)
                    next_state_index = self.state_to_index(next_state)
                    p = P[current_state_index,a,next_state_index]
                    p_from_state_given_a_to_any_state += p
                if not (0.999 <= p_from_state_given_a_to_any_state <= 1.001):
                    # raise ValueError(f"Transition probabilities for state {state} and action {action} do not sum to 1. They sum to {sum_probs}.")
                    print(f"P({state}, {action}, . ) sum up to {p_from_state_given_a_to_any_state}, not 1.")
                
        print("MDP girdworld - validating transition probability matrix - OK.")


    def __build_C(self, step_cost = 0.1, target_cost = -2, trap_cost = 2, randomize=0.0):
        C = np.full((self.num_states, self.num_actions), step_cost)
        if randomize == 0.0:
            # The reward for each state-action pair
            goal_state = self.state_to_index((8, 7))
            C[goal_state, :] = target_cost  # Negative cost for reaching the goal (we'll minimise)
            trap_state = self.state_to_index((1, 1))
            C[trap_state, :] = trap_cost  # Penalty for falling into a trap
            trap_state = self.state_to_index((1, 5))
            C[trap_state, :] = trap_cost  # Penalty for falling into a trap
            trap_state = self.state_to_index((6, 4))
            C[trap_state, :] = trap_cost  # Penalty for falling into a trap
        else: 
            # We want each state to be reward or trap with a probability of `randomize`
            for i in range(0, self.grid_size):
                for j in range(0, self.grid_size):
                    if np.random.rand() < randomize:
                        trap_state = self.state_to_index((i, j))
                        C[trap_state, :] = trap_cost   # Penalty for falling into a trap
                    elif np.random.rand() < randomize:
                        goal_state = self.state_to_index((i, j))
                        C[goal_state, :] = target_cost   # Negative cost for reaching the goal (we'll minimise)

        return C




    def heatmap_plot_V(self, V):
        # Reshape the value function for 2D grid
        V_grid = V.reshape((self.grid_size, self.grid_size))

        # Create a heatmap plot of the value function
        plt.figure(figsize=(6, 6))
        plt.imshow(V_grid, cmap='viridis', origin='upper')

        # Add a color bar to show the value scale
        plt.colorbar(label='Value Function')

        # Add labels and title
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.title('Heatmap of the Value Function over Grid')

        plt.show()


    def heatmap_plot_3V(self, V1, V2, V3, title1='fn1', title2='fn2', title3='fn3', fig_filename='heatmap_3V.pdf'):
        """
        Plot three heatmaps of value functions in a single figure with color bars.

        Parameters:
            V1 (numpy array): First value function.
            V2 (numpy array): Second value function.
            V3 (numpy array): Third value function.
            grid_size (int): Size of the grid for reshaping the value functions.
        """

        grid_size = self.grid_size
        # Reshape each value function for 2D grid
        V1_grid = V1.reshape((grid_size, grid_size))
        V2_grid = V2.reshape((grid_size, grid_size))
        V3_grid = V3.reshape((grid_size, grid_size))

        # Create a figure with three subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot each heatmap in its respective subplot and add a color bar
        cax1 = axs[0].imshow(V1_grid, cmap='viridis', origin='upper')
        cbar1 = fig.colorbar(cax1, ax=axs[0])  # Add color bar for the first heatmap
        cbar1.set_label('Value Scale')  # Label for color bar
        axs[0].set_xlabel('Grid X')
        axs[0].set_ylabel('Grid Y')
        axs[0].set_title(title1)

        cax2 = axs[1].imshow(V2_grid, cmap='viridis', origin='upper')
        cbar2 = fig.colorbar(cax2, ax=axs[1])  # Add color bar for the second heatmap
        cbar2.set_label('Value Scale')
        axs[1].set_xlabel('Grid X')
        axs[1].set_ylabel('Grid Y')
        axs[1].set_title(title2)

        cax3 = axs[2].imshow(V3_grid, cmap='viridis', origin='upper')
        cbar3 = fig.colorbar(cax3, ax=axs[2])  # Add color bar for the third heatmap
        cbar3.set_label('Value Scale')
        axs[2].set_xlabel('Grid X')
        axs[2].set_ylabel('Grid Y')
        axs[2].set_title(title3)

        # Layout so plots and color bars do not overlap
        fig.tight_layout()
        plt.savefig(fig_filename)
        plt.close()

    def heatmap_plot_4V(self, V1, V2, V3, V4, title1='fn1', title2='fn2', title3='fn3', title4='fn4'):
            """
            Plot three heatmaps of value functions in a single figure with color bars.

            Parameters:
                V1 (numpy array): First value function.
                V2 (numpy array): Second value function.
                V3 (numpy array): Third value function.
                V4 (numpy array): Fourthvalue function.
            """

            grid_size = self.grid_size
            # Reshape each value function for 2D grid
            V1_grid = V1.reshape((grid_size, grid_size))
            V2_grid = V2.reshape((grid_size, grid_size))
            V3_grid = V3.reshape((grid_size, grid_size))
            V4_grid = V4.reshape((grid_size, grid_size))

            # Create a figure with three subplots
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))

            # Plot each heatmap in its respective subplot and add a color bar
            cax1 = axs[0,0].imshow(V1_grid, cmap='viridis', origin='upper')
            cbar1 = fig.colorbar(cax1, ax=axs[0,0])  # Add color bar for the first heatmap
            cbar1.set_label('Value Scale')  # Label for color bar
            axs[0,0].set_xlabel('Grid X')
            axs[0,0].set_ylabel('Grid Y')
            axs[0,0].set_title(title1)

            cax2 = axs[0,1].imshow(V2_grid, cmap='viridis', origin='upper')
            cbar2 = fig.colorbar(cax2, ax=axs[0,1])  # Add color bar for the second heatmap
            cbar2.set_label('Value Scale')
            axs[0,1].set_xlabel('Grid X')
            axs[0,1].set_ylabel('Grid Y')
            axs[0,1].set_title(title2)

            cax3 = axs[1,0].imshow(V3_grid, cmap='viridis', origin='upper')
            cbar3 = fig.colorbar(cax3, ax=axs[1,0])  # Add color bar for the third heatmap
            cbar3.set_label('Value Scale')
            axs[1,0].set_xlabel('Grid X')
            axs[1,0].set_ylabel('Grid Y')
            axs[1,0].set_title(title3)

            cax4 = axs[1,1].imshow(V4_grid, cmap='viridis', origin='upper')
            cbar4 = fig.colorbar(cax4, ax=axs[1,1])  # Add color bar for the fourth heatmap
            cbar4.set_label('Value Scale')
            axs[1,1].set_xlabel('Grid X')
            axs[1,1].set_ylabel('Grid Y')
            axs[1,1].set_title(title4)

            # Layout so plots and color bars do not overlap
            fig.tight_layout()
            plt.show()



if __name__ == "__main__":
    # setup the MDP
    mdp = GridworldMDP(grid_size=11, gamma=0.95)
    