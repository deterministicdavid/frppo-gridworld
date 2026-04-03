import numpy as np
from GridworldMDP import GridworldMDP

# Policy iteration algorithm with loops
def policy_evaluation_loopy(policy, mdp: GridworldMDP, theta=1e-8):
    num_states = mdp.num_states
    P = mdp.P
    C = mdp.C
    gamma = mdp.gamma
    V = np.zeros(num_states)  # Value function
    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            V[s] = sum([P[s, policy[s], s_next] * (C[s, policy[s]] + gamma * V[s_next]) for s_next in range(num_states)])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

# Vectorized policy evaluation algorithm
def policy_evaluation(policy, mdp: GridworldMDP, theta=1e-8):
    num_states = mdp.num_states
    P = mdp.P
    C = mdp.C
    gamma = mdp.gamma
    
    V = np.zeros(num_states)  # Value function
    while True:
        delta = 0
        V_prev = np.copy(V)
        
        # Vectorized update for the value function
        # For each state s, we calculate the expected value V[s] for the chosen action policy[s]
        
        # Get transition probabilities and costs for the chosen actions
        P_pi = P[np.arange(num_states), policy]  # P_pi[s, s_next] is the transition probability for policy[s]
        C_pi = C[np.arange(num_states), policy]  # C_pi[s] is the cost for action policy[s]

        # Update the value function: V[s] = sum_over_s_next(P_pi[s, s_next] * (C_pi[s] + gamma * V[s_next]))
        V = np.sum(P_pi * (C_pi[:, np.newaxis] + gamma * V_prev[np.newaxis, :]), axis=1)
        
        # Check for convergence
        delta = np.max(np.abs(V - V_prev))
        if delta < theta:
            break
    
    return V

def test_policy_evaluation():
    # Create a simple MDP instance
    mdp = GridworldMDP(grid_size=11, gamma=0.9)  # Example gridworld MDP, adjust parameters as needed

    # Create a random policy for testing (just random actions for each state)
    num_states = mdp.num_states
    num_actions = mdp.num_actions
    random_policy = np.random.randint(num_actions, size=num_states)

    # Run both the non-vectorized and vectorized versions
    V_non_vectorized = policy_evaluation_loopy(random_policy, mdp)
    V_vectorized = policy_evaluation(random_policy, mdp)

    # Define tolerance for floating-point comparison
    tolerance = 1e-6

    # Compare the results
    max_diff = np.max(np.abs(V_non_vectorized - V_vectorized))
    if max_diff < tolerance:
        print(f"Both versions produce the same output within tolerance {tolerance}. Max difference: {max_diff}")
    else:
        print(f"Outputs differ! Max difference: {max_diff}")


def policy_improvement(V, mdp: GridworldMDP):
    num_states = mdp.num_states
    P = mdp.P
    C = mdp.C
    gamma = mdp.gamma
    actions = mdp.actions
    policy = np.zeros(num_states, dtype=int)  # Initial policy
    for s in range(num_states):
        action_values = np.zeros(len(actions))
        for a in range(len(actions)):
            action_values[a] = sum([P[s, a, s_next] * (C[s, a] + gamma * V[s_next]) for s_next in range(num_states)])
        policy[s] = np.argmin(action_values)
    return policy

def policy_iteration(mdp: GridworldMDP):
    num_states = mdp.num_states
    policy = np.ones(num_states, dtype=int)
    i = 0
    while True:
        print(f"Policy iteration step {i} ", end='')

        V = policy_evaluation(policy, mdp)
        max_V = np.max(abs(V))
        print(f"Max V: {max_V}")
        
        new_policy = policy_improvement(V, mdp)
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy
        i = i + 1       
    return policy, V

if __name__ == "__main__":
    # run tests
    test_policy_evaluation()
    
    # setup the MDP
    mdp = GridworldMDP(grid_size=11, gamma=0.99)
    
    # run policy iteration on strict policies
    optimal_policy, V = policy_iteration(mdp)

    # plot the resulting value function
    mdp.heatmap_plot_V(V)
