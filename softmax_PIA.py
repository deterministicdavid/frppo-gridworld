import numpy as np
from GridworldMDP import GridworldMDP

# For 64-bit floating-point numbers, 
# the largest value x such that np.exp(x) does not return infinity 
# is approximately: x = 709.78 while the smallest value that doesn't return 0 is about -700 

MAX_EXP_INPUT = 700.0
MIN_EXP_INPUT = -700.0

# the mirror map from log policy to policy
def mirrormap_pi(z):
    z_max = np.max(z)  # Find the maximum value in z for stability
    z = z - z_max
    exp_z = np.exp(z) # We'll renormalise in a bit so doesn't change things
    if np.isnan(exp_z).any() or np.isinf(exp_z).any():
        raise ValueError("Array exp_z contains NaN, inf, or -inf values.")
    return exp_z / np.sum(exp_z)


# Vectorized Log policy evaluation using softmax policies
def log_policy_evaluation_softmax(log_policy, V_old, mdp: GridworldMDP, tau, theta):
    # we add a small epsilon to the policy before doing log to avoid -inf values
    epsilon = 1e-25
    
    P = mdp.P
    C = mdp.C
    gamma = mdp.gamma
    
    V = np.copy(V_old)  # Value function
    while True:
        # Apply mirrormap_pi to all states at once
        policies = np.apply_along_axis(mirrormap_pi, 1, log_policy)
        
        # Compute the expected reward and transition based on the current policy
        Q = C + gamma * np.sum(P * V[np.newaxis, np.newaxis, :], axis=-1)
        
        # Include the entropy regularization term tau * log(policy[a])
        log_policies = np.log(policies+epsilon)
        softmax_term = tau * log_policies
        
        # Compute the new value function: V[s] = sum(policy[a] * (Q[s, a] + softmax_term[s, a]))
        V_new = np.sum(policies * (Q + softmax_term), axis=1)

        # Check for convergence
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            break
    
    if np.isnan(V).any() or np.isinf(V).any():
        print("Array V contains NaN, inf, or -inf values.")

    return V



def improving(old_V, new_V):
    n = old_V.shape[0]
    improving = True
    for s in range(n):
        # we are minimizing, so everything is fine if new_V <= old_V
        # if not, we want to know. 
        if old_V[s] < new_V[s]:
            # raise ValueError("Value function has not decreased at state %s" % s)
            err = new_V[s] - old_V[s]
            print(f"Value function has not decreased at state {s}, error is = {err}")
            improving = False
    return improving

# Compute Q function and advantage function
def calculate_A_and_Q(V, mdp: GridworldMDP):
    # num_states = mdp.num_states
    # num_actions = mdp.num_actions
    P = mdp.P
    C = mdp.C
    gamma = mdp.gamma
    
    # Vectorized Q calculation:
    # Q[s, a] = C[s, a] + sum_over_s_next(P[s, a, s_next] * gamma * V[s_next])
    Q = C + gamma * np.sum(P * V[np.newaxis, np.newaxis, :], axis=-1)
    
    # Vectorized Advantage calculation:
    # A[s, a] = Q[s, a] - V[s]
    A = Q - V[:, np.newaxis]

    if np.isnan(A).any() or np.isinf(A).any():
            print("Array advantage contains NaN, inf, or -inf values.")

    return A, Q


# Compute Q function and advantage function
# this is only for testing purposes 
def calculate_A_and_Q_loopy(V, mdp: GridworldMDP):
    num_states = mdp.num_states
    num_actions = mdp.num_actions
    P = mdp.P
    C = mdp.C
    gamma = mdp.gamma
    
    A = np.zeros((num_states, num_actions))
    Q = np.zeros((num_states, num_actions))
    for s in range(num_states):
        for a in range(num_actions):
            Q[s,a] = C[s,a] + sum([P[s, a, s_next] * gamma * V[s_next] for s_next in range(num_states)])
            A[s,a] = Q[s,a] - V[s]
    
    if np.isnan(A).any() or np.isinf(A).any():
            print("Array advantage contains NaN, inf, or -inf values.")

    return A, Q

def test_calculate_A_and_Q(mdp):
    # Generate a random value function V to test the functions
    V = np.random.rand(mdp.num_states)
    
    # Call both the vectorized and loopy versions
    A_vec, Q_vec = calculate_A_and_Q(V, mdp)
    A_loopy, Q_loopy = calculate_A_and_Q_loopy(V, mdp)
    
    # Define the tolerance for floating-point comparisons
    tolerance = 1e-6
    
    # Check if the results for Q are the same within the tolerance
    Q_diff = np.max(np.abs(Q_vec - Q_loopy))
    if Q_diff < tolerance:
        print("Q functions match within tolerance.")
    else:
        print(f"Q functions differ by {Q_diff} (greater than tolerance).")

    # Check if the results for A are the same within the tolerance
    A_diff = np.max(np.abs(A_vec - A_loopy))
    if A_diff < tolerance:
        print("A functions match within tolerance.")
    else:
        print(f"A functions differ by {A_diff} (greater than tolerance).")
    
    # Final assertion for automated testing (optional)
    assert Q_diff < tolerance, f"Q function difference {Q_diff} exceeds tolerance."
    assert A_diff < tolerance, f"Advantage function difference {A_diff} exceeds tolerance."


# Log policy iteration
def log_policy_iteration_softmax(mdp: GridworldMDP, tau, tolerance = 1e-6, V_old=None):
    theta = 1e-8 # tolerance for policy evaluation step 
    num_states = mdp.num_states
    num_actions = mdp.num_actions
        
    if not(isinstance(V_old, np.ndarray)):
        log_policy = np.random.uniform(low=-1.0, high=1.0, size=(num_states, num_actions)) #  np.zeros((num_states, num_actions))  
        V_old = log_policy_evaluation_softmax(log_policy, V_old=np.zeros(num_states), mdp=mdp, tau=tau, theta=theta)

    MAX_POLICY_ITERATION_STEPS = 200
    for i in range(0,MAX_POLICY_ITERATION_STEPS):
        print(f"Policy iteration step {i}", end='')        
        A, _ = calculate_A_and_Q(V_old, mdp)

        log_policy = -(1.0/tau)*A # - np.log(clipped_sum_exp)
        
        V = log_policy_evaluation_softmax(log_policy, V_old=V_old, mdp=mdp, tau=tau, theta=theta)
        if np.isnan(V).any() or np.isinf(V).any():
            print("Array V contains NaN, inf, or -inf values.")

        diff = np.max(np.abs(V - V_old))
        print(f"Max diff: {diff}")
        if diff < tolerance:  # Check convergence of value functions in l infty norm
            break

        # if not improving(old_V=V_old, new_V=V):
        #     print("Policy improvement step hasn't improved the value function")

        V_old = V
    
    return log_policy, V


def calculate_log_sum_exp(Z):
    max_Z = np.max(Z, axis=1)
    Z_minus_max = Z - max_Z[:, np.newaxis]
    log_sum_exp = np.log(np.sum(np.exp(Z_minus_max), axis=1)) + max_Z
    # sum_exp = np.sum(np.exp(-(1.0/tau) * Q), axis=1)
    if np.isnan(log_sum_exp).any() or np.isinf(log_sum_exp).any():
            print("Array sum_exp contains NaN, inf, or -inf values.")

    return log_sum_exp

# Log policy iteration
def log_value_iteration_softmax(mdp: GridworldMDP, tau, tolerance = 1e-6):
    num_states = mdp.num_states
    V_old = np.random.uniform(low=-1.0, high=1.0, size=num_states) # np.zeros(num_states)
    
    MAX_ITERATION_STEPS = 4000
    for i in range(0,MAX_ITERATION_STEPS):
        print(f"Value iteration step {i} ", end='')
        
        # here we really need the Q function as we're interested in the value of the minimum intself of the quantity
        # \sum_a [c(s,a)+\tau \log \pi(a|s) + \gamma\sum_s u(s')P(s'|s,a)] pi(a|s) over policies pi. 
        # Since 
        # \sum_a [c(s,a)+\tau \log \pi(a|s) + \gamma\sum_s u(s')P(s'|s,a)] pi(a|s) = \sum_a [Q(s,a) + \tau \log \pi(a|s)] pi(a|s)
        # we see we want minimum of a linear function of the measure + entropic penalty and for that we know the formula for the minimum 
        _, Q = calculate_A_and_Q(V_old,mdp=mdp)
        
        log_sum_exp = calculate_log_sum_exp(-Q/tau)
        V = -tau*log_sum_exp
        if np.isnan(V).any() or np.isinf(V).any():
            print("Array V contains NaN, inf, or -inf values.")

        diff = np.max(np.abs(V - V_old))
        print(f"Max diff: {diff}")
        if diff < tolerance:  # Check convergence of value functions in l infty norm
            break

        V_old = V
    
    return V



if __name__ == "__main__":
    # setup the MDP
    mdp = GridworldMDP(grid_size=20, gamma=0.85, randomize=5.0/400) # we want on average 5 reward / trap states so randomize = 5 x 1/grid_size^2
    
    # choose softmax tau
    tau = 1e-2
    
    # run tests for vectorised methods
    test_calculate_A_and_Q(mdp)
    # test_vectorized_policy_evaluation(mdp,tau)

    # run policy iteration on log policies
    optimal_log_policy_softmax, V_from_pia = log_policy_iteration_softmax(mdp, tau=tau, tolerance=1e-7)

    # try bootstrap for tau
    optimal_log_policy_softmax2, V_from_pia2 = log_policy_iteration_softmax(mdp, tau=0.1*tau, tolerance=1e-7, V_old=V_from_pia)


    # run value iteration on log policies
    V_from_val_iter = log_value_iteration_softmax(mdp, tau=tau, tolerance=1e-7)

    diff = np.max(np.abs(V_from_val_iter - V_from_pia))
    print(f"Max diff: {diff}")
    
    # plot the resulting value function
    mdp.heatmap_plot_3V(V1=V_from_pia,
                        V2=V_from_val_iter,
                        V3=np.abs(V_from_pia-V_from_val_iter),
                        title1='from s\'max PIA',
                        title2='from value iteration',
                        title3='diff')
