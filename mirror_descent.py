import numpy as np

from GridworldMDP import GridworldMDP
from softmax_PIA import log_policy_evaluation_softmax, mirrormap_pi


# For testing only: get advantage table A and sum exp correction
def get_advantage_and_log_sum_exp_loopy(log_policy, V, mdp: GridworldMDP, tau, h):
    num_states = mdp.num_states
    num_actions = mdp.num_actions
    P = mdp.P
    C = mdp.C
    gamma = mdp.gamma
    A = np.zeros((num_states,num_actions))
    sum_exp = np.zeros(num_states)
    log_sum_exp = np.zeros(num_states)
    for s in range(num_states):
        # for each state and action first compute advantage
        
        pi_s = mirrormap_pi(log_policy[s])
        max_of_minus_h_times_A = np.max(-h*A[s,:])
        for a in range(num_actions):
            A[s,a] = C[s, a] + sum([P[s, a, s_next] * gamma * V[s_next] for s_next in range(num_states)]) - V[s]
            sum_exp[s] += np.exp(-h*A[s,a]-max_of_minus_h_times_A)*pi_s[a]
        log_sum_exp[s] = np.log(sum_exp[s]) + max_of_minus_h_times_A
    return A, np.log(sum_exp)

# Get advantage table A and sum exp correction
def get_advantage_and_log_sum_exp(log_policy, V, mdp: GridworldMDP, tau, h):
    P = mdp.P
    C = mdp.C
    gamma = mdp.gamma
    
    # Compute pi_s for all states at once using mirrormap_pi for log_policy
    pi_s = np.apply_along_axis(mirrormap_pi, 1, log_policy)
    
    # Compute Q function: Q[s, a] = C[s, a] + sum(P[s, a, s_next] * gamma * V[s_next])
    Q = C + gamma * np.sum(P * V[np.newaxis, np.newaxis, :], axis=-1)
    
    # Compute A (Advantage): A[s, a] = Q[s, a] - V[s]
    A = Q - V[:, np.newaxis]
    
    # Compute sum_exp: sum_exp[s] = sum(exp(-h * A[s, a]) * pi_s[s, a] for a in actions)
    max_vals = np.max(A, axis=1)
    sum_exp = np.sum(np.exp(-h * A - max_vals[:, np.newaxis]) * pi_s, axis=1)
    log_sum_exp = np.log(sum_exp) + max_vals
    return A, log_sum_exp

def test_get_advantage_and_sum_exp(mdp, tau, h):
    # Generate random log_policy and V for testing
    log_policy = np.random.rand(mdp.num_states, mdp.num_actions)
    V = np.random.rand(mdp.num_states)

    # Call both the loopy and vectorized versions
    A_loopy, log_sum_exp_loopy = get_advantage_and_log_sum_exp_loopy(log_policy, V, mdp, tau, h)
    A_vec, log_sum_exp_vec = get_advantage_and_log_sum_exp(log_policy, V, mdp, tau, h)
    
    # Define the tolerance for floating-point comparisons
    tolerance = 1e-6
    
    # Check if the results for A are the same within the tolerance
    A_diff = np.max(np.abs(A_vec - A_loopy))
    if A_diff < tolerance:
        print("Advantage function A matches within tolerance.")
    else:
        print(f"Advantage function A differs by {A_diff} (greater than tolerance).")
    
    # Check if the results for sum_exp are the same within the tolerance
    log_sum_exp_diff = np.max(np.abs(log_sum_exp_vec - log_sum_exp_loopy))
    if log_sum_exp_diff < tolerance:
        print("sum_exp matches within tolerance.")
    else:
        print(f"sum_exp differs by {log_sum_exp_diff} (greater than tolerance).")
    
    # Final assertion for automated testing (optional)
    assert A_diff < tolerance, f"Advantage function difference {A_diff} exceeds tolerance."
    assert log_sum_exp_diff < tolerance, f"sum_exp difference {log_sum_exp_diff} exceeds tolerance."



# Mirror flow RHS
def mirror_f(Z, old_V, mdp: GridworldMDP, tau, h):
    # advantage
    V = log_policy_evaluation_softmax(Z, old_V, mdp, tau, theta=1e-8)
    A, log_sum_exp = get_advantage_and_log_sum_exp(Z, V=old_V, mdp=mdp, tau=tau, h=h)
    
    # add the term arising from entropy in the flat derivative
    A = A + tau * Z
        
    # note that the log-sum-exp term in the mirror stepping is *not* to be multiplied by step size
    # which is why we scale up here... so we can scale down later
    # f = - A - (1.0/h)*log_sum_exp[:, np.newaxis]  
    f = - A

    if np.isnan(f).any() or np.isinf(f).any():
            print("mirror_f contains NaN, inf, or -inf values.")
    
    
    return f, V
    

# Mirror flow RHS
def mirror_f_for_semi_implicit(Z, old_V, mdp: GridworldMDP, tau, h):
    # advantage
    A, log_sum_exp = get_advantage_and_log_sum_exp(Z, V=old_V, mdp=mdp, tau=tau, h=h)
    
    # we don't add the term arising from entropy in the flat derivative
    # that will come later in the semi-implicit step 
        
    # note that the log-sum-exp term in the mirror stepping is *not* to be multiplied by step size
    # which is why we scale up here... so we can scale down later
    f = - A - (1.0/h)*log_sum_exp[:, np.newaxis] 

    if np.isnan(f).any() or np.isinf(f).any():
            print("mirror_f contains NaN, inf, or -inf values.")
    
    V = log_policy_evaluation_softmax(Z, old_V, mdp, tau, theta=1e-8)
    return f, V


# Policy mirror scheme with softmax policies
# z_{n+1} = z_n + h f (z_n)
def policy_mirror_stepping(mdp: GridworldMDP, tau, h=1, grad_time_T=10, annealing=False, tau_min=0.0):
    num_states = mdp.num_states
    num_actions = mdp.num_actions
    tau_init = tau
    
    V_old = np.zeros(num_states)
    Z_old=  np.zeros((num_states,num_actions))
    
    MAX_MIRROR_ITERATION_STEPS = np.floor(grad_time_T/h).astype(np.int32)
    for i in range(0,MAX_MIRROR_ITERATION_STEPS):
        # if annealing is on, update tau 
        if annealing:
            tau = max(tau_init / (1+i*h),tau_min)

        print(f"Policy mirror step {i+1} out of {MAX_MIRROR_ITERATION_STEPS} with h={h}, T={grad_time_T} ", end='')
        
        # the minus for doing descent is already done in mirror_f
        f, V = mirror_f(Z_old, V_old, mdp, tau=tau, h=h)
        Z = Z_old + h * f

        diff = np.max(np.abs(V - V_old))
        print(f"Max diff: {diff}")
        
        V_old = V
        Z_old = Z
    
    # we should update the value function for the final policy we found 
    V = log_policy_evaluation_softmax(Z_old, V_old, mdp, tau, theta=1e-8)

    return Z_old, V

# Policy mirror scheme with softmax policies
# z_{n+1} = z_n + h f (z_n)
def policy_mirror_stepping_semi_implicit(mdp: GridworldMDP, tau, h=1, grad_time_T=10, annealing=False, tau_min=0.0):
    num_states = mdp.num_states
    num_actions = mdp.num_actions
    tau_init = tau
    
    V_old = np.zeros(num_states)
    Z_old=  np.zeros((num_states,num_actions))
    
    MAX_MIRROR_ITERATION_STEPS = np.floor(grad_time_T/h).astype(np.int32)
    for i in range(0,MAX_MIRROR_ITERATION_STEPS):
        # if annealing is on, update tau 
        if annealing:
            tau = max(tau_init / (1+i*h),tau_min)

        print(f"Policy mirror semi-imp step {i+1} out of {MAX_MIRROR_ITERATION_STEPS} with h={h}, T={grad_time_T} ", end='')
        
        f, V = mirror_f_for_semi_implicit(Z_old, V_old, mdp, tau=tau, h=h)
        # Z = (Z_old + h * f)/(1-h*tau)
        Z = (Z_old + h * f)/(1.0+h*tau)
        

        diff = np.max(np.abs(V - V_old))
        print(f"Max diff: {diff}")
        
        V_old = V
        Z_old = Z
    
    # we should update the value function for the final policy we found 
    V = log_policy_evaluation_softmax(Z_old, V_old, mdp, tau, theta=1e-8)

    return Z_old, V

# Policy mirror scheme with softmax policies
# For z' = -\tau z + f(z) we can use the Duhamel formula
# z(t_{n+1}) = z(t_n)e^{-\tau h} + \int_{t_n}^{t_{n+1}} e^{-\tau(t_{n+1}-r)} f(z(r))dr
# taking the value at the left-hand point for the integral over [t_n, t_{n+1}] we get
# z_{n+1} = z_n e^{-\tau h} + f(z_n)) (1/\tau) (1-e^{-\tau h})
# z_{n+1} = z_n + h f (z_n)
def policy_mirror_stepping_exponential_integrator(mdp: GridworldMDP, tau, h=1, grad_time_T=10, annealing=False, tau_min=0.0):
    num_states = mdp.num_states
    num_actions = mdp.num_actions
    tau_init = tau
    
    V_old = np.zeros(num_states)
    Z_old=  np.zeros((num_states,num_actions))
    
    MAX_MIRROR_ITERATION_STEPS = np.floor(grad_time_T/h).astype(np.int32)
    for i in range(0,MAX_MIRROR_ITERATION_STEPS):
        # if annealing is on, update tau 
        if annealing:
            tau = max(tau_init / (1+i*h),tau_min)

        exp_minus_tau_h = np.exp(-tau*h)
        print(f"Policy mirror exp int step {i+1} out of {MAX_MIRROR_ITERATION_STEPS} with h={h}, T={grad_time_T} ", end='')
        
        f, V = mirror_f_for_semi_implicit(Z_old, V_old, mdp, tau=tau, h=h)
        Z = Z_old * exp_minus_tau_h  + (1.0/tau) * (1.0-exp_minus_tau_h) * f
        
        diff = np.max(np.abs(V - V_old))
        print(f"Max diff: {diff}")
        
        V_old = V
        Z_old = Z
    
    # we should update the value function for the final policy we found 
    V = log_policy_evaluation_softmax(Z_old, V_old, mdp, tau, theta=1e-8)

    return Z_old, V


# Policy mirror schme with midpoint stepping and softmax policies
# want
# z_{n+1} = z_n + h f (z_n + 0.5 f(z_n))

def policy_mirror_midpoint_stepping(mdp: GridworldMDP, tau, h = 1.0, grad_time_T=10, annealing=False, tau_min=0.0):
    num_states = mdp.num_states
    num_actions = mdp.num_actions
    tau_init = tau
    
    V_old = np.zeros(num_states)
    Z_old=  np.zeros((num_states,num_actions))
    
    MAX_MIRROR_ITERATION_STEPS = np.floor(grad_time_T/h).astype(np.int32)
    for i in range(0,MAX_MIRROR_ITERATION_STEPS):
        # if annealing is on, update tau 
        if annealing:
            tau = max(tau_init / (1+i*h),tau_min)

        print(f"Policy mirror midpoint step {i+1} out of {MAX_MIRROR_ITERATION_STEPS} with h={h}, T={grad_time_T} ", end='')
        
        # we put in 1/2 the step size and get the policy for that; minus for descent in mirror_f
        f_mid, V_mid = mirror_f(Z_old, V_old, mdp, tau=tau, h=0.5*h)
        Z_mid = Z_old + 0.5*h * f_mid
        
        # and now it's just the mirror step but using the midpoint value for Z
        # minus for descent already in mirror_f
        f, V = mirror_f(Z_mid, V_mid, mdp, tau=tau, h=h)
        Z = Z_old + h * f
        
        diff = np.max(np.abs(V - V_old))
        print(f"Max diff: {diff}")
        
        V_old = V
        Z_old = Z
    
    
    # we should update the value function for the final policy we found 
    V = log_policy_evaluation_softmax(Z_old, V_old, mdp, tau, theta=1e-8)

    return Z_old , V


def policy_mirror_midpoint_stepping_semi_implicit(mdp: GridworldMDP, tau, h = 1.0, grad_time_T=10, annealing=False, tau_min=0.0):
    num_states = mdp.num_states
    num_actions = mdp.num_actions
    tau_init = tau
    
    V_old = np.zeros(num_states)
    Z_old=  np.zeros((num_states,num_actions))
    
    MAX_MIRROR_ITERATION_STEPS = np.floor(grad_time_T/h).astype(np.int32)
    for i in range(0,MAX_MIRROR_ITERATION_STEPS):
        # if annealing is on, update tau 
        if annealing:
            tau = max(tau_init / (1+i*h),tau_min)

        print(f"Policy mirror midpoint semi-imp step {i+1} out of {MAX_MIRROR_ITERATION_STEPS} with h={h}, T={grad_time_T} ", end='')
        
        # we put in 1/2 the step size and get the policy for that; minus for descent in mirror_f
        f_mid, V_mid = mirror_f(Z_old, V_old, mdp, tau=tau, h=0.5*h)
        Z_mid = Z_old + 0.5*h * f_mid
        
        # and now it's just the mirror step but using the midpoint value for Z and semi implicit for tau
        # minus for descent already in mirror_f
        f, V = mirror_f_for_semi_implicit(Z_mid, V_mid, mdp, tau=tau, h=h)
        Z = (Z_old + h*f) / (1.0+h*tau)

        diff = np.max(np.abs(V - V_old))
        print(f"Max diff: {diff}")
        
        V_old = V
        Z_old = Z
    
    
    # we should update the value function for the final policy we found 
    V = log_policy_evaluation_softmax(Z_old, V_old, mdp, tau, theta=1e-8)

    return Z_old , V



# For Runge-Kutta 4th order we want
# z_{n+1} = z_n + (1/6) (k_1 + 2 k_2 + 2 k_3 + k_4), where
# k_1 = h f(z_n)
# k_2 = h f(z_n + (1/2) k_1),
# k_3 = h f(z_n + (1/2) k_2),
# k_4 = h f(z_n + k_3).

def policy_mirror_RK4_stepping(mdp: GridworldMDP, tau, h = 1.0, grad_time_T=10, annealing=False, tau_min=0.0):
    num_states = mdp.num_states
    num_actions = mdp.num_actions
    tau_init = tau
    
    V_old = np.zeros(num_states)
    Z_old=  np.zeros((num_states,num_actions))
    
    MAX_MIRROR_ITERATION_STEPS = np.floor(grad_time_T/h).astype(np.int32)
    for i in range(0,MAX_MIRROR_ITERATION_STEPS):
        # if annealing is on, update tau 
        if annealing:
            tau = max(tau_init / (1+i*h),tau_min)

        print(f"Policy mirror RK4 step {i+1} out of {MAX_MIRROR_ITERATION_STEPS} with h={h}, T={grad_time_T} ", end='')
        
        # k_1 = h f(z_n)
        f_for_k1, V_for_k1 = mirror_f(Z_old, V_old, mdp, tau=tau, h=h)
        k_1 = h * f_for_k1

        # k_2 = h f(z_n + (1/2) k_1),
        f_for_k2, V_for_k2 = mirror_f(Z_old + 0.5*k_1, V_for_k1, mdp, tau=tau, h=h)
        k_2 = h * f_for_k2

        # k_3 = h f(z_n + (1/2) k_2),
        f_for_k3, V_for_k3 = mirror_f(Z_old + 0.5*k_2, V_for_k2, mdp, tau=tau, h=h)
        k_3 = h * f_for_k3

        # k_4 = h f(z_n + k_3).
        f_for_k4, V_for_k4 = mirror_f(Z_old + k_3, V_for_k3, mdp, tau=tau, h=h)
        k_4 = h * f_for_k4

        # z_{n+1} = z_n + (1/6) (k_1 + 2 k_2 + 2 k_3 + k_4)
        Z = Z_old + (k_1  + 2 * k_2  +  2 * k_3  + k_4)/6.0

        V = V_for_k4 # log_policy_evaluation_softmax(Z, V_old, mdp, tau, theta=1e-8)
        diff = np.max(np.abs(V - V_old))
        print(f"Max diff: {diff}")
        
        V_old = V
        Z_old = Z
    
    # we should update the value function for the final policy we found 
    V = log_policy_evaluation_softmax(Z_old, V_old, mdp, tau, theta=1e-8)
    
    return Z_old , V




if __name__ == "__main__":
    from PIA import policy_iteration
    from softmax_PIA import log_policy_iteration_softmax, log_value_iteration_softmax
    
    # Setup the mdp
    mdp = GridworldMDP(grid_size=11, gamma=0.85, randomize=0.0)

    # Run strict policy iteration 
    optimal_policy_strict, optimal_value_pia = policy_iteration(mdp)

    # Set the temperature parameter for softmax
    tau = 1e-2
    
    # Run the test to compare loopy and vectorized methods
    test_get_advantage_and_sum_exp(mdp, tau, h=0.5)

    # Run policy iteration with softmax
    _, optimal_value_soft_pol_iter = log_policy_iteration_softmax(mdp, tau=tau)

    # Run value iteration with softmax
    _, optimal_value_mirror_semi_imp = policy_mirror_stepping_semi_implicit(mdp, tau=tau, h = 0.5, grad_time_T=6)

    # Run mirror stepping
    _, V_mirror = policy_mirror_stepping(mdp, tau=tau, h=0.5, grad_time_T=6)
    _, V_mirror_midpt = policy_mirror_midpoint_stepping(mdp, tau=tau, h=0.5, grad_time_T=6)

    mdp.heatmap_plot_4V(V1=optimal_value_soft_pol_iter, 
                    V2=optimal_value_mirror_semi_imp, 
                    V3=V_mirror,
                    V4=V_mirror_midpt,
                    title1='Value fn from soft policy iteration',
                    title2='Value fn from mirror descent semi-implicit on s\'max policies',
                    title3='Value fn from mirror descent on s\'max policies',
                    title4='Value fn from midpt mirror descent on s\'max policies',
                    )

