import numpy as np
from GridworldMDP import GridworldMDP

def project_onto_simplex(v):
    """
    Projects a given 1D vector v onto the probability simplex using an O(n log n) algorithm.
    """
    v_np = np.asarray(v)
    n = v_np.shape[0]
    
    u = np.sort(v_np)[::-1]
    cssv = np.cumsum(u)
    
    condition = u * np.arange(1, n + 1) > (cssv - 1)
    rho = np.nonzero(condition)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v_np - theta, 0)
    
    return w

def policy_evaluation_stochastic(pi, V_old, mdp: GridworldMDP, tau, theta=1e-8):
    """
    Vectorized policy evaluation for a stochastic policy matrix.
    Adds a KL/entropy penalty term directly to the cost.
    """
    P = mdp.P
    C = mdp.C
    gamma = mdp.gamma
    
    V = np.copy(V_old)
    epsilon = 1e-25
    
    # Calculate the KL penalty / entropy term: tau * log(pi)
    # We compute this once outside the loop since the policy is fixed during evaluation
    penalty = tau * np.log(pi + epsilon)
    
    while True:
        V_prev = np.copy(V)
        
        # Calculate Q[s, a] = C[s, a] + sum_s'(P[s, a, s_next] * gamma * V[s_next])
        Q = C + gamma * np.sum(P * V_prev[np.newaxis, np.newaxis, :], axis=-1)
        
        # V[s] = sum_a pi(a|s) * (Q[s, a] + penalty[s, a])
        V = np.sum(pi * (Q + penalty), axis=1)
        
        delta = np.max(np.abs(V - V_prev))
        if delta < theta:
            break
            
    return V

# Get advantage table A
def get_advantage(V, mdp: GridworldMDP):
    P = mdp.P
    C = mdp.C
    gamma = mdp.gamma
    
    # Compute standard Q function
    Q = C + gamma * np.sum(P * V[np.newaxis, np.newaxis, :], axis=-1)
    
    # Compute A (Advantage)
    A = Q - V[:, np.newaxis]
    
    return A

# FR2 update
def fr2_update(pi_old, old_V, mdp: GridworldMDP, tau, h):
    epsilon = 1e-25
    
    # 1. Evaluate the stochastic policy to get the regularized value function V
    V = policy_evaluation_stochastic(pi_old, old_V, mdp, tau, theta=1e-8)
    
    # 2. Calculate advantage using the evaluated V
    A = get_advantage(V=V, mdp=mdp)
    
    # 3. Add the KL gradient (tau * log(pi)). 
    # (Note: the +1 from the derivative of x*log(x) is a uniform shift 
    # and safely vanishes when we project onto the simplex)
    A_reg = A + tau * np.log(pi_old + epsilon)
    
    # 4. Move in the direction of NEGATIVE regularized advantage (minimizing costs)
    pi_new = pi_old - h * A_reg

    # 5. Do L^2 projection onto prob simplex across the action dimension (axis=1)
    pi_new = np.apply_along_axis(project_onto_simplex, 1, pi_new)    
    
    return pi_new, V

def policy_fr2_stepping(mdp: GridworldMDP, tau: float, h:float = 1.0, grad_time_T: float = 10, annealing: bool=False, tau_min:float=0.0):
    num_states = mdp.num_states
    num_actions = mdp.num_actions
    tau_init = tau
    
    V_old = np.zeros(num_states)
    
    # Initialize to a valid uniform policy over actions
    pi_old = np.ones((num_states, num_actions)) / num_actions
    
    MAX_ITERATION_STEPS = np.floor(grad_time_T/h).astype(np.int32)
    for i in range(0, MAX_ITERATION_STEPS):
        if annealing:
            tau = max(tau_init / (1+i*h), tau_min)

        print(f"Policy FR2 step {i+1} out of {MAX_ITERATION_STEPS} with h={h}, T={grad_time_T} ", end='')
        
        pi, V = fr2_update(pi_old, V_old, mdp, tau=tau, h=h)
        
        diff = np.max(np.abs(V - V_old))
        print(f"Max diff: {diff}")
        
        V_old = V
        pi_old = pi
    
    # Update the value function one last time for the final policy we found 
    V = policy_evaluation_stochastic(pi_old, V_old, mdp, tau, theta=1e-8)

    return pi_old, V

if __name__ == "__main__":
    from PIA import policy_iteration
    from softmax_PIA import log_policy_iteration_softmax, log_value_iteration_softmax
    
    # Setup the mdp
    mdp = GridworldMDP(grid_size=11, gamma=0.85, randomize=0.0)

    # Run strict policy iteration 
    _, optimal_value_pia = policy_iteration(mdp)

    # Set the temperature parameter for softmax
    tau = 1e-2
    
    
    # Run policy iteration with softmax
    _, optimal_value_soft_pol_iter = log_policy_iteration_softmax(mdp, tau=tau)

    # Run FR2 stepping
    _, V_fr2 = policy_fr2_stepping(mdp, tau=tau, h=0.1, grad_time_T=12)
    
    # Diff
    V_diff = np.abs(V_fr2 - optimal_value_soft_pol_iter)

    mdp.heatmap_plot_3V(V1=optimal_value_soft_pol_iter, 
                    V2=V_fr2,
                    V3=V_diff,
                    title1='Value fn from strict policy iteration',
                    title2='Value fn from FR2 iteration',
                    title3='Diff',
                    fig_filename='fr2_result_heatmap.pdf'
                    )