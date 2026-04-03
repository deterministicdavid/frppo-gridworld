
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime


from GridworldMDP import GridworldMDP
from softmax_PIA import log_policy_iteration_softmax, log_value_iteration_softmax
from PIA import policy_iteration

from fr_descent import policy_fr2_stepping
from mirror_descent import policy_mirror_stepping



def figure_fr2_convergence2():
    # Setup the mdp
    mdp = GridworldMDP(grid_size=11, gamma=0.95, randomize=0.0)
    tau = 0     
    
    # Run policy iteration with softmax
    # _, optimal_value = log_policy_iteration_softmax(mdp, tau=tau)
    _, optimal_value = policy_iteration(mdp)

    h_fr2 = 0.05
    h_md = 0.2

    steps = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    # steps = [2, 4, 6, 8]
    
    diffs_fr2 = []
    diffs_md = []
    
    for num_steps in steps:
        # Run FR2 and MD stepping
        _, V_fr2 = policy_fr2_stepping(mdp, tau=tau, h=h_fr2, grad_time_T=num_steps*h_fr2)
        _, V_md = policy_mirror_stepping(mdp, tau=tau, h=h_md, grad_time_T=num_steps*h_md)

        V_diff_fr2 = np.max(np.abs(V_fr2 - optimal_value))
        V_diff_md = np.max(np.abs(V_md - optimal_value))
        
        diffs_fr2.append(V_diff_fr2)
        diffs_md.append(V_diff_md)
        print(f"Steps {num_steps} (h={h_fr2}, T={num_steps*h_fr2}) | Max diff FR2: {V_diff_fr2:.6f}")
        print(f"Steps {num_steps} (h={h_md}, T={num_steps*h_md}) | Max diff  MD: {V_diff_md:.6f}")
        
    steps_np = np.array(steps, dtype=float)
    diffs_fr2_np = np.array(diffs_fr2, dtype=float)
    diffs_md_np = np.array(diffs_md, dtype=float)
    
    # Helper function to do the log-space fitting and L^2 error calculation
    def fit_convergence(steps_arr, diffs_arr):
        safe_diffs = np.maximum(diffs_arr, 1e-30) 
        log_diffs = np.log(safe_diffs)
        
        # ---------------------------------------------------------
        # 1. Fit O(1/N) in log space
        # ---------------------------------------------------------
        log_C_inv = np.mean(log_diffs + np.log(steps_arr))
        C_inv = np.exp(log_C_inv)
        
        # Predicted log values: log(C) - log(N)
        log_fit_inv = log_C_inv - np.log(steps_arr)
        fit_inv = np.exp(log_fit_inv) # for linear plotting
        
        # ---------------------------------------------------------
        # 2. Fit O(kappa^N) in log space
        # ---------------------------------------------------------
        m, c = np.polyfit(steps_arr, log_diffs, 1)
        kappa_exp = np.exp(m)
        C_exp = np.exp(c)
        
        # Predicted log values: N * log(kappa) + log(C)
        log_fit_exp = m * steps_arr + c
        fit_exp = np.exp(log_fit_exp) # for linear plotting
        
        # ---------------------------------------------------------
        # Calculate L^2 error in LOG SPACE
        # ---------------------------------------------------------
        l2_inv = np.linalg.norm(log_diffs - log_fit_inv)
        l2_exp = np.linalg.norm(log_diffs - log_fit_exp)
        
        # Lower L^2 error is better
        best_fit_name = "$O(\kappa^N)$" if l2_exp < l2_inv else "$O(1/N)$"
        
        return {
            'safe_diffs': safe_diffs,
            'C_inv': C_inv, 'fit_inv': fit_inv, 'l2_inv': l2_inv,
            'kappa_exp': kappa_exp, 'C_exp': C_exp, 'fit_exp': fit_exp, 'l2_exp': l2_exp,
            'best_fit_name': best_fit_name
        }
    

    # Perform fitting for both datasets
    fit_fr2 = fit_convergence(steps_np, diffs_fr2_np)
    fit_md = fit_convergence(steps_np, diffs_md_np)

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Subplot 1: Unscaled Y-axis ---
    # FR2
    ax1.plot(steps_np, diffs_fr2_np, 'o', color='tab:blue', markersize=6, label='FR2 Empirical')
    ax1.plot(steps_np, fit_fr2['fit_inv'], '--', color='tab:blue', alpha=0.6, 
             label=f"FR2 $O(1/N)$: $C={fit_fr2['C_inv']:.4f}$ ($L^2 err={fit_fr2['l2_inv']:.4f}$)")
    ax1.plot(steps_np, fit_fr2['fit_exp'], '-.', color='tab:blue', 
             label=f"FR2 $O(\kappa^N)$: $\kappa={fit_fr2['kappa_exp']:.4f}$ ($L^2 err={fit_fr2['l2_exp']:.4f}$)")
    
    # MD
    ax1.plot(steps_np, diffs_md_np, 's', color='tab:orange', markersize=6, label='MD Empirical')
    ax1.plot(steps_np, fit_md['fit_inv'], '--', color='tab:orange', alpha=0.6, 
             label=f"MD $O(1/N)$: $C={fit_md['C_inv']:.4f}$ ($L^2 err={fit_md['l2_inv']:.4f}$)")
    ax1.plot(steps_np, fit_md['fit_exp'], '-.', color='tab:orange', 
             label=f"MD $O(\kappa^N)$: $\kappa={fit_md['kappa_exp']:.4f}$ ($L^2 err={fit_md['l2_exp']:.4f}$)")
             
    ax1.set_xlabel('Steps (N)')
    ax1.set_ylabel('$|V^* - V^N|$')
    ax1.set_title(f"Convergence Rate (Linear Scale)\nBest FR2: {fit_fr2['best_fit_name']} | Best MD: {fit_md['best_fit_name']}")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Log Y-axis ---
    # FR2
    ax2.plot(steps_np, np.log(fit_fr2['safe_diffs']), 'o', color='tab:blue', markersize=6, label='FR2 Empirical')
    ax2.plot(steps_np, np.log(fit_fr2['fit_inv']), '--', color='tab:blue', alpha=0.6, 
             label=f"FR2 $O(1/N)$: $C={fit_fr2['C_inv']:.4f}$ ($L^2 err={fit_fr2['l2_inv']:.4f}$)")
    ax2.plot(steps_np, np.log(fit_fr2['fit_exp']), '-.', color='tab:blue', 
             label=f"FR2 $O(\kappa^N)$: $\kappa={fit_fr2['kappa_exp']:.4f}$ ($L^2 err={fit_fr2['l2_exp']:.4f}$)")
    
    # MD
    ax2.plot(steps_np, np.log(fit_md['safe_diffs']), 's', color='tab:orange', markersize=6, label='MD Empirical')
    ax2.plot(steps_np, np.log(fit_md['fit_inv']), '--', color='tab:orange', alpha=0.6, 
             label=f"MD $O(1/N)$: $C={fit_md['C_inv']:.4f}$ ($L^2 err={fit_md['l2_inv']:.4f}$)")
    ax2.plot(steps_np, np.log(fit_md['fit_exp']), '-.', color='tab:orange', 
             label=f"MD $O(\kappa^N)$: $\kappa={fit_md['kappa_exp']:.4f}$ ($L^2 err={fit_md['l2_exp']:.4f}$)")
             
    ax2.set_xlabel('Steps (N)')
    ax2.set_ylabel('$\log |V^* - V^N|$') 
    ax2.set_title(f"Convergence Rate (Log Scale)\nBest FR2: {fit_fr2['best_fit_name']} | Best MD: {fit_md['best_fit_name']}")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Layout and Save
    plt.tight_layout()
    plt.savefig('fr2_error_plot.pdf')
    plt.close()

if __name__ == "__main__":
    figure_fr2_convergence2()
