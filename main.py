
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime


from GridworldMDP import GridworldMDP
from softmax_PIA import log_policy_iteration_softmax, log_value_iteration_softmax
from PIA import policy_iteration

from fr_descent import policy_fr2_stepping
from mirror_descent import policy_mirror_stepping



def figure_fr2_convergence():
    
    
    # Setup the mdp
    mdp = GridworldMDP(grid_size=11, gamma=0.85, randomize=0.0)
    tau = 1e-2     
    
    # Run policy iteration with softmax
    _, optimal_value_soft_pol_iter = log_policy_iteration_softmax(mdp, tau=tau)

    h=0.05
    steps = [2,4,6,8,10,12,14,16,18,20]
    # steps = [2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,256]
    diffs = []
    
    for num_steps in steps:
        # Run FR2 stepping
        _, V_fr2 = policy_fr2_stepping(mdp, tau=tau, h=0.1, grad_time_T=num_steps*h)
        V_diff = np.max(np.abs(V_fr2 - optimal_value_soft_pol_iter))
        diffs.append(V_diff)
    
    
    plt.plot(steps, np.log(diffs), 'o')
    plt.xlabel('setps')
    plt.ylabel('log|V^*-V^N|')
    plt.title('FR2 errors')
    plt.savefig('fr2_error_plot.pdf')
    plt.close()




def figure_fr2_convergence2():
    # Setup the mdp
    mdp = GridworldMDP(grid_size=11, gamma=0.99, randomize=0.0)
    tau = 0     
    
    # Run policy iteration with softmax
    # _, optimal_value = log_policy_iteration_softmax(mdp, tau=tau)
    _, optimal_value = policy_iteration(mdp)

    h = 0.1
    # steps = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    steps = [2, 4, 6, 8]
    
    diffs_fr2 = []
    diffs_md = []
    
    for num_steps in steps:
        # Run FR2 and MD stepping
        _, V_fr2 = policy_fr2_stepping(mdp, tau=tau, h=h, grad_time_T=num_steps*h)
        _, V_md = policy_mirror_stepping(mdp, tau=tau, h=h, grad_time_T=num_steps*h)

        V_diff_fr2 = np.max(np.abs(V_fr2 - optimal_value))
        V_diff_md = np.max(np.abs(V_md - optimal_value))
        
        diffs_fr2.append(V_diff_fr2)
        diffs_md.append(V_diff_md)
        print(f"Steps {num_steps} (h={h}, T={num_steps*h}) | Max diff FR2: {V_diff_fr2:.6f} | Max diff MD: {V_diff_md:.6f}")
        
    steps_np = np.array(steps, dtype=float)
    diffs_fr2_np = np.array(diffs_fr2, dtype=float)
    diffs_md_np = np.array(diffs_md, dtype=float)
    
    # Helper function to do the log-space fitting and R^2 calculation
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
        # Calculate R^2 in LOG SPACE
        # ---------------------------------------------------------
        ss_tot_log = np.sum((log_diffs - np.mean(log_diffs))**2)
        
        ss_res_inv_log = np.sum((log_diffs - log_fit_inv)**2)
        r2_inv = 1 - (ss_res_inv_log / ss_tot_log)
        
        ss_res_exp_log = np.sum((log_diffs - log_fit_exp)**2)
        r2_exp = 1 - (ss_res_exp_log / ss_tot_log)
        
        best_fit_name = "$O(\kappa^N)$" if r2_exp > r2_inv else "$O(1/N)$"
        
        return {
            'safe_diffs': safe_diffs,
            'C_inv': C_inv, 'fit_inv': fit_inv, 'r2_inv': r2_inv,
            'kappa_exp': kappa_exp, 'C_exp': C_exp, 'fit_exp': fit_exp, 'r2_exp': r2_exp,
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
             label=f"FR2 $O(1/N)$: $C={fit_fr2['C_inv']:.4f}$ ($R^2={fit_fr2['r2_inv']:.4f}$)")
    ax1.plot(steps_np, fit_fr2['fit_exp'], '-.', color='tab:blue', 
             label=f"FR2 $O(\kappa^N)$: $\kappa={fit_fr2['kappa_exp']:.4f}$ ($R^2={fit_fr2['r2_exp']:.4f}$)")
    
    # MD
    ax1.plot(steps_np, diffs_md_np, 's', color='tab:orange', markersize=6, label='MD Empirical')
    ax1.plot(steps_np, fit_md['fit_inv'], '--', color='tab:orange', alpha=0.6, 
             label=f"MD $O(1/N)$: $C={fit_md['C_inv']:.4f}$ ($R^2={fit_md['r2_inv']:.4f}$)")
    ax1.plot(steps_np, fit_md['fit_exp'], '-.', color='tab:orange', 
             label=f"MD $O(\kappa^N)$: $\kappa={fit_md['kappa_exp']:.4f}$ ($R^2={fit_md['r2_exp']:.4f}$)")
             
    ax1.set_xlabel('Steps (N)')
    ax1.set_ylabel('$|V^* - V^N|$')
    ax1.set_title(f"Convergence Rate (Linear Scale)\nBest FR2: {fit_fr2['best_fit_name']} | Best MD: {fit_md['best_fit_name']}")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Log Y-axis ---
    # FR2
    ax2.plot(steps_np, np.log(fit_fr2['safe_diffs']), 'o', color='tab:blue', markersize=6, label='FR2 Empirical')
    ax2.plot(steps_np, np.log(fit_fr2['fit_inv']), '--', color='tab:blue', alpha=0.6, 
             label=f"FR2 $O(1/N)$: $C={fit_fr2['C_inv']:.4f}$ ($R^2={fit_fr2['r2_inv']:.4f}$)")
    ax2.plot(steps_np, np.log(fit_fr2['fit_exp']), '-.', color='tab:blue', 
             label=f"FR2 $O(\kappa^N)$: $\kappa={fit_fr2['kappa_exp']:.4f}$ ($R^2={fit_fr2['r2_exp']:.4f}$)")
    
    # MD
    ax2.plot(steps_np, np.log(fit_md['safe_diffs']), 's', color='tab:orange', markersize=6, label='MD Empirical')
    ax2.plot(steps_np, np.log(fit_md['fit_inv']), '--', color='tab:orange', alpha=0.6, 
             label=f"MD $O(1/N)$: $C={fit_md['C_inv']:.4f}$ ($R^2={fit_md['r2_inv']:.4f}$)")
    ax2.plot(steps_np, np.log(fit_md['fit_exp']), '-.', color='tab:orange', 
             label=f"MD $O(\kappa^N)$: $\kappa={fit_md['kappa_exp']:.4f}$ ($R^2={fit_md['r2_exp']:.4f}$)")
             
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
    # figure1a()
    # figure1b()
    # figure2a()
    # figure2b()
    # figure2a_only_mirror_and_midpt()
    # figure_fr2_convergence()
    figure_fr2_convergence2()
