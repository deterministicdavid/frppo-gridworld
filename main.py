
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


from GridworldMDP import GridworldMDP
from softmax_PIA import log_policy_iteration_softmax, log_value_iteration_softmax
from PIA import policy_iteration
from mirror_descent import policy_mirror_stepping
from mirror_descent import policy_mirror_stepping_semi_implicit
from mirror_descent import policy_mirror_midpoint_stepping
from mirror_descent import policy_mirror_midpoint_stepping_semi_implicit
from mirror_descent import policy_mirror_RK4_stepping
from mirror_descent import policy_mirror_stepping_exponential_integrator

from fr_descent import policy_fr2_stepping


def error_plot1(x_vals, y1, reverse_x_axis=False,x_label='x label',y_label='y label',title='title',name_prefix='plot',show_plot=False,legend1='-'):
    """
    Plot the errors for two different methods (e.g. mirror and mid-point)
    against a range of grid sizes.

    Parameters:
        x_vals (list): List of grid size values.
        y1 (list): List of error values for method 1 (e.g. mirror).
        y2 (list): List of error values for method 2 (e.g. mid-point).
    """
    plt.figure(figsize=(8,6))

    # Plot the errors for the two methods
    plt.plot(x_vals, np.log(y1), label=legend1, marker='o')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if reverse_x_axis:
        # Reverse the x-axis
        ax = plt.gca()
        ax.invert_xaxis()

    # Add legend
    plt.legend()

    # Save the plot as a PDF
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name_prefix}_{date_time}.pdf"
    print(f"Saving plot to {filename}")
    plt.savefig(filename)

    # Show the plot
    if show_plot: 
        plt.show()


def error_plot3(x_vals, y1, y2, y3, reverse_x_axis=False,x_label='x label',y_label='y label',title='title',name_prefix='plot',show_plot=False,legend1='-',legend2='-',legend3='-'):
    """
    Plot the errors for two different methods (e.g. mirror and mid-point)
    against a range of grid sizes.

    Parameters:
        x_vals (list): List of grid size values.
        y1 (list): List of error values for method 1 (e.g. mirror).
        y2 (list): List of error values for method 2 (e.g. mid-point).
    """
    plt.figure(figsize=(8,6))

    # Plot the errors for the two methods
    plt.plot(x_vals, np.log(y1), label=legend1, marker='o')
    plt.plot(x_vals, np.log(y2), label=legend2, marker='s')
    plt.plot(x_vals, np.log(y3), label=legend3, marker='s')

    # Add title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if reverse_x_axis:
        # Reverse the x-axis
        ax = plt.gca()
        ax.invert_xaxis()

    # Add legend
    plt.legend()

    # Save the plot as a PDF
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name_prefix}_{date_time}.pdf"
    print(f"Saving plot to {filename}")
    plt.savefig(filename)

    # Show the plot
    if show_plot: 
        plt.show()


def error_plot2(x_vals, y1, y2, reverse_x_axis=False,x_label='x label',y_label='y label',title='title',name_prefix='plot',show_plot=False,legend1='-',legend2='-'):
    """
    Plot the errors for two different methods (e.g. mirror and mid-point)
    against a range of grid sizes.

    Parameters:
        x_vals (list): List of grid size values.
        y1 (list): List of error values for method 1 (e.g. mirror).
        y2 (list): List of error values for method 2 (e.g. mid-point).
    """
    plt.figure(figsize=(8,6))

    # Plot the errors for the two methods
    plt.plot(x_vals, np.log(y1), label=legend1, marker='o')
    plt.plot(x_vals, np.log(y2), label=legend2, marker='s')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if reverse_x_axis:
        # Reverse the x-axis
        ax = plt.gca()
        ax.invert_xaxis()

    # Add legend
    plt.legend()

    # Save the plot as a PDF
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name_prefix}_{date_time}.pdf"
    print(f"Saving plot to {filename}")
    plt.savefig(filename)

    # Show the plot
    if show_plot: 
        plt.show()


def error_plot3(x_vals, y1, y2, y3, reverse_x_axis=False,x_label='x label',y_label='y label',title='title',name_prefix='plot',show_plot=False,legend1='-',legend2='-',legend3='-'):
    """
    Plot the errors for two different methods (e.g. mirror and mid-point)
    against a range of grid sizes.

    Parameters:
        x_vals (list): List of grid size values.
        y1 (list): List of error values for method 1 (e.g. mirror).
        y2 (list): List of error values for method 2 (e.g. mid-point).
    """
    plt.figure(figsize=(8,6))

    # Plot the errors for the two methods
    plt.plot(x_vals, np.log(y1), label=legend1, marker='o')
    plt.plot(x_vals, np.log(y2), label=legend2, marker='s')
    plt.plot(x_vals, np.log(y3), label=legend3, marker='s')

    # Add title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if reverse_x_axis:
        # Reverse the x-axis
        ax = plt.gca()
        ax.invert_xaxis()

    # Add legend
    plt.legend()

    # Save the plot as a PDF
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name_prefix}_{date_time}.pdf"
    print(f"Saving plot to {filename}")
    plt.savefig(filename)

    # Show the plot
    if show_plot: 
        plt.show()


def error_plot4(x_vals, y1, y2, y3, y4, reverse_x_axis=False,x_label='x label',y_label='y label',title='title',name_prefix='plot',show_plot=False,legend1='-',legend2='-',legend3='-', legend4='-'):
    """
    Plot the errors for two different methods (e.g. mirror and mid-point)
    against a range of grid sizes.

    Parameters:
        x_vals (list): List of grid size values.
        y1 (list): List of error values for method 1 (e.g. mirror).
        y2 (list): List of error values for method 2 (e.g. mid-point).
    """
    plt.figure(figsize=(8,6))

    # Plot the errors for the two methods
    plt.plot(x_vals, np.log(y1), label=legend1, marker='o')
    plt.plot(x_vals, np.log(y2), label=legend2, marker='s')
    plt.plot(x_vals, np.log(y3), label=legend3, marker='s')
    plt.plot(x_vals, np.log(y4), label=legend4, marker='s')

    # Add title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if reverse_x_axis:
        # Reverse the x-axis
        ax = plt.gca()
        ax.invert_xaxis()

    # Add legend
    plt.legend()

    # Save the plot as a PDF
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name_prefix}_{date_time}.pdf"
    print(f"Saving plot to {filename}")
    plt.savefig(filename)

    # Show the plot
    if show_plot: 
        plt.show()

def error_plot5(x_vals, y1, y2, y3, y4, y5, reverse_x_axis=False,x_label='x label',y_label='y label',title='title',name_prefix='plot',show_plot=False,legend1='-',legend2='-',legend3='-', legend4='-', legend5='-'):
    """
    Plot the errors for two different methods (e.g. mirror and mid-point)
    against a range of grid sizes.

    Parameters:
        x_vals (list): List of grid size values.
        y1 (list): List of error values for method 1 (e.g. mirror).
        y2 (list): List of error values for method 2 (e.g. mid-point).
    """
    plt.figure(figsize=(8,6))

    # Plot the errors for the two methods
    plt.plot(x_vals, np.log(y1), label=legend1, marker='o')
    plt.plot(x_vals, np.log(y2), label=legend2, marker='s')
    plt.plot(x_vals, np.log(y3), label=legend3, marker='s')
    plt.plot(x_vals, np.log(y4), label=legend4, marker='s')
    plt.plot(x_vals, np.log(y5), label=legend5, marker='s')

    # Add title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if reverse_x_axis:
        # Reverse the x-axis
        ax = plt.gca()
        ax.invert_xaxis()

    # Add legend
    plt.legend()

    # Save the plot as a PDF
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name_prefix}_{date_time}.pdf"
    print(f"Saving plot to {filename}")
    plt.savefig(filename)

    # Show the plot
    if show_plot: 
        plt.show()




# Runs mirror descent, midpoint descent and RK4 and plots the results
# all use the same stepsize and we vary the time horizon
# The midpoint method uses 2x and RK4 4x the number of policy evaluations so this is an "unfair" setup.
def figure1a():
    # Setup the mdp
    gamma=0.8
    mdp = GridworldMDP(grid_size=11, gamma=gamma)

    # Set the temperature parameter for softmax
    tau = 1e-1

    # Run policy iteration with softmax to get the reference policy and value
    _, V_ref = log_policy_iteration_softmax(mdp, tau=tau)
    # V_ref = log_value_iteration_softmax(mdp, tau=tau)

    # Choose the time steps to consider
    # h = 2.5
    # T_max_vals = [5,10,15,20]
    h = 1.0
    T_max_vals = [1,2,3,4,5,6,7,8,10,12,16,20,24]
    
    # Set up some strings we'll use later for producing the plot
    # plot_title = f'Error plot with gamma={gamma}, tau={tau}, h={h}'
    plot_title = ''

    V_fn_errors_mirror_linf = []
    # V_fn_errors_semi_imp = []
    # V_fn_errors_mirror_exp_int = []
    V_fn_errors_mirror_midpt_linf = []
    V_fn_errors_mirror_RK4 = []
    
    for i in range(0, len(T_max_vals)):
        T_max=T_max_vals[i]

        # Run mirror stepping
        _, V_mirror = policy_mirror_stepping(mdp, tau=tau, h=h, grad_time_T=T_max)
        # _, V_mirror_semi_imp = policy_mirror_stepping_semi_implicit(mdp, tau=tau, h=h, grad_time_T=T_max)
        # _, V_mirror_exp_int = policy_mirror_stepping_exponential_integrator(mdp, tau=tau, h=h, grad_time_T=T_max)
        _, V_mirror_midpt = policy_mirror_midpoint_stepping(mdp, tau=tau, h=h, grad_time_T=T_max)
        _, V_mirror_RK4 = policy_mirror_RK4_stepping(mdp, tau=tau, h=h, grad_time_T=T_max)
        

        V_fn_errors_mirror_linf.append(np.max(np.abs(V_ref - V_mirror)))
        print(f'l infty error for mirror stepping: {V_fn_errors_mirror_linf[-1]}')

        # V_fn_errors_semi_imp.append(np.max(np.abs(V_ref - V_mirror_semi_imp)))
        # print(f'l infty error for mirror stepping: {V_fn_errors_semi_imp[-1]}')

        # V_fn_errors_mirror_exp_int.append(np.max(np.abs(V_ref  - V_mirror_exp_int)))
        # print(f'l infty for mirror exponential integrator: {V_fn_errors_mirror_exp_int[-1]}')

        V_fn_errors_mirror_midpt_linf.append(np.max(np.abs(V_ref - V_mirror_midpt)))
        print(f'l infty for mirror midpoint stepping: {V_fn_errors_mirror_midpt_linf[-1]}')

        V_fn_errors_mirror_RK4.append(np.max(np.abs(V_ref - V_mirror_RK4)))
        print(f'l infty error for mirror RK4: {V_fn_errors_mirror_RK4[-1]}')


    error_plot3(T_max_vals, 
               V_fn_errors_mirror_linf, 
               V_fn_errors_mirror_midpt_linf,
               V_fn_errors_mirror_RK4,
               reverse_x_axis=False,
               x_label='Number of iterations', 
               y_label='log error',
               title = plot_title,
               name_prefix='figure_1a',
               legend1='1st order',
               legend2='2nd order',
               legend3='4th order',
               )

    
# Runs mirror descent, midpoint and RK4 descent and plots the results
# midpoint uses stepsize 2x as long as mirror method and RK4 uses stepsize 4x as long
# Thus all same the number of policy evaluations so this is a "fair" setup.

def figure2a():
    # Setup the mdp
    gamma=0.80
    mdp = GridworldMDP(grid_size=11, gamma=gamma)

    # Set the temperature parameter for softmax
    tau = 1e-1

    # Run policy iteration with softmax to get the reference policy and value
    _, V_ref = log_policy_iteration_softmax(mdp, tau=tau)
    

    h_mirror = 0.25
    h_midpoint = 0.5
    h_RK4 = 1.0
    
    T_max_vals = [2,4,6,8,10,12]
    
    # Set up some strings we'll use later for producing the plot
    # plot_title = f'Error plot with gamma={gamma}, tau={tau}, h={h_mirror} for mirror and h={h_midpoint} for midpoint'
    plot_title = ''
    
    V_fn_errors_mirror_linf = []
    # V_fn_errors_semi_imp_linf = []
    # V_fn_errors_mirror_exp_int = []
    V_fn_errors_midpt_linf = []
    V_fn_errors_RK4_linf = []
    
    for i in range(0, len(T_max_vals)):
        T_max=T_max_vals[i]

        # Run mirror stepping
        _, V_mirror = policy_mirror_stepping(mdp, tau=tau, h=h_mirror, grad_time_T=T_max)
        # _, V_semi_imp = policy_mirror_stepping_semi_implicit(mdp, tau=tau, h=h_midpoint, grad_time_T=T_max)
        # _, V_mirror_exp_int = policy_mirror_stepping_exponential_integrator(mdp, tau=tau, h=h_mirror, grad_time_T=T_max)
        _, V_mirror_midpt = policy_mirror_midpoint_stepping(mdp, tau=tau, h=h_midpoint, grad_time_T=T_max)
        _, V_RK4 = policy_mirror_RK4_stepping(mdp, tau=tau, h=h_RK4, grad_time_T=T_max)

        V_fn_errors_mirror_linf.append(np.max(np.abs(V_ref - V_mirror)))
        print(f'l infty error for mirror stepping: {V_fn_errors_mirror_linf[-1]}')
        
        # V_fn_errors_semi_imp_linf.append(np.max(np.abs(V_ref - V_semi_imp)))
        # print(f'l infty error semi-implicit stepping: {V_fn_errors_semi_imp_linf[-1]}')

        # V_fn_errors_mirror_exp_int.append(np.max(np.abs(V_ref - V_mirror_exp_int)))
        # print(f'l infty error exp-int stepping: {V_fn_errors_mirror_exp_int[-1]}')

        V_fn_errors_midpt_linf.append(np.max(np.abs(V_ref - V_mirror_midpt)))
        print(f'l infty for mirror midpoint stepping: {V_fn_errors_midpt_linf[-1]}')

        V_fn_errors_RK4_linf.append(np.max(np.abs(V_ref - V_RK4)))
        print(f'l infty for mirror midpoint semi stepping: {V_fn_errors_RK4_linf[-1]}')

    error_plot3(T_max_vals, 
               V_fn_errors_mirror_linf, 
               V_fn_errors_midpt_linf,
               V_fn_errors_RK4_linf,
               reverse_x_axis=False,
               x_label='Number of iterations',
               y_label='log error',
               title = plot_title,
               name_prefix='figure_2a',
               legend1='1st order',
               legend2='2nd order',
               legend3='4th order')


# runs the two mirror descent methods with a small tau (smaller than what PIA copes with)
def experiment3():
    # Setup the mdp
    gamma=0.80
    mdp = GridworldMDP(grid_size=11, gamma=gamma)

    # Set the temperature parameter for softmax
    tau = 1e-4

    h_mirror = 0.1
    h_midpoint = 0.2
    T_max_vals = [2,3,4,5,6,7,8,9,10]
    
    # Set up some strings we'll use later for producing the plot
    plot_title = f'Difference between mirror and midpoint with gamma={gamma}, tau={tau}, h={h_mirror} for mirror and h={h_midpoint} for midpoint'

    V_fn_diff_l2 = []
    V_fn_diff_linf = []
    
    for i in range(0, len(T_max_vals)):
        T_max=T_max_vals[i]

        # Run mirror stepping
        _, V_mirror = policy_mirror_stepping(mdp, tau=tau, h=h_mirror, grad_time_T=T_max)
        _, V_mirror_midpt = policy_mirror_midpoint_stepping(mdp, tau=tau, h=h_midpoint, grad_time_T=T_max)

        err_mirror_l2 =  np.linalg.norm(V_mirror_midpt - V_mirror)/mdp.num_states      # np.max(np.abs(V_ref - V_mirror))
        print(f'l2 diff between mirror and midpoint: {err_mirror_l2}')

        err_mirror_linf =  np.max(np.abs(V_mirror_midpt - V_mirror))
        print(f'l infty diff between mirror and midpoint: {err_mirror_linf}')

        err_midpt_linf = np.max(np.abs(V_mirror - V_mirror_midpt))
        print(f'l infty for mirror midpoint stepping: {err_midpt_linf}')

        V_fn_diff_l2.append(err_mirror_l2)

        V_fn_diff_linf.append(err_mirror_linf)



    error_plot3(T_max_vals, 
               V_fn_diff_l2, 
               V_fn_diff_linf,
               reverse_x_axis=False,
               x_label='Mirror flow time horizon, time step fixed', 
               y_label='difference',
               title = plot_title,
               name_prefix='experiment3_plot')

    
    

# Runs mirror descent and midpoint descent with annealing and plots the results against non-entropy-regularized value
def experiment4_annealing():
    # Setup the mdp
    gamma=0.90
    mdp = GridworldMDP(grid_size=11, gamma=gamma)

    # Set the initial tau for annealing
    init_tau = 0.5

    # Run policy iteration with softmax to get the reference policy and value
    _, V_ref = policy_iteration(mdp)
    

    h_mirror = 0.1
    h_midpoint = 0.2
    T_max_vals = [2,3,4,5,6,7,8,9,10]
    
    # Set up some strings we'll use later for producing the plot
    plot_title = f'Error plot with gamma={gamma}, annealed tau, h={h_mirror} for mirror and h={h_midpoint} for midpoint'

    V_fn_errors_mirror_l2 = []
    V_fn_errors_mirror_midpt_l2 = []
    V_fn_errors_mirror_linf = []
    V_fn_errors_mirror_midpt_linf = []
    
    for i in range(0, len(T_max_vals)):
        T_max=T_max_vals[i]

        # Run mirror stepping
        _, V_mirror = policy_mirror_stepping(mdp, tau=init_tau, h=h_mirror, grad_time_T=T_max, annealing=True, tau_min=0.0)
        _, V_mirror_midpt = policy_mirror_midpoint_stepping(mdp, tau=init_tau, h=h_midpoint, grad_time_T=T_max, annealing=True, tau_min=0.0)

        err_mirror_l2 =  np.linalg.norm(V_ref - V_mirror)/mdp.num_states      # np.max(np.abs(V_ref - V_mirror))
        print(f'l2 error for mirror stepping: {err_mirror_l2}')

        err_midpt_l2 = np.linalg.norm(V_ref - V_mirror_midpt)/mdp.num_states  # np.max(np.abs(V_ref - V_mirror_midpt))
        print(f'l2 error for mirror midpoint stepping: {err_midpt_l2}')

        err_mirror_linf =  np.max(np.abs(V_ref - V_mirror))
        print(f'l infty error for mirror stepping: {err_mirror_linf}')

        err_midpt_linf = np.max(np.abs(V_ref - V_mirror_midpt))
        print(f'l infty for mirror midpoint stepping: {err_midpt_linf}')

        V_fn_errors_mirror_l2.append(err_mirror_l2)
        V_fn_errors_mirror_midpt_l2.append(err_midpt_l2)

        V_fn_errors_mirror_linf.append(err_mirror_linf)
        V_fn_errors_mirror_midpt_linf.append(err_midpt_linf)



    error_plot3(T_max_vals, 
               V_fn_errors_mirror_l2, 
               V_fn_errors_mirror_midpt_l2,
               reverse_x_axis=False,
               x_label='Mirror flow time horizon, time step fixed', 
               y_label='log error l2 norm',
               title = plot_title,
               name_prefix='experiment4_errors_plot_l2')

    error_plot3(T_max_vals, 
               V_fn_errors_mirror_linf, 
               V_fn_errors_mirror_midpt_linf,
               reverse_x_axis=False,
               x_label='Mirror flow time horizon, time step fixed', 
               y_label='log error l infinity norm',
               title = plot_title,
               name_prefix='experiment4_errors_plot_linf')



def experiment5():
    # Setup the mdp
    gamma=0.80
    mdp = GridworldMDP(grid_size=11, gamma=gamma)

    # Set the temperature parameter for softmax
    tau = 0.1

    # Run policy iteration with softmax to get the reference policy and value
    _, V_ref = log_policy_iteration_softmax(mdp, tau=tau)
    
    h_mirror = 0.05
    h_midpoint = 0.1
    
    T_max_vals = [1,2,4,6,8,10,12,14,16,18,20]
    
    # Set up some strings we'll use later for producing the plot
    plot_title = f'Error plot with gamma={gamma}, tau={tau}, h={h_mirror} for mirror and h={h_midpoint} for midpoint'

    
    V_fn_errors_mirror_linf = []
    V_fn_errors_mirror_semi_imp_linf = []
    V_fn_errors_midpt_linf = []
    V_fn_errors_midpt_semi_imp_linf = []
    
    for i in range(0, len(T_max_vals)):
        T_max=T_max_vals[i]

        # Run mirror stepping
        _, V_mirror = policy_mirror_stepping(mdp, tau=tau, h=h_mirror, grad_time_T=T_max)
        _, V_mirror_semi_imp = policy_mirror_stepping_semi_implicit(mdp, tau=tau, h=h_mirror, grad_time_T=T_max)
        _, V_midpt = policy_mirror_midpoint_stepping(mdp, tau=tau, h=h_midpoint, grad_time_T=T_max)
        _, V_midpt_semi_imp = policy_mirror_midpoint_stepping_semi_implicit(mdp, tau=tau, h=h_midpoint, grad_time_T=T_max)

        V_fn_errors_mirror_linf.append(np.max(np.abs(V_ref - V_mirror)))
        print(f'l infty error for mirror stepping: {V_fn_errors_mirror_linf[-1]}')

        V_fn_errors_mirror_semi_imp_linf.append(np.max(np.abs(V_ref - V_mirror_semi_imp)))
        print(f'l infty error for semi-implicit mirror stepping: {V_fn_errors_mirror_semi_imp_linf[-1]}')
        
        V_fn_errors_midpt_linf.append(np.max(np.abs(V_ref - V_midpt)))
        print(f'l infty for mirror midpoint stepping: {V_fn_errors_midpt_linf[-1]}')
        
        V_fn_errors_midpt_semi_imp_linf.append(np.max(np.abs(V_ref - V_midpt_semi_imp)))
        print(f'l infty for mirror semi_imp stepping: {V_fn_errors_midpt_semi_imp_linf[-1]}')
        




    error_plot4(T_max_vals, 
               V_fn_errors_mirror_linf, 
               V_fn_errors_mirror_semi_imp_linf,
               V_fn_errors_midpt_linf,
               V_fn_errors_midpt_semi_imp_linf,
               reverse_x_axis=False,
               x_label='Mirror flow time horizon, time step fixed', 
               y_label='log error l infinity norm',
               title = plot_title,
               name_prefix='experiment5_errors_plot_linf',
               legend1='mirror',
               legend2='mirror semi-implicit',
               legend3='midpoint',
               legend4='midpoint semi-imp')

# fix time horizon, change number of steps
# keeping number of policy evaluations constants to make
# a fair comparison

def figure2b():
    # Setup the mdp
    gamma=0.80
    mdp = GridworldMDP(grid_size=11, gamma=gamma)

    # Set the temperature parameter for softmax
    tau = 0.1

    # Run policy iteration with softmax to get the reference policy and value
    _, V_ref = log_policy_iteration_softmax(mdp, tau=tau)
    
    T_max = 25
    h_scaling = 1.0 # so we don't start with too big steps 
    min_steps = 1
    max_steps = 6 # this is for midpoint, mirror will get 2x
    
    # Set up some strings we'll use later for producing the plot
    #plot_title = f'Error plot with gamma={gamma}, tau={tau}, T={T_max}'
    plot_title = ''

    V_fn_errors_mirror_linf = []
    # V_fn_errors_semi_imp_linf = []
    # V_fn_errors_exp_int_linf = []
    V_fn_errors_midpt_linf = []
    V_fn_errors_RK4_linf = []
    numsteps = []

    for i in range(min_steps, max_steps):
        h_mirror = 0.25*h_scaling*T_max/i
        h_midpoint = 0.5*h_scaling*T_max/i
        h_RK4 = h_scaling*T_max/i
        numsteps.append(T_max/h_mirror)

        # Run mirror stepping
        _, V_RK4 = policy_mirror_RK4_stepping(mdp, tau=tau, h=h_RK4, grad_time_T=T_max)
        _, V_mirror = policy_mirror_stepping(mdp, tau=tau, h=h_mirror, grad_time_T=T_max)
        # _, V_mirror_semi_imp= policy_mirror_stepping_semi_implicit(mdp, tau=tau, h=h_mirror, grad_time_T=T_max)
        # _, V_mirror_exp_int = policy_mirror_stepping_exponential_integrator(mdp, tau=tau, h=h_mirror, grad_time_T=T_max)
        _, V_midpt = policy_mirror_midpoint_stepping(mdp, tau=tau, h=h_midpoint, grad_time_T=T_max)
        
        
        
        V_fn_errors_mirror_linf.append(np.max(np.abs(V_ref - V_mirror)))
        print(f'l infty error for mirror stepping: {V_fn_errors_mirror_linf[-1]}')

        # V_fn_errors_semi_imp_linf.append(np.max(np.abs(V_ref - V_mirror_semi_imp)))
        # print(f'l infty error for semi-implicit stepping: {V_fn_errors_semi_imp_linf[-1]}')

        # V_fn_errors_exp_int_linf.append(np.max(np.abs(V_ref - V_mirror_exp_int)))
        # print(f'l infty error for exponential integrator stepping: {V_fn_errors_exp_int_linf[-1]}')

        V_fn_errors_midpt_linf.append(np.max(np.abs(V_ref - V_midpt)))
        print(f'l infty for mirror midpoint stepping: {V_fn_errors_midpt_linf[-1]}')
        
        V_fn_errors_RK4_linf.append(np.max(np.abs(V_ref - V_RK4)))
        print(f'l infty for mirror RK4 stepping: {V_fn_errors_RK4_linf[-1]}')
        


    # numsteps = np.array((range(1+min_steps,1+max_steps)))/h_scaling
    error_plot3(numsteps, 
               V_fn_errors_mirror_linf, 
               V_fn_errors_midpt_linf,
               V_fn_errors_RK4_linf,
               reverse_x_axis=False,
               x_label='Number of iterations', 
               y_label='log error',
               title = plot_title,
               name_prefix='figure_2b',
               legend1='1st order',
               legend2='2nd order',
               legend3='4th order')

def figure1b():
    # Setup the mdp
    gamma=0.80
    mdp = GridworldMDP(grid_size=11, gamma=gamma)

    # Set the temperature parameter for softmax
    tau = 0.1

    # Run policy iteration with softmax to get the reference policy and value
    _, V_ref = log_policy_iteration_softmax(mdp, tau=tau)
    
    T_max = 25
    h_scaling = 0.5 # so we don't start with too big steps 
    min_steps = 1
    max_steps = 6 # this is for midpoint, mirror will get 2x
    
    # Set up some strings we'll use later for producing the plot
    # plot_title = f'Error plot with gamma={gamma}, tau={tau}, T={T_max}'
    plot_title = ''

    V_fn_errors_mirror_linf = []
    #V_fn_errors_semi_imp_linf = []
    #V_fn_errors_exp_int_linf = []
    V_fn_errors_midpt_linf = []
    V_fn_errors_RK4_linf = []
    numsteps = []
    
    for i in range(min_steps, max_steps):
        h = h_scaling*T_max/i
        numsteps.append(T_max/h)

        # Run mirror stepping
        _, V_mirror = policy_mirror_stepping(mdp, tau=tau, h=h, grad_time_T=T_max)
        #_, V_mirror_semi_imp= policy_mirror_stepping_semi_implicit(mdp, tau=tau, h=h, grad_time_T=T_max)
        #_, V_mirror_exp_int = policy_mirror_stepping_exponential_integrator(mdp, tau=tau, h=h, grad_time_T=T_max)
        _, V_midpt = policy_mirror_midpoint_stepping(mdp, tau=tau, h=h, grad_time_T=T_max)
        _, V_RK4 = policy_mirror_RK4_stepping(mdp, tau=tau, h=h, grad_time_T=T_max)
        
        
        V_fn_errors_mirror_linf.append(np.max(np.abs(V_ref - V_mirror)))
        print(f'l infty error for mirror stepping: {V_fn_errors_mirror_linf[-1]}')

        # V_fn_errors_semi_imp_linf.append(np.max(np.abs(V_ref - V_mirror_semi_imp)))
        # print(f'l infty error for semi-implicit stepping: {V_fn_errors_semi_imp_linf[-1]}')

        # V_fn_errors_exp_int_linf.append(np.max(np.abs(V_ref - V_mirror_exp_int)))
        # print(f'l infty error for exponential integrator stepping: {V_fn_errors_exp_int_linf[-1]}')

        V_fn_errors_midpt_linf.append(np.max(np.abs(V_ref - V_midpt)))
        print(f'l infty for mirror midpoint stepping: {V_fn_errors_midpt_linf[-1]}')
        
        V_fn_errors_RK4_linf.append(np.max(np.abs(V_ref - V_RK4)))
        print(f'l infty for mirror RK4 stepping: {V_fn_errors_RK4_linf[-1]}')
        


    
    error_plot3(numsteps, 
               V_fn_errors_mirror_linf, 
               V_fn_errors_midpt_linf,
               V_fn_errors_RK4_linf,
               reverse_x_axis=False,
               x_label='Number of iterations', 
               y_label='log error',
               title = plot_title,
               name_prefix='figure_1b',
               legend1='1st order',
               legend2='2nd order',
               legend3='4th order')


def experiment7():
    # Setup the mdp
    gamma=0.80
    mdp = GridworldMDP(grid_size=11, gamma=gamma)

    # Set the temperature parameter for softmax
    tau = 0.1

    # Run policy iteration with softmax to get the reference policy and value
    _, V_ref = log_policy_iteration_softmax(mdp, tau=tau)
    
    h_mirror = 0.05
    h_midpoint = 0.1
    
    T_max_vals = [1,2,4,6,8,10,12,14,16,18,20]
    
    # Set up some strings we'll use later for producing the plot
    plot_title = f'Error plot with gamma={gamma}, tau={tau}, h={h_mirror} for mirror and h={h_midpoint} for midpoint'

    
    V_fn_errors_mirror_linf = []
    V_fn_errors_mirror_semi_imp_linf = []
    V_fn_errors_midpt_linf = []
    V_fn_errors_mirror_exp_int = []
    
    for i in range(0, len(T_max_vals)):
        T_max=T_max_vals[i]

        # Run mirror stepping
        _, V_mirror = policy_mirror_stepping(mdp, tau=tau, h=h_mirror, grad_time_T=T_max)
        _, V_mirror_semi_imp = policy_mirror_stepping_semi_implicit(mdp, tau=tau, h=h_mirror, grad_time_T=T_max)
        _, V_midpt = policy_mirror_midpoint_stepping(mdp, tau=tau, h=h_midpoint, grad_time_T=T_max)
        _, V_mirror_exp_int = policy_mirror_stepping_exponential_integrator(mdp, tau=tau, h=h_mirror, grad_time_T=T_max)

        V_fn_errors_mirror_linf.append(np.max(np.abs(V_ref - V_mirror)))
        print(f'l infty error for mirror stepping: {V_fn_errors_mirror_linf[-1]}')

        V_fn_errors_mirror_semi_imp_linf.append(np.max(np.abs(V_ref - V_mirror_semi_imp)))
        print(f'l infty error for semi-implicit mirror stepping: {V_fn_errors_mirror_semi_imp_linf[-1]}')
        
        V_fn_errors_midpt_linf.append(np.max(np.abs(V_ref - V_midpt)))
        print(f'l infty for mirror midpoint stepping: {V_fn_errors_midpt_linf[-1]}')
        
        V_fn_errors_mirror_exp_int.append(np.max(np.abs(V_ref - V_mirror_exp_int)))
        print(f'l infty for mirror exponential integrator: {V_fn_errors_mirror_exp_int[-1]}')
        




    error_plot4(T_max_vals, 
               V_fn_errors_mirror_linf, 
               V_fn_errors_mirror_semi_imp_linf,
               V_fn_errors_midpt_linf,
               V_fn_errors_mirror_exp_int,
               reverse_x_axis=False,
               x_label='Mirror flow time horizon, time step fixed', 
               y_label='log error l infinity norm',
               title = plot_title,
               name_prefix='experiment7_',
               legend1='mirror',
               legend2='mirror semi-implicit',
               legend3='midpoint',
               legend4='mirror exp integrator')



def experiment8():
    # Setup the mdp
    gamma=0.90
    mdp = GridworldMDP(grid_size=11, gamma=gamma)

    # Set the temperature parameter for softmax
    tau = 0.05

    # Run policy iteration with softmax to get the reference policy and value
    _, V_ref = log_policy_iteration_softmax(mdp, tau=tau)
    
    min_steps = 1
    max_steps = 10 # this is for midpoint, mirror will get 2x
    
    # Set up some strings we'll use later for producing the plot
    plot_title = f'Error plot with gamma={gamma}, tau={tau}'

    V_fn_errors_mirror_linf = []
    V_fn_errors_semi_imp_linf = []
    V_fn_errors_exp_int_linf = []
    V_fn_errors_midpt_linf = []
    V_fn_errors_RK4_linf = []
    
    for i in range(min_steps, max_steps):
        T_max = i
        h = 1/i

        # Run mirror stepping
        _, V_mirror = policy_mirror_stepping(mdp, tau=tau, h=h, grad_time_T=T_max)
        _, V_mirror_semi_imp= policy_mirror_stepping_semi_implicit(mdp, tau=tau, h=h, grad_time_T=T_max)
        _, V_mirror_exp_int = policy_mirror_stepping_exponential_integrator(mdp, tau=tau, h=h, grad_time_T=T_max)
        _, V_midpt = policy_mirror_midpoint_stepping(mdp, tau=tau, h=h, grad_time_T=T_max)
        _, V_RK4 = policy_mirror_RK4_stepping(mdp, tau=tau, h=h, grad_time_T=T_max)
        
        
        V_fn_errors_mirror_linf.append(np.max(np.abs(V_ref - V_mirror)))
        print(f'l infty error for mirror stepping: {V_fn_errors_mirror_linf[-1]}')

        V_fn_errors_semi_imp_linf.append(np.max(np.abs(V_ref - V_mirror_semi_imp)))
        print(f'l infty error for semi-implicit stepping: {V_fn_errors_semi_imp_linf[-1]}')

        V_fn_errors_exp_int_linf.append(np.max(np.abs(V_ref - V_mirror_exp_int)))
        print(f'l infty error for exponential integrator stepping: {V_fn_errors_exp_int_linf[-1]}')

        V_fn_errors_midpt_linf.append(np.max(np.abs(V_ref - V_midpt)))
        print(f'l infty for mirror midpoint stepping: {V_fn_errors_midpt_linf[-1]}')
        
        V_fn_errors_RK4_linf.append(np.max(np.abs(V_ref - V_RK4)))
        print(f'l infty for mirror RK4 stepping: {V_fn_errors_RK4_linf[-1]}')
        


    numsteps = np.array((range(1+min_steps,1+max_steps)))
    error_plot5(numsteps, 
               V_fn_errors_mirror_linf, 
               V_fn_errors_semi_imp_linf,
               V_fn_errors_exp_int_linf,
               V_fn_errors_midpt_linf,
               V_fn_errors_RK4_linf,
               reverse_x_axis=False,
               x_label='Number of steps', 
               y_label='log error l infinity norm',
               title = plot_title,
               name_prefix='experiment8_errors_plot_linf',
               legend1='mirror',
               legend2='semi-implicit',
               legend3='exponential integrator',
               legend4='midpoint',
               legend5='RK4')


def plot_heatmap_of_errors_vs_gw_size_and_num_steps(numsteps, gw_grid_sizes, errors,name_prefix='errors_plot'):
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(errors, aspect='auto', cmap='viridis', 
            extent=[numsteps[0], numsteps[-1], gw_grid_sizes[-1], gw_grid_sizes[0]])

    # Add a colorbar
    plt.colorbar(label='Log Error')

    # Set axis labels
    plt.xlabel('Number of Steps')
    plt.ylabel('Grid Size')

    # Set axis ticks
    plt.xticks(numsteps)
    plt.yticks(gw_grid_sizes)

    # Set title
    plt.title('Heatmap of Errors with Respect to Grid Size and Number of Steps')

    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name_prefix}_{date_time}.pdf"
    print(f"Saving plot to {filename}")
    plt.savefig(filename)

def plot_3d_of_errors_vs_gw_size_and_num_steps(numsteps, gw_grid_sizes, errors,name_prefix='errors_plot2'):
    # Meshgrid for plotting
    numsteps, grid_size = np.meshgrid(numsteps, gw_grid_sizes)

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(numsteps, grid_size, errors, cmap='viridis')

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Error')

    # Set axis labels
    ax.set_xlabel('Number of Steps')
    ax.set_ylabel('Grid Size')
    ax.set_zlabel('Log Error')

    # Set title
    ax.set_title('3D Plot of Errors with Respect to Grid Size and Number of Steps')

    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name_prefix}_{date_time}.pdf"
    print(f"Saving plot to {filename}")
    plt.savefig(filename)


def experiment9():
    # Setup the mdp
    gamma=0.85
    
    # Set the temperature parameter for softmax
    tau = 0.05
    
    min_steps = 1
    max_steps = 20 
    
    min_gw_size = 10
    max_gw_size_scaling = 4
    max_gw_size_scaling_steps = 10
    gw_grid_sizes = []

    errors_wrt_grid_size = []

    for gw_size_step in range(0,max_gw_size_scaling_steps):
        grid_size = min_gw_size + gw_size_step * max_gw_size_scaling
        gw_grid_sizes.append(grid_size)
        mdp = GridworldMDP(grid_size=grid_size, gamma=gamma, randomize=5.0/(grid_size*grid_size))
        
        # Run policy iteration with softmax to get the reference policy and value
        _, V_ref = log_policy_iteration_softmax(mdp, tau=tau)

        V_fn_errors = []
        for i in range(min_steps, max_steps):
            h = 0.5/tau
            T_max = i*h

            # Run mirror stepping
            _, V_mirror = policy_mirror_stepping(mdp, tau=tau, h=h, grad_time_T=T_max)
            
            V_fn_errors.append(np.max(np.abs(V_ref - V_mirror)))
            print(f'size: {grid_size}, l infty error for mirror stepping: {V_fn_errors[-1]}')

        errors_wrt_grid_size.append(V_fn_errors)
            

    numsteps = np.array((range(1+min_steps,1+max_steps)))
    log_errors_wrt_grid_size_np = np.log(np.array(errors_wrt_grid_size))
    plot_heatmap_of_errors_vs_gw_size_and_num_steps(numsteps=numsteps, gw_grid_sizes=gw_grid_sizes, errors=log_errors_wrt_grid_size_np, name_prefix='experiment9_hm')
    plot_3d_of_errors_vs_gw_size_and_num_steps(numsteps=numsteps, gw_grid_sizes=gw_grid_sizes, errors=log_errors_wrt_grid_size_np, name_prefix='experiment9_3d')



def figure2a_only_mirror_and_midpt():
    # Setup the mdp
    gamma=0.80
    mdp = GridworldMDP(grid_size=11, gamma=gamma)

    # Set the temperature parameter for softmax
    tau = 1e-1

    # Run policy iteration with softmax to get the reference policy and value
    _, V_ref = log_policy_iteration_softmax(mdp, tau=tau)
    

    h_mirror = 4
    
    T_max_vals = [4,8,12,20,32,40,52,60,72,80]
    
    # Set up some strings we'll use later for producing the plot
    # plot_title = f'Error plot with gamma={gamma}, tau={tau}, h={h_mirror} for mirror and h={h_midpoint} for midpoint'
    plot_title = ''
    
    V_fn_errors_mirror_linf = []
    V_fn_errors_midpt_linf = []
    
    for i in range(0, len(T_max_vals)):
        T_max=T_max_vals[i]

        # Run mirror stepping
        _, V_mirror = policy_mirror_stepping(mdp, tau=tau, h=h_mirror, grad_time_T=T_max)
        V_fn_errors_mirror_linf.append(np.max(np.abs(V_ref - V_mirror)))
        print(f'l infty error for mirror stepping: {V_fn_errors_mirror_linf[-1]}')
        
        _, V_mirror_midpt = policy_mirror_midpoint_stepping(mdp, tau=tau, h=h_mirror*2, grad_time_T=T_max)
        V_fn_errors_midpt_linf.append(np.max(np.abs(V_ref - V_mirror_midpt)))
        print(f'l infty for mirror midpoint stepping: {V_fn_errors_midpt_linf[-1]}')

        
    error_plot2(np.array(T_max_vals)/h_mirror, 
               V_fn_errors_mirror_linf, 
               V_fn_errors_midpt_linf,
               reverse_x_axis=False,
               x_label='Number of iterations',
               y_label='log error',
               title = plot_title,
               name_prefix='figure_2a_mirror_and_mid',
               legend1='1st order',
               legend2='2nd order'
               )

def figure_fr2_convergence():
    
    
    # Setup the mdp
    mdp = GridworldMDP(grid_size=11, gamma=0.85, randomize=0.0)
    tau = 1e-2     
    
    # Run policy iteration with softmax
    _, optimal_value_soft_pol_iter = log_policy_iteration_softmax(mdp, tau=tau)

    
    # steps = [2,4,6,8,10,12,14,16,18,20]
    h=0.05
    steps = [2,4,8,16,32,64,128,256]
    diffs = []
    
    for num_steps in steps:
        # Run FR2 stepping
        _, V_fr2 = policy_fr2_stepping(mdp, tau=tau, h=0.1, grad_time_T=num_steps*h)
        V_diff = np.max(np.abs(V_fr2 - optimal_value_soft_pol_iter))
        diffs.append(V_diff)
    
    # error_plot1(np.array(steps), np.array(diffs), 
    #             reverse_x_axis=False, 
    #             x_label='steps',
    #             y_label='|V^*-V^N|',
    #             title='FR2 errors', 
    #             name_prefix='fr2_errors_plot',
    #             show_plot=False,
    #             legend1='-')

    plt.plot(steps, np.log(diffs))
    plt.xlabel('setps')
    plt.ylabel('log|V^*-V^N|')
    plt.title('FR2 errors')
    plt.savefig('fr2_error_plot.pdf')
    plt.close()


if __name__ == "__main__":
    # figure1a()
    # figure1b()
    # figure2a()
    # figure2b()
    # figure2a_only_mirror_and_midpt()
    figure_fr2_convergence()
