import numpy as np
from fr_descent import project_onto_simplex

if __name__ == "__main__":
    r = np.zeros(3)
    r[1] = 1; r[2] = 2
    A = r - np.sum(r)/3
    tau = 0.1
    N = 35

    pi_old = np.zeros(3)
    pi_old[0] = 0.5; pi_old[1] = 1/3; pi_old[2] = 1/6;  
    print("================ FR-PPO ============================")
    for i in range(0,N):
        pi_mid = pi_old + (tau/3) * A 
        pi_old = project_onto_simplex(pi_mid)
        print(f"FRPPO step {i+1}, policy {pi_old}, value {np.dot(r,pi_old)}")

    N = 2_000
    tau = (3/2)/N    
    pi_old = np.zeros(3)
    pi_old[0] = 0.5; pi_old[1] = 1/3; pi_old[2] = 1/6;  
    print("================Plain Euler with Advantage ============================")
    for i in range(0,N):
        pi_old = pi_old + (tau/3) * A 
        if (i+1) % 100 == 0:
            print(f"Euler step {i+1}, t={i*tau}, policy {pi_old}, value {np.dot(r,pi_old)}")
    
    N = 2_000
    tau = 0.1
    pi_old = np.zeros(3)
    pi_old[0] = 0.5; pi_old[1] = 1/3; pi_old[2] = 1/6;  
    M = 2 * np.max(r)
    print("================Adaptive stepping with Advantage ============================")
    T = 0
    for i in range(0,N):
        m = np.min(pi_old)
        tau_use = m/(2*M) - 1e-12
        T += tau_use
        pi_old = pi_old + (tau_use/3) * A 
        if (i+1) % 100 == 0:
            print(f"Adapt Euler {i+1}, tau={tau_use}, t={T}, policy {pi_old}, value {np.dot(r,pi_old)}")