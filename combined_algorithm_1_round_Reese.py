import cvxpy as cp
import numpy as np
import itertools
import tqdm
import matplotlib.pyplot as plt


def agent_update(prev_not_i_strategies, S_i, alpha_i, M, regularizer="euclidean", responder= False):

    utility_feedback_vector = [0.0 for s in S_i]


    if responder == True: # candidate
        for w_f in prev_not_i_strategies:
            for j, s_c in enumerate(S_i):
                for k, w_f_supp in enumerate(S_i):
                    if w_f_supp >= s_c:
                        utility_feedback_vector[j] += w_f[k] *  w_f_supp
    else: # firm
        for w_c in prev_not_i_strategies:
            for j, s_f in enumerate(S_i):
                for k, w_c_supp in enumerate(S_i):
                    if w_c_supp <= s_f:
                        utility_feedback_vector[j] += w_c[k] *  (1-s_f)
    

    w_var = cp.Variable(len(S_i))
    
    if regularizer=="negative_entropy":
        # Negative Entropy regularizer
        objective = cp.Maximize(w_var@utility_feedback_vector + 1/M * cp.sum(cp.entr(w_var)))
    else:
        # Euclidean regularizer (default)
        objective = cp.Maximize(w_var@utility_feedback_vector - cp.norm(w_var-alpha_i, 2)**2/2*M)

    constraints = [cp.sum(w_var)==1, cp.min(w_var)>=0]
    problem = cp.Problem(objective,constraints)

    problem.solve()

    return [max(0.0, round(w_p.value, 5)) for w_p in w_var] #objective maximizing strategy
    
    # keep largest non-zero support
    # largest = 0
    # for i, w_supp in enumerate(w_t_p_1_all):
    #     if w_supp >0:
    #         w_supp_prev = w_supp
    #         largest = i


    # w_t_p_1_all = [1 if i == largest else 0 for i in range(len(w_t_p_1_all))]    
    return w_t_p_1_all


def get_support(w_i, S_i):
    return [S_i[i] for i in range(len(w_i)) if w_i[i] > 0]

def run_simulation(T=100, M=None, strategy="pure", reference=None):
    if M is None:
        M = 1 / np.sqrt(T)

    S_f = [i / T for i in range(T + 1)]
    S_c = [i / T for i in range(T + 1)]

    beta_f_idx = np.random.randint(len(S_f))
    beta_c_idx = np.random.randint(len(S_c))
    alpha_f_idx = np.random.randint(len(S_f))
    alpha_c_idx = np.random.randint(len(S_c))
    if strategy == "biased pure":
        beta_f_idx = np.random.randint(len(S_f) // 2) # lower
        beta_c_idx = np.random.randint(len(S_c) // 2, len(S_c)) # upper
    beta_f = [1 if i == beta_f_idx else 0 for i in range(len(S_f))]
    beta_c = [1 if i == beta_c_idx else 0 for i in range(len(S_c))]
    alpha_f = [1 if i == alpha_f_idx else 0 for i in range(len(S_f))]
    alpha_c = [1 if i == alpha_c_idx else 0 for i in range(len(S_c))]
    if reference:
        if reference == "uniform":
            alpha_f = [1/len(S_f) for _ in range(len(S_f))]
            alpha_c = [1/len(S_c) for _ in range(len(S_c))]
        elif reference == "biased firm":
            # Biased towards lower values
            alpha_f = [np.exp(-i) for i in range(len(S_f))]
            alpha_f = [a/sum(alpha_f) for a in alpha_f]
    
    print_initial_conditions(beta_f, beta_c, alpha_f, alpha_c, S_f, S_c)
    
    prev_f = [beta_f]
    prev_c = [beta_c]

    iterations_until_convergence = []

    for t in tqdm.tqdm(range(T)):
        w_f_t_p_1 = agent_update(prev_c, S_i=S_f, alpha_i=alpha_f, M=M)
        w_c_t_p_1 = agent_update(prev_f, S_i=S_c, alpha_i=alpha_c, M=M, responder=True)

        # print(f"w_f_t_p_1 support: {get_support(w_f_t_p_1, S_f)}")
        # print(f"w_c_t_p_1 support: {get_support(w_c_t_p_1, S_c)}")

        # if np.argmax(w_f_t_p_1) >= np.argmax(w_c_t_p_1):
        #     print(f"Offer likely accepted at time {t}: {np.argmax(w_f_t_p_1)}.")

        convergence_threshhold = 1e-5
        # Check convergence (defined correctly?)
        if t > 0 and (np.linalg.norm(np.array(w_f_t_p_1) - np.array(w_c_t_p_1)) < convergence_threshhold):
            print(f"Converged after {t} iterations.")
            iterations_until_convergence.append(t)
            break

        prev_f.append(w_f_t_p_1)
        prev_c.append(w_c_t_p_1)
    print_final_convergence(w_f_t_p_1, w_c_t_p_1, S_f, S_c, iterations_until_convergence)
    return prev_f, prev_c, iterations_until_convergence


def plot_max_strategies(results):
    plt.figure(figsize=(12, 6))
    max_strats = [[],[]]

    for run_results in results:
        for result in run_results[0]:
            max_strats[0].append(np.argmax(result))
        for result in run_results[1]:
            max_strats[1].append(np.argmax(result))

        plt.plot(max_strats[0], label='Firm Strategies')
        plt.plot(max_strats[1], label='Candidate Strategies')

    plt.xlabel('Time Steps')
    plt.ylabel('Strategy Probability')
    plt.title('Evolution of Strategies Over Time')
    plt.legend()
    plt.show()

def print_initial_conditions(beta_f, beta_c, alpha_f, alpha_c, S_f, S_c):
    print("----initial parameters----")
    print(f"beta_f: {get_support(beta_f,S_f)}")
    print(f"beta_c: {get_support(beta_c,S_c)}")
    print(f"alpha_f: {get_support(alpha_f,S_f)}")
    print(f"alpha_c: { get_support(alpha_c,S_c)}")
   
def print_final_convergence(w_f_t_p_1, w_c_t_p_1, S_f, S_c, iterations_until_convergence):
    print("----final convergence----")
    print(f"w_f_T: {w_f_t_p_1}")
    print("--non-zero support--")
    print(get_support(w_f_t_p_1, S_f))
    print(f"w_c_T: {w_c_t_p_1}")
    print("--non-zero support--")
    print(get_support(w_c_t_p_1, S_c))
    if iterations_until_convergence:
        print("--number of iterations until convergence--")
        print(iterations_until_convergence[-1])

if __name__ == "__main__":
    T = 100  # time steps
    M = 1 / np.sqrt(T)  # regularizer constant

    all_runs_results = []
    
    # Run multiple simulations
    num_runs = 1
    for _ in range(num_runs):
        run_results = run_simulation(T=T, M=M, strategy="biased pure")
        all_runs_results.append(run_results)

    plot_max_strategies(all_runs_results)



