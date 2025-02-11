import cvxpy as cp
import numpy as np
import tqdm
import matplotlib.pyplot as plt

def NE_check(w_f, w_w, S):
    utilities = []
    pure = False
    for i in range(len(S)):
        if i <= np.argmax(w_f):
            utilities.append(sum([w_w[j] for j in range(i+1)])*(1-S[i]))


    if np.argmax(utilities) == np.argmax(w_f):
        max_firm = max(w_f)
        tied_firm = [ind for ind, ele in enumerate(w_f) if ele == max_firm]
        firm_offer = S_f[min(tied_firm)]
        if len(tied_firm) == 1 and 1-w_f[min(tied_firm)]<eps:
            pure = True
        return True, firm_offer, pure
    else:
        return False, -1, pure

def agent_update(prev_not_i_strategies, S_i, alpha_i, M, worker= False):

    utility_feedback_vector = np.zeros_like(S_i, dtype=np.float64)

    if worker == True:
        for w_f in prev_not_i_strategies:
            for j, S_w in enumerate(S_i):
                for k, w_f_supp in enumerate(S_i):
                    if w_f_supp >= S_w:
                        utility_feedback_vector[j] += w_f[k] *  w_f_supp
    else: # firm
        for w_w in prev_not_i_strategies:
            for j, s_f in enumerate(S_i):
                for k, w_w_supp in enumerate(S_i):
                    if w_w_supp <= s_f:
                        utility_feedback_vector[j] += w_w[k] *  (1-s_f)

    w_var = cp.Variable(len(S_i))
    
    # Euclidean regularizer (default)
    objective = cp.Maximize(w_var@utility_feedback_vector - cp.norm(w_var-alpha_i, 2)**2/(2*M))

    constraints = [cp.sum(w_var)==1, cp.min(w_var)>=0.0]
    problem = cp.Problem(objective,constraints)
    problem.solve()
    obj_min_strat = [max(np.float64(0.0), w_p.value) for w_p in w_var] #objective maximizing strategy

    return obj_min_strat

def get_support(w_i, S_i):
    return [S_i[i] for i in range(len(w_i)) if w_i[i] > 0]

def check_convergence(w_f_t_p_1, w_w_t_p_1, prev_f, prev_w, t, window_size=10, convergence_threshold=1e-8):
    if t >= window_size:
        f_converged = all(np.linalg.norm(np.array(w_f_t_p_1) - np.array(prev_f[-i])) < convergence_threshold for i in range(1, window_size + 1))
        c_converged = all(np.linalg.norm(np.array(w_w_t_p_1) - np.array(prev_w[-i])) < convergence_threshold for i in range(1, window_size + 1))
        return f_converged and c_converged
    return False

def generate_betas(strategy=None):
    # default random pure initial strategy
    beta_f_idx = np.random.randint(len(S_f))
    beta_c_idx = np.random.randint(len(S_w))
    # input manual initial strategy (by index); param format: f"manual {idxf} {idxc}"
    if strategy and strategy.find("manual")!=-1:
        _, idxf, idxc = strategy.split(' ')
        beta_f_idx = int(idxf)
        beta_c_idx = int(idxc)
    # biased initial strategies; param format: "biased"
    elif strategy == "biased":
        beta_f_idx = np.random.randint(len(S_f) // 2) # lower
        beta_c_idx = np.random.randint(len(S_w) // 2, len(S_w)) # upper
    # reverse biased initial strategies; param format: "reversed"
    elif strategy == "reversed":
        beta_f_idx = np.random.randint(len(S_f) // 2, len(S_f)) # upper
        beta_c_idx = np.random.randint(len(S_w) // 2) # lower
    # fixed initial strategy (relative to length of strategy space)
    # param format: f"fixed {f} {c}"  (where f,c denote what to divide the length of the strategy space by)
    elif strategy and strategy.find("fixed")!=-1: 
        _, idxf, idxc = strategy.split(' ')
        if idxf == "len":
            beta_f_idx = len(S_f)-1
        elif idxf == "0":
            beta_f_idx = 0
        else:
            beta_f_idx = int(len(S_f)/float(idxf))
        if idxc == "len":
            beta_c_idx = len(S_w)-1
        elif idxc == "0":
            beta_c_idx = 0
        else:
            beta_c_idx = int(len(S_w)/float(idxc))

    # create pure initial strategies
    beta_f = [1 if i == beta_f_idx else 0 for i in range(len(S_f))]
    beta_c = [1 if i == beta_c_idx else 0 for i in range(len(S_w))]
    return beta_f, beta_c

def generate_alphas(reference=None):
    # default random pure reference strategy
    alpha_f_idx = np.random.randint(len(S_f))
    alpha_w_idx = np.random.randint(len(S_w))

    # biased reference strategy; param format: "biased"
    if reference == "biased":
        alpha_f_idx = np.random.randint(len(S_f) // 2) # lower
        alpha_w_idx = np.random.randint(len(S_w) // 2, len(S_w)) # upper
    # reversed bias reference strategy; param format: "reversed"
    elif reference == "reversed":
        alpha_f_idx = np.random.randint(len(S_f) // 2, len(S_f)) # upper
        alpha_w_idx = np.random.randint(len(S_w) // 2) # lower
    # fixed reference strategy (relative to length of strategy space)
    # param format: f"fixed {f} {c}"  (where f,c denote what to divide the length of the strategy space by)
    elif reference and reference.find("fixed")!=-1: # input fixed f c  (where f,c denote what to divide the strategy space by)
        _, idxf, idxc = reference.split(' ')
        if idxf == "len":
            alpha_f_idx = len(S_f)-1
        elif idxf == "0":
            alpha_f_idx = 0
        else:
            alpha_f_idx = int(len(S_f)/float(idxf))
        if idxc == "len":
            alpha_w_idx = len(S_w)-1
        elif idxc == "0":
            alpha_w_idx = 0
        else:
            alpha_w_idx = int(len(S_w)/float(idxc))
    
    # create pure reference strategies
    alpha_f = [1 if i == alpha_f_idx else 0 for i in range(len(S_f))]
    alpha_w = [1 if i == alpha_w_idx else 0 for i in range(len(S_w))]
    return alpha_f, alpha_w

def run_simulation(S_f, S_w, T=100, M=None, strategy=None, reference=None):
    if M is None:
        M = 1 / np.sqrt(T)
    
    beta_f, beta_c = generate_betas(strategy)
    alpha_f, alpha_w = generate_alphas(reference)
    initial_conditions = print_initial_conditions(beta_f, beta_c, alpha_f, alpha_w, S_f, S_w)
    
    prev_f = [beta_f]
    prev_w = [beta_c]

    for t in tqdm.tqdm(range(T)):
        w_f_t_p_1 = agent_update(prev_w, S_i=S_f, alpha_i=alpha_f, M=M)
        w_w_t_p_1 = agent_update(prev_f, S_i=S_w, alpha_i=alpha_w, M=M, worker=True)

        # Check convergence - use the past 10 strats & threshold 1e-7 by default
        num_iter=0
        if check_convergence(w_f_t_p_1, w_w_t_p_1, prev_f, prev_w, t):
            print(f"Converged after {t} iterations.")
            num_iter=t
            break

        prev_f.append(w_f_t_p_1)
        prev_w.append(w_w_t_p_1)

    return prev_f, prev_w, initial_conditions, num_iter


def plot_max_strategies(results, S_f, S_w):
    plt.figure(figsize=(12, 6))

    for run_results in results:
        max_strats = [[],[]]
        for idx, result in enumerate(run_results[0]):
            opt = max(result)
            tied = [ind for ind, ele in enumerate(result) if ele == opt]
            max_strats[0].append(S_f[min(tied)])

            # get max cumulative probability for worker
            cum_probs = np.zeros_like(result)
            for i in range(len(S_w)):
                if i <= np.argmax(result):
                    cum_probs[i] = (sum([run_results[1][idx][j] for j in range(i+1)]))
            max_strats[1].append(S_w[np.argmax(cum_probs)])

        plt.plot(max_strats[0], label='Firm Strategies')
        plt.plot(max_strats[1], label='Worker Strategies')

        plt.xlabel('Time Steps')
        plt.ylabel('Max Strategy')
        plt.title('Evolution of Strategies Over Time')
        plt.legend()
        plt.show()

def print_initial_conditions(beta_f, beta_c, alpha_f, alpha_w, S_f, S_w):
    initial_conditions = {
        'beta_f': get_support(beta_f,S_f),
        'beta_c': get_support(beta_c,S_w),
        'alpha_f': get_support(alpha_f,S_f),
        'alpha_w': get_support(alpha_w,S_w)
    }
    print("----initial parameters----")
    for condition in initial_conditions.keys():
        print(f"{condition}: {initial_conditions[condition]}")
    return initial_conditions
   
def print_final_convergence(w_f_t_p_1, w_w_t_p_1, S_f, S_w):
    print("----final convergence----")
    print(f"w_f_T: {w_f_t_p_1}")
    print("--non-zero support--")
    print(get_support(w_f_t_p_1, S_f))
    print(f"w_w_T: {w_w_t_p_1}")
    print("--non-zero support--")
    print(get_support(w_w_t_p_1, S_w))


if __name__ == "__main__":
    T = 350  # time steps
    D = 10 # Size of strategy space range
    M = 1 / (T**(1/4.0))  # regularizer constant (learning rate)
    
    S_f = [i / (D) for i in range(D + 1)]
    S_w = [i / (D) for i in range(D + 1)]

    all_runs_results = []
    eps = 1e-6
    # Run multiple simulations
    num_runs = 10
    ne_convergence_data = []

    for _ in range(num_runs):
        run_results = run_simulation(S_f, S_w, T=T, M=M, strategy=None, reference=None)
        all_runs_results.append(run_results)
        converged_to_NE, payoff, pure = NE_check(run_results[0][-1], run_results[1][-1], S_f)
        
        if run_results[3] and converged_to_NE:
            if pure:
                print("Converged to pure NE in ", T, " steps")
            else:
                print("Converged to NE in ", T, " steps")
            print("Achieved Nash Equilibrium: ", converged_to_NE, f"; Offer: {payoff:.4f}; Pure NE: {pure}")
            ne_convergence_data.append({'initial_conditions': run_results[2], 'final_deal': payoff})
        elif converged_to_NE:
            print("Achieved Nash Equilibrium: ", converged_to_NE, f"; Offer: {payoff:.4f}; Pure NE: {pure}")
            ne_convergence_data.append({'initial_conditions': run_results[2], 'final_deal': payoff})
            print("Did not converge in ", T, f" steps. Final strategies were in NE")
        else:
            print("Did not converge in ", T, " steps")

    print(f"Total convergences to NE: {len(ne_convergence_data)}")

    for data in ne_convergence_data:
        print(f"Initial conditions: {data['initial_conditions']}")
        print(f"Worker payoff: {data['final_deal']}")
        print("---")
    
    # plot_max_strategies(all_runs_results, S_f, S_w)
