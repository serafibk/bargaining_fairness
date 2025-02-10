import cvxpy as cp
import numpy as np
import tqdm
import matplotlib.pyplot as plt

def NE_check(w_f, w_c, S):
    utilities = []
    pure = False
    for i in range(len(S)):
        if i <= np.argmax(w_f):
            utilities.append(sum([w_c[j] for j in range(i+1)])*(1-S[i]))


    if np.argmax(utilities) == np.argmax(w_f):
        max_firm = max(w_f)
        tied_firm = [ind for ind, ele in enumerate(w_f) if ele == max_firm]
        firm_offer = S_f[min(tied_firm)]
        if len(tied_firm) == 1 and 1-w_f[min(tied_firm)]<eps:
            pure = True
        return True, firm_offer, pure
    else:
        return False, -1, pure

def agent_update(prev_not_i_strategies, S_i, alpha_i, M, responder= False):

    utility_feedback_vector = np.zeros_like(S_i, dtype=np.float64)

    if responder == True:
        for w_f in prev_not_i_strategies:
            for j, s_c in enumerate(S_i):
                for k, w_f_supp in enumerate(S_i):
                    if w_f_supp >= s_c:
                        utility_feedback_vector[j] += w_f[k] *  w_f_supp
    else: # proposer
        for w_c in prev_not_i_strategies:
            for j, s_f in enumerate(S_i):
                for k, w_c_supp in enumerate(S_i):
                    if w_c_supp <= s_f:
                        utility_feedback_vector[j] += w_c[k] *  (1-s_f)

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

def check_convergence(w_f_t_p_1, w_c_t_p_1, prev_f, prev_c, t, window_size=10, convergence_threshold=1e-8):
    if t >= window_size:
        f_converged = all(np.linalg.norm(np.array(w_f_t_p_1) - np.array(prev_f[-i])) < convergence_threshold for i in range(1, window_size + 1))
        c_converged = all(np.linalg.norm(np.array(w_c_t_p_1) - np.array(prev_c[-i])) < convergence_threshold for i in range(1, window_size + 1))
        return f_converged and c_converged
    return False

def generate_betas(strategy=None):
    # default random pure initial strategy
    beta_f_idx = np.random.randint(len(S_f))
    beta_c_idx = np.random.randint(len(S_c))
    # input manual initial strategy (by index); param format: f"manual {idxf} {idxc}"
    if strategy and strategy.find("manual")!=-1:
        _, idxf, idxc = strategy.split(' ')
        beta_f_idx = int(idxf)
        beta_c_idx = int(idxc)
    # biased initial strategies; param format: "biased"
    elif strategy == "biased":
        beta_f_idx = np.random.randint(len(S_f) // 2) # lower
        beta_c_idx = np.random.randint(len(S_c) // 2, len(S_c)) # upper
    # reverse biased initial strategies; param format: "reversed"
    elif strategy == "reversed":
        beta_f_idx = np.random.randint(len(S_f) // 2, len(S_f)) # upper
        beta_c_idx = np.random.randint(len(S_c) // 2) # lower
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
            beta_c_idx = len(S_c)-1
        elif idxc == "0":
            beta_c_idx = 0
        else:
            beta_c_idx = int(len(S_c)/float(idxc))

    # create pure initial strategies
    beta_f = [1 if i == beta_f_idx else 0 for i in range(len(S_f))]
    beta_c = [1 if i == beta_c_idx else 0 for i in range(len(S_c))]
    return beta_f, beta_c

def generate_alphas(reference=None):
    # default random pure reference strategy
    alpha_f_idx = np.random.randint(len(S_f))
    alpha_c_idx = np.random.randint(len(S_c))

    # biased reference strategy; param format: "biased"
    if reference == "biased":
        alpha_f_idx = np.random.randint(len(S_f) // 2) # lower
        alpha_c_idx = np.random.randint(len(S_c) // 2, len(S_c)) # upper
    # reversed bias reference strategy; param format: "reversed"
    elif reference == "reversed":
        alpha_f_idx = np.random.randint(len(S_f) // 2, len(S_f)) # upper
        alpha_c_idx = np.random.randint(len(S_c) // 2) # lower
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
            alpha_c_idx = len(S_c)-1
        elif idxc == "0":
            alpha_c_idx = 0
        else:
            alpha_c_idx = int(len(S_c)/float(idxc))
    
    # create pure reference strategies
    alpha_f = [1 if i == alpha_f_idx else 0 for i in range(len(S_f))]
    alpha_c = [1 if i == alpha_c_idx else 0 for i in range(len(S_c))]
    return alpha_f, alpha_c

def run_simulation(S_f, S_c, T=100, M=None, strategy=None, reference=None):
    if M is None:
        M = 1 / np.sqrt(T)
    
    beta_f, beta_c = generate_betas(strategy)
    alpha_f, alpha_c = generate_alphas(reference)
    initial_conditions = print_initial_conditions(beta_f, beta_c, alpha_f, alpha_c, S_f, S_c)
    
    prev_f = [beta_f]
    prev_c = [beta_c]

    for t in tqdm.tqdm(range(T)):
        w_f_t_p_1 = agent_update(prev_c, S_i=S_f, alpha_i=alpha_f, M=M)
        w_c_t_p_1 = agent_update(prev_f, S_i=S_c, alpha_i=alpha_c, M=M, responder=True)

        # Check convergence - use the past 10 strats & threshold 1e-7 by default
        num_iter=0
        if check_convergence(w_f_t_p_1, w_c_t_p_1, prev_f, prev_c, t):
            print(f"Converged after {t} iterations.")
            num_iter=t
            break

        prev_f.append(w_f_t_p_1)
        prev_c.append(w_c_t_p_1)

    return prev_f, prev_c, initial_conditions, num_iter


def plot_max_strategies(results, S_f, S_c):
    plt.figure(figsize=(12, 6))

    for run_results in results:
        max_strats = [[],[]]
        for idx, result in enumerate(run_results[0]):
            opt = max(result)
            tied = [ind for ind, ele in enumerate(result) if ele == opt]
            max_strats[0].append(S_f[min(tied)])

            # get max cumulative probability for responder
            cum_probs = np.zeros_like(result)
            for i in range(len(S_c)):
                if i <= np.argmax(result):
                    cum_probs[i] = (sum([run_results[1][idx][j] for j in range(i+1)]))
            max_strats[1].append(S_c[np.argmax(cum_probs)])

        plt.plot(max_strats[0], label='Firm Strategies')
        plt.plot(max_strats[1], label='Candidate Strategies')

        plt.xlabel('Time Steps')
        plt.ylabel('Max Strategy')
        plt.title('Evolution of Strategies Over Time')
        plt.legend()
        plt.show()

def print_initial_conditions(beta_f, beta_c, alpha_f, alpha_c, S_f, S_c):
    initial_conditions = {
        'beta_f': get_support(beta_f,S_f),
        'beta_c': get_support(beta_c,S_c),
        'alpha_f': get_support(alpha_f,S_f),
        'alpha_c': get_support(alpha_c,S_c)
    }
    print("----initial parameters----")
    for condition in initial_conditions.keys():
        print(f"{condition}: {initial_conditions[condition]}")
    return initial_conditions
   
def print_final_convergence(w_f_t_p_1, w_c_t_p_1, S_f, S_c):
    print("----final convergence----")
    print(f"w_f_T: {w_f_t_p_1}")
    print("--non-zero support--")
    print(get_support(w_f_t_p_1, S_f))
    print(f"w_c_T: {w_c_t_p_1}")
    print("--non-zero support--")
    print(get_support(w_c_t_p_1, S_c))


if __name__ == "__main__":
    T = 350  # time steps
    D = 10 # Size of strategy space range
    M = 1 / (T**(1/4.0))  # regularizer constant (learning rate)
    
    S_f = [i / (D) for i in range(D + 1)]
    S_c = [i / (D) for i in range(D + 1)]

    all_runs_results = []
    eps = 1e-6
    # Run multiple simulations
    num_runs = 10
    ne_convergence_data = []

    for _ in range(num_runs):
        run_results = run_simulation(S_f, S_c, T=T, M=M, strategy=None, reference=None)
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
        print(f"Final deal: {data['final_deal']}")
        print("---")
    
    # plot_max_strategies(all_runs_results, S_f, S_c)
