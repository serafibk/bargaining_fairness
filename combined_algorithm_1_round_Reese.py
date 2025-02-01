import cvxpy as cp
import numpy as np
import itertools
import tqdm
import matplotlib.pyplot as plt
import re

def NE_check(w_f,w_c, S):

    utilities = []
    for i in range(len(S)):
        if i <= np.argmax(w_f):
            utilities.append(sum([w_c[j] for j in range(i+1)])*(1-S[i]))
    
    # print(utilities)
    if np.argmax(utilities) == np.argmax(w_f):
        return True
    else:
        # print("here 3")
        return False
    
def check_mixed_NE(w_c,w_f,S):

    if 1-max(w_f)>1e-5:
        # print("here 1")
        return False


    if np.argmax(get_support(w_c,S)) != np.argmax(w_f):
        # print("here 2")
        return False
    
    utilities = []
    for i in range(len(S)):
        if i <= np.argmax(w_f):
            utilities.append(sum([w_c[j] for j in range(i+1)])*(1-S[i]))
    
    print(utilities)
    if np.argmax(utilities) == np.argmax(w_f):
        return True
    else:
        # print("here 3")
        return False

def agent_update(prev_not_i_strategies, S_i, alpha_i, M, regularizer="euclidean", solver='CLARABEL', responder= False):

    utility_feedback_vector = [np.float64(0.0) for s in S_i]


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
        objective = cp.Maximize(w_var@utility_feedback_vector - cp.norm(w_var-alpha_i, 2)**2/(2*M))

    constraints = [cp.sum(w_var)==1, cp.min(w_var)>=0.0]
    problem = cp.Problem(objective,constraints)
    # print(problem.is_dcp())
    problem.solve(solver=solver)
    obj_min_strat = [max(np.float64(0.0), w_p.value) for w_p in w_var] #objective maximizing strategy
    # print("Solver used:", problem.solver_stats.solver_name)

    return obj_min_strat
   
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

def check_convergence(w_f_t_p_1, w_c_t_p_1, prev_f, prev_c, t, window_size=10, convergence_threshold=1e-8, fast=False):
    if t >= window_size:
        f_converged = all(np.linalg.norm(np.array(w_f_t_p_1) - np.array(prev_f[-i])) < convergence_threshold for i in range(1, window_size + 1))
        c_converged = all(np.linalg.norm(np.array(w_c_t_p_1) - np.array(prev_c[-i])) < convergence_threshold for i in range(1, window_size + 1))

        return f_converged and c_converged
    elif fast and t >= window_size:
        max_firm = max(w_f_t_p_1)
        max_cand = max(w_c_t_p_1)
        if (1-max_firm<0.001) and (1-max_cand<0.05):
            return True
    return False

def run_simulation(S_f, S_c, T=100, M=None, strategy=None, solver='CLARABEL', reference=None, fast=False):
    if M is None:
        M = 1 / np.sqrt(T)

    beta_f_idx = np.random.randint(len(S_f))
    beta_c_idx = np.random.randint(len(S_c))
    alpha_f_idx = np.random.randint(len(S_f))
    alpha_c_idx = np.random.randint(len(S_c))

    if strategy and strategy.find("manual")!=-1:
        _, idxf, idxc = strategy.split(' ')
        beta_f_idx = int(idxf)
        beta_c_idx = int(idxc)
    elif strategy == "biased":
        beta_f_idx = np.random.randint(len(S_f) // 2) # lower
        beta_c_idx = np.random.randint(len(S_c) // 2, len(S_c)) # upper
    elif strategy == "reversed":
        beta_f_idx = np.random.randint(len(S_f) // 2, len(S_f)) # upper
        beta_c_idx = np.random.randint(len(S_c) // 2) # lower
    elif strategy and strategy.find("fixed")!=-1: # input fixed f c  (where f,c denote what to divide the strategy space by)
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
    if reference == "biased":
        alpha_f_idx = np.random.randint(len(S_f) // 2) # lower
        alpha_c_idx = np.random.randint(len(S_c) // 2, len(S_c)) # upper
    elif reference == "reversed":
        alpha_f_idx = np.random.randint(len(S_f) // 2, len(S_f)) # upper
        alpha_c_idx = np.random.randint(len(S_c) // 2) # lower
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
    beta_f = [1 if i == beta_f_idx else 0 for i in range(len(S_f))]
    beta_c = [1 if i == beta_c_idx else 0 for i in range(len(S_c))]
    alpha_f = [1 if i == alpha_f_idx else 0 for i in range(len(S_f))]
    alpha_c = [1 if i == alpha_c_idx else 0 for i in range(len(S_c))]
    
    initial_conditions = print_initial_conditions(beta_f, beta_c, alpha_f, alpha_c, S_f, S_c)
    
    prev_f = [beta_f]
    prev_c = [beta_c]

    final_deal = None

    for t in tqdm.tqdm(range(T)):
        w_f_t_p_1 = agent_update(prev_c, S_i=S_f, alpha_i=alpha_f, M=M, solver=solver)
        w_c_t_p_1 = agent_update(prev_f, S_i=S_c, alpha_i=alpha_c, M=M, solver=solver, responder=True)

        # print(f"w_f_t_p_1 support: {get_support(w_f_t_p_1, S_f)}")
        # print(f"w_c_t_p_1 support: {get_support(w_c_t_p_1, S_c)}")

        # if np.argmax(w_f_t_p_1) >= np.argmax(w_c_t_p_1):
        #     print(f"Offer likely accepted at time {t}: {np.argmax(w_f_t_p_1)}.")

        # Check convergence - use the past 10 strats & threshold 1e-7 by default
        num_iter=0
        if check_convergence(w_f_t_p_1, w_c_t_p_1, prev_f, prev_c, t, fast=fast):
            print(f"Converged after {t} iterations.")
            num_iter=t
            break

        prev_f.append(w_f_t_p_1)
        prev_c.append(w_c_t_p_1)
    # print_final_convergence(w_f_t_p_1, w_c_t_p_1, S_f, S_c, iterations_until_convergence)
    return prev_f, prev_c, initial_conditions, num_iter


def plot_max_strategies(results, S_f, S_c):
    plt.figure(figsize=(12, 6))

    for run_results in results:
        max_strats = [[],[]]
        for result in run_results[0]:
            opt = max(result)
            tied = [ind for ind, ele in enumerate(result) if ele == opt]
            max_strats[0].append(S_f[min(tied)])
        for result in run_results[1]:
            opt = max(result)
            tied = [ind for ind, ele in enumerate(result) if ele == opt]
            max_strats[1].append(S_c[max(tied)])

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


def generate_imshow(M, D, T, alpha_c_i=None, alpha_f_i=None):
    # DxD matrix
    payoffs = np.zeros((D+1, D+1))

    S_f = [i / (D) for i in range(D + 1)]
    S_c = [i / (D) for i in range(D + 1)]

    for s_f in range(D+1):
        for s_c in range(D+1):
            print(f"candidate strategy {(S_c[s_c])}, firm strategy {(S_f[s_f])}")
        
            beta_c_idx = s_c
            beta_f_idx = s_f
            alpha_f_idx = alpha_f_i if alpha_f_i else -1
            alpha_c_idx = alpha_c_i if alpha_c_i else -1

            beta_f = [1 if i == beta_f_idx else 0 for i in range(len(S_f))]
            beta_c = [1 if i == beta_c_idx else 0 for i in range(len(S_c))]
            alpha_f = [1 if i == alpha_f_idx else 0 for i in range(len(S_f))]
            alpha_c = [1 if i == alpha_c_idx else 0 for i in range(len(S_c))]

            prev_f = [beta_f]
            prev_c = [beta_c]

            for t in tqdm.tqdm(range(T)):
                w_f_t_p_1 = agent_update(prev_c, S_i=S_f, alpha_i=alpha_f, M=M, solver=solver)
                w_c_t_p_1 = agent_update(prev_f, S_i=S_c, alpha_i=alpha_c, M=M, solver=solver, responder=True)

                # print(f"w_f_t_p_1 support: {get_support(w_f_t_p_1, S_f)}")
                # print(f"w_c_t_p_1 support: {get_support(w_c_t_p_1, S_c)}")

                # if np.argmax(w_f_t_p_1) >= np.argmax(w_c_t_p_1):
                #     print(f"Offer likely accepted at time {t}: {np.argmax(w_f_t_p_1)}.")

                # Check convergence - use the past 10 strats & threshold 1e-7 by default

                if check_convergence(w_f_t_p_1, w_c_t_p_1, prev_f, prev_c, t):
                    print(f"Converged after {t} iterations.")
                    break

                prev_f.append(w_f_t_p_1)
                prev_c.append(w_c_t_p_1)
 
            print("----initial parameters----")
            print(f"beta_f: {get_support(beta_f,S_f)}")
            print(f"beta_c: {get_support(beta_c,S_c)}")
            print(f"alpha_f: {get_support(alpha_f,S_f)}")
            print(f"alpha_c: { get_support(alpha_c,S_c)}")
        

            print("----final convergence----")
            print(f"w_f_T: {w_f_t_p_1}")
            print("--non-zero support--")
            print(get_support(w_f_t_p_1, S_f))
            print(f"w_c_T: {w_c_t_p_1}")
            print("--non-zero support--")
            print(get_support(w_c_t_p_1, S_c))

            check_NE = NE_check(w_f_t_p_1,w_c_t_p_1, S_f)

            # print(f"NE: {NE_check}")

            if check_NE == False:
                firm_offer=-1
            else:
                max_firm = max(w_f_t_p_1)
                tied_firm = [ind for ind, ele in enumerate(w_f_t_p_1) if ele == max_firm]
                firm_offer = S_f[min(tied_firm)]
                print("Firm offer: ", firm_offer)

            payoffs[s_f][s_c] = firm_offer

    with open("imshows_1_round.txt", "a") as f:
        # Write data to the file
        msg = '\nD: ' + str(D) + ' M: ' + str(M) + ' T: ' + str(T) + ' alpha_c_idx: ' + str(alpha_c_idx) + ' alpha_f_idx: ' + str(alpha_f_idx) + '\n'
        f.write(msg)
        for arr in payoffs:
            f.write(str(arr))
            f.write('\n')
        f.write('\n')


    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(payoffs, cmap='viridis', origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Average Payoff', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('Candidate Initial Strategy')
    ax.set_ylabel('Firm Initial Strategy')
    title = f"Average Payoffs (η={M:.4f}, D={D}, T={T}"
    if alpha_f_i:
        title+=f', alpha_f: {S_f[alpha_f_idx]}'
    if alpha_c_i:
        title+=f', alpha_c: {S_c[alpha_c_idx]}'
    if not alpha_f_i and not alpha_c_i:
        title+=', No Reference'
    title+=')'
    ax.set_title(title)
    
    # Set tick labels
    ax.set_xticks(np.arange(0, D+1, D//5))
    ax.set_yticks(np.arange(0, D+1, D//5))
    ax.set_xticklabels([f"{S_c[i]:.2f}" for i in range(0, D+1, D//5)])
    ax.set_yticklabels([f"{S_f[i]:.2f}" for i in range(0, D+1, D//5)])
    
    plt.tight_layout()
    plt.show()


def generate_imshow_from_file(filename):
    with open(f"{filename}.txt", "r") as f:
        # read D, M, T, alpha_c_idx, alpha_f_idx from the file and store
        first_line = f.readline().strip().split()
        D = int(first_line[1])
        M = float(first_line[3])
        T = int(first_line[5])
        alpha_c_idx = int(first_line[7])
        alpha_f_idx = int(first_line[9])
        S_f = [i / (D) for i in range(D + 1)]
        S_c = [i / (D) for i in range(D + 1)]
        # msg = '\nD: ' + str(D) + ' M: ' + str(M) + ' T: ' + str(T) + ' alpha_c_idx: ' + str(alpha_c_idx) + ' alpha_f_idx: ' + str(alpha_f_idx) + '\n'
        # read each line as an array in the file and store in payoffs[line_index]
        payoffs = np.zeros((D+1, D+1))
        content=f.read()
        arrays = re.findall(r'\[([\s\S]*?)\]', content)

        # Initialize payoffs array
        payoffs = np.zeros((D+1, D+1))

        # Process each array
        for i, array_str in enumerate(arrays):
            # Remove newlines and split the array string into values
            values = array_str.replace('\n', '').split()
            # Convert string values to float and store in payoffs array
            payoffs[i] = [float(val) for val in values]
        # for i, line in enumerate(f):
        #     # Remove brackets and split the line into values
        #     values = line.strip()[1:-1].split()
        #     # Convert string values to float and store in payoffs array
        #     payoffs[i] = [float(val) for val in values]
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(payoffs, cmap='viridis', origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Average Payoff', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('Candidate Initial Strategy')
    ax.set_ylabel('Firm Initial Strategy')
    title = f"Average Payoffs (η={M:.4f}, D={D}, T={T}"
    if alpha_f_idx!=-1:
        title+=f', alpha_f: {S_f[alpha_f_idx]:.2f}'
    if alpha_c_idx!=-1:
        title+=f', alpha_c: {S_c[alpha_c_idx]:.2f}'
    if alpha_f_idx==-1 and alpha_c_idx==-1:
        title+=', No Reference'
    title+=')'
    ax.set_title(title)
    
    # Set tick labels
    ax.set_xticks(np.arange(0, D+1, D//5))
    ax.set_yticks(np.arange(0, D+1, D//5))
    ax.set_xticklabels([f"{S_c[i]:.2f}" for i in range(0, D+1, D//5)])
    ax.set_yticklabels([f"{S_f[i]:.2f}" for i in range(0, D+1, D//5)])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    T = 350  # time steps
    D = 30 # range 50-100 (fix in run sim)
    M = 1 / (T**(1/4.0))  # regularizer constant
    
    # add manual input for initial conditions for early convergences (not pure firm)
    idxf = 9 #10
    idxc = 2 #4
    strategy = f"manual {idxf} {idxc}"

    # strategy = None
    reference = "Fixed 0 1"
    solver = None
    S_f = [i / (D) for i in range(D + 1)]
    S_c = [i / (D) for i in range(D + 1)]

    all_runs_results = []
    purity_threshold = 5e-7
    # Run multiple simulations
    num_runs = 0
    pure_count = 0
    ne_convergence_data = []
    # generate_imshow(M, D, T, alpha_f_i=D, alpha_c_i=D)
    generate_imshow_from_file('imshows_1_round_regen')
    exit()
    for _ in range(num_runs):
        pure = False
        run_results = run_simulation(S_f, S_c, T=T, M=M, strategy=strategy, solver=solver, reference=reference)
        all_runs_results.append(run_results)

        max_firm = max(run_results[0][-1])
        tied_firm = [ind for ind, ele in enumerate(run_results[0][-1]) if ele == max_firm]
        max_cand = max(run_results[1][-1])
        tied_cand = [ind for ind, ele in enumerate(run_results[1][-1]) if ele == max_cand]
        firm_offer = S_f[min(tied_firm)]
        candidate_offer = S_c[max(tied_cand)]


        offer_gap = firm_offer-candidate_offer
        print("Firm offer = ", firm_offer)
        print("Candidate offer = ", candidate_offer)
        print("Offer gap = ", offer_gap)
        print("Firm probability of offer = ", max_cand)
        print("Candidate probability of offer = ", max_firm)
        if run_results[3]:
            if abs(max_firm - 1.0) < purity_threshold and abs(max_cand - 1) < purity_threshold:
                print("pure convergence")
                pure = True
            
            if offer_gap==0.0:
                if pure:
                    pure_count+=1
                ne_convergence_data.append({'initial_conditions': run_results[2], 'final_deal': run_results[3]})
        else:
            if offer_gap==0.0 and (abs(max_firm - 1.0) < .1):
                # likely will converge to pure NE
                final_deal = {
                'firm_offer': firm_offer,
                'candidate_offer': candidate_offer,
                'iterations': T
                }
                if abs(max_firm - 1.0) < purity_threshold and abs(max_cand - 1) < purity_threshold:
                    print("pure convergence")
                    pure_count+=1
                ne_convergence_data.append({'initial_conditions': run_results[2], 'final_deal': final_deal})
            else:
                print("Did not converge in ", T, " steps")

    print(f"Total convergences to NE: {len(ne_convergence_data)}")
    print(f"Total pure convergences to NE: {pure_count}")

    for data in ne_convergence_data:
        print(f"Initial conditions: {data['initial_conditions']}")
        print(f"Final deal: {data['final_deal']}")
        print("---")
        # print mixed strats at end
    # print("Installed solvers:", cp.installed_solvers())
    # plot_max_strategies(all_runs_results, S_f, S_c)


