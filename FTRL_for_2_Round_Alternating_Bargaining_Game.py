import cvxpy as cp
import numpy as np
import tqdm
import collections

def agent_update(prev_not_i_strategies, A, delta, M, worker= False):
    # updating realization probabilities now
    if worker == False: # firm
        # create utility feedback vector
        utility_feedback_vector = [0.0 for a in range(len(A)+2*len(A)**2)] # a_i, a_iAa_j,a_iRa_j low to high ordering (should never reject any offer)

        for r_w in prev_not_i_strategies: # go through record of worker strategies
            for i,a in enumerate(A): 
                utility_feedback_vector[i] += (1-a) * r_w[f"A{a:.2f}"][0] # a_i accepted
                for j,b in enumerate(A):
                    utility_feedback_vector[len(A)*(i+1)+j] += delta * b * r_w[f"R{a:.2f}{b:.2f}"][0] # accepts counter offer from worker       

        # set up problem
        r_var = cp.Variable(len(A)+2*len(A)**2)
        objective = cp.Maximize(r_var@utility_feedback_vector - cp.norm(r_var, 2)**2/(2*M)) 

        # create summation constraints
        constraints = [r_var[i]>=0 for i in range(len(A)+2*len(A)**2)] # non-negativity
        constraints.append(cp.sum(r_var[:len(A)])==1)
        for i in range(len(A)):
            for j in range(len(A)):
                constraints.append(r_var[i] == cp.sum(r_var[(i+1)*len(A)+j]+r_var[(i+1)*len(A)+len(A)**2+j])) # mass on a_iAa_j and a_iRa_j (after worker rejects a_i) sums to mass on a_i

    else: # worker
        # create utility feedback vector
        utility_feedback_vector = [0.0 for a in range(len(A)+len(A)**2)] # Aa_i, Ra_ia_j low to high ordering for each

        for r_f in prev_not_i_strategies: # go through record of firm strategies
            for i,a in enumerate(A):
                utility_feedback_vector[i] += a * r_f[f"{a:.2f}"][0] # Aa_i, acceptance of initial offer
                for j,b in enumerate(A):
                    utility_feedback_vector[len(A)*(i+1)+j] += delta * (1-b) * r_f[f"{a:.2f}"][1][f"A{b:.2f}"][0] # Ra_ia_j, firm accepts second offer

        # set up problem
        r_var = cp.Variable(len(A)+len(A)**2)
        objective = cp.Maximize(r_var@utility_feedback_vector - cp.norm(r_var, 2)**2/(2*M))

        # create summation constraints
        constraints = [r_var[i]>=0 for i in range(len(A)+len(A)**2)] # non-negativity
        for i in range(len(A)):
            constraints.append(r_var[i] + cp.sum(r_var[len(A)*(i+1):len(A)*(i+2)]) == 1) # A_i, R_ia_j for each firm offer a_i is an extension of the empty sequence for the worker
        

    # solve problem and populate strategy dictionary r_t_p_1
    problem = cp.Problem(objective,constraints)
    problem.solve()

    mass_values = [max(0,m) for m in r_var.value] # accounts for computational errors (cp "correct" up to 1e-8)
    
    # create realization plans
    if worker == False: # firm
        r_t_p_1 = {f"{a:.2f}" : [mass_values[i],dict({f"A{b:.2f}": [mass_values[(i+1)*len(A)+j]] for j,b in enumerate(A)}.items(), **{f"R{b:.2f}":[mass_values[(i+1)*len(A)+len(A)**2+j]] for j,b in enumerate(A)})] for i,a in enumerate(A)}
    else: # worker
        r_t_p_1 = dict({f"A{a:.2f}":[mass_values[i]] for i,a in enumerate(A)}, **{f"R{a:.2f}{b:.2f}":[mass_values[len(A)*(i+1)+j]] for i,a in enumerate(A) for j,b in enumerate(A)})
    return r_t_p_1


def NE_check_2_rounds(r_f,r_w, A, delta, eps):
    # firm's best response: given the worker's strategy, determine the offer a_i that maximizes the firm's expected payoff.
    p_r1_expected_utility = np.zeros(len(A))
    f_offer_masses = np.zeros(len(A))
    for i,a in enumerate(A):
        p_r1_expected_utility[i] += r_w[f"A{a:.2f}"][0]*(1-a)
        f_offer_masses[i] = r_f[f"{a:.2f}"][0]

    # worker's best response: knowing firm gave the offer (pr 1), utility = a if accepted
    r_w1_expected_utility = [a for a in A] 

    # In the second round, decide whether to accept or reject a_j based on the worker's counteroffer strategy.
    p_r2_expected_utility = collections.defaultdict(list)
    r_w2_expected_utility = collections.defaultdict(list)
    p_cumulative_utility = collections.defaultdict(int)
    for i,a in enumerate(A):
        p_cumulative_utility[a] += p_r1_expected_utility[i]
        for j, b in enumerate(A):
            # worker second round utility
            if(r_f[f"{a:.2f}"][0] > eps):
                r_w2_expected_utility[a].append((r_f[f"{a:.2f}"][1][f"A{b:.2f}"][0])/(r_f[f"{a:.2f}"][0])*delta*(1-b))
            # firm second round utility
            p_r2_expected_utility[a].append((r_w[f"R{a:.2f}{b:.2f}"][0]/1)*delta*b)
        p_cumulative_utility[a] += sum(p_r2_expected_utility[a])

    # get the branch(es) (offer indices) for which firm yields the max utility
    max_f_utility = 0
    max_f_utility_idxs = []
    for i,a in enumerate(A):
        if abs(p_cumulative_utility[a] - max_f_utility) < eps:
            max_f_utility_idxs.append(i)
        elif p_cumulative_utility[a] > max_f_utility:
            max_f_utility = p_cumulative_utility[a]
            max_f_utility_idxs = [i]

    f_offer_idx = np.argmax(f_offer_masses)
    f_offer = A[f_offer_idx]
    
    if len(max_f_utility_idxs) == 1:
        print("firm's utility for offer ", f_offer, " = ", f"{p_cumulative_utility[f_offer]:.4f}", " with probability ", f"{f_offer_masses[f_offer_idx]:.4f}")
        print("Highest utility branch = offered: ", f_offer_idx in max_f_utility_idxs)
        if f_offer_idx not in max_f_utility_idxs:
            print("Error: Branch with highest probability of being offered is not getting most expected utility")
            return (False, -1)
        
        if abs(1-f_offer_masses[f_offer_idx])<eps:
            print("Pure NE found for first round!")
            # check that worker playing best strat for the only branch with > eps prob mass
            worker_flaying_best_strategy, w_offer_idx = check_worker_flaying_best_strategy(A, i, r_w, eps, r_w1_expected_utility, r_w2_expected_utility[A[f_offer_idx]])
            if worker_flaying_best_strategy and w_offer_idx == -1:
                print("Converge after first round")
                print("firm offers ", f_offer, f" with probability {f_offer_masses[f_offer_idx]:.4f}")
                print("worker accepts ", f_offer, " as best response.") 
                return (True, f_offer)
        return (False, -1)
    else:
        if len(max_f_utility_idxs) > 1:
            # mixed NE
            print("Mixed nash equilibrium found!")
        else:
            print("No NE found.")
            return (False, -1)
        
        w_offer_idx_max = None
        # check that worker playing best strat for all branches (first round offers) with > eps prob mass
        for i, mass in enumerate(f_offer_masses):
            if mass > eps:
                # make sure worker is playing the best strategy in each
                worker_flaying_best_strategy, w_offer_idx = check_worker_flaying_best_strategy(A, i, r_w, eps, r_w1_expected_utility, r_w2_expected_utility[A[i]])
                if not worker_flaying_best_strategy:
                    return (False, -1)
                if i == f_offer_idx:
                    w_offer_idx_max = w_offer_idx
        if not w_offer_idx_max:
            return (False, -1)
        elif w_offer_idx_max==-1:
            # worker's best strategy is to accept
            print("Converge after first round")
            print("firm offers ", f_offer, f" with probability {f_offer_masses[f_offer_idx]:.4f}")
            print("worker accepts ", f_offer, " as best response.")  
            return (True, f_offer)
        else:
            w_offer = A[w_offer_idx_max]
            print("worker's offer yield's maximum utility")
            print("Converge after second round, worker rejects initial offer: ", f_offer)
            print("worker counteroffers: ", w_offer, " with probability ", f"{r_w[f"R{f_offer:.2f}{w_offer:.2f}"][0]:.4f}")
            print("firm accepts counteroffer: ", w_offer, " with probability ", f"{(r_f[f"{f_offer:.2f}"][1][f"A{w_offer:.2f}"][0])/(r_f[f"{f_offer:.2f}"][0]):.4f}")
            return (True, delta*(1-w_offer))


def check_worker_flaying_best_strategy(A, offer_idx, r_w, eps, w_acc_expected_utility, w_rej_expected_utility):
    offer = A[offer_idx]
    w_rej_offer_masses = np.zeros(len(A))
    for j,b in enumerate(A):
        w_rej_offer_masses[j] = r_w[f"R{offer:.2f}{b:.2f}"][0]
    w_offer_idx = np.argmax(w_rej_offer_masses)
    w_offer = A[w_offer_idx]
    
    max_w_utility = 0
    max_w_utility_idxs = []
    for i,b in enumerate(w_rej_expected_utility):
        if abs(b - max_w_utility) < eps:
            max_w_utility_idxs.append(i)
        elif b > max_w_utility:
            max_w_utility = b
            max_w_utility_idxs = [i]

    if r_w[f"A{offer:.2f}"][0]>=r_w[f"R{offer:.2f}{w_offer:.2f}"][0]:
        # verify that utility of accepting > rejecting
        return (w_acc_expected_utility[offer_idx] >= w_rej_expected_utility[w_offer_idx], -1)
    else:
        # verify worker's offer yield's maximum utility and utility of rejecting > accepting
        if w_offer_idx in max_w_utility_idxs:
            return (w_acc_expected_utility[offer_idx] <= w_rej_expected_utility[w_offer_idx], w_offer_idx)
        print()
        return (False, None)


def get_support(w_i,S_i):
    # print("----non-zero support----")
    non_zero_support = []
    for i,w in enumerate(w_i):
        if w>0:
            non_zero_support.append(S_i[i])
    return non_zero_support

def generate_new_betas(D):
    beta_f = {}
    for i in range(0,D + 1):
        key = f"{i/D:.2f}"
        inner_dict = {}
        for j in range(0,D + 1):
            inner_key_a = f"A{j/D:.2f}"
            inner_key_r = f"R{j/D:.2f}"
            inner_dict[inner_key_a] = [0, '-']
            inner_dict[inner_key_r] = [0, '-']
        beta_f[key] = [0, inner_dict]

    beta_w = {}
    for i in range(0,D + 1):
        key_a = f"A{i/D:.2f}"
        beta_w[key_a] = [0, '-']
        for j in range(0,D + 1):
            key_r = f"R{i/D:.2f}{j/D:.2f}"
            beta_w[key_r] = [0, '-']
    return (beta_f, beta_w)

if __name__ == "__main__":
    D = 5
    T = 1000
    M = 0.5
    delta = 0.9
    eps = 1e-5
    num_rounds = 5
    A = [round(i/D,2) for i in range(0,D+1)] # action list for offers {1/D, ... , 1}
    ne_convergence_data = []
    verbose = True

    for _ in range(num_rounds):
        # generate random pure initial strategies
        n_f1 = np.random.randint(len(A))
        n_f2 = np.random.randint(len(A))
        n_w1 = np.random.randint(len(A))
        n_w2 = np.random.randint(len(A))
        beta_f, beta_w = generate_new_betas(D)
        
        # generate strategy arrays
        # firm 
        first_offer_idx = n_f1 
        second_response_idx = n_f2
        beta_f[f"{A[first_offer_idx]:.2f}"][0] = 1
        for b in A:
            if b <= A[second_response_idx]:
                beta_f[f"{A[first_offer_idx]:.2f}"][1][f"A{b:.2f}"][0] = 1
            else:
                beta_f[f"{A[first_offer_idx]:.2f}"][1][f"R{b:.2f}"][0] = 1
        # worker
        for a in A:
            if a >= A[n_w1]:
                beta_w[f"A{a:.2f}"][0] = 1
            else:
                beta_w[f"R{a:.2f}{A[n_w2]:.2f}"][0]=1
            
        prev_f = [beta_f]
        prev_w = [beta_w]

        for t in tqdm.tqdm(range(T)):

            r_f_t_p_1 = agent_update(prev_w,A=A,delta=delta, M=M)
            r_w_t_p_1 = agent_update(prev_f,A=A,delta=delta, M=M, worker=True)
            prev_f.append(r_f_t_p_1)
            prev_w.append(r_w_t_p_1)
        if verbose:
            print("----initial strategies----")
            print(f"beta_f: {beta_f}")
            print(f"beta_w: {beta_w}")

        if verbose:
            print("----final convergence----")
            print("--firm--")
            print(r_f_t_p_1)
            print("--worker--")
            print(r_w_t_p_1)

        NE_check, payoff = NE_check_2_rounds(r_f_t_p_1,r_w_t_p_1, A, delta, eps)
        if NE_check:
            ne_convergence_data.append({'initial_conditions': (prev_f[0], prev_w[0]), 'final_deal': payoff, 'NE': NE_check})
        
        if verbose:
            print(f"Coverged to NE: {NE_check}")
            print(f"Worker Payoff: {payoff:.4f}")
            print("--------------------------")
            print()

    print(f"Total convergences to NE: {len(ne_convergence_data)}")

    for data in ne_convergence_data:
        print(f"Initial conditions: {data['initial_conditions']}")
        print(f"Worker Payoff: {data['final_deal']:.4f}")
        print(f"Converged to NE: {data['NE']}")
        print("---")