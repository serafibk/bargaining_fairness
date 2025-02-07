import cvxpy as cp
import numpy as np
import itertools
import tqdm
import matplotlib.pyplot as plt
import re
import collections
import json

def agent_update(prev_not_i_strategies, A, delta, M, responder= False):
    # updating realization probabilities now
    max_utility_idx = 0
    if responder == False: # proposer
        # create utility feedback vector
        utility_feedback_vector = [0.0 for a in range(len(A)+2*len(A)**2)] # a_i, a_iAa_j,a_iRa_j low to high ordering (should never reject any offer)

        for r_r in prev_not_i_strategies: # go through record of responder strategies
            for i,a in enumerate(A): 
                utility_feedback_vector[i] += (1-a) * r_r[f"A{a:.2f}"][0] # a_i accepted
                for j,b in enumerate(A):
                    utility_feedback_vector[len(A)*(i+1)+j] += delta * b * r_r[f"R{a:.2f}{b:.2f}"][0] # accepts counter offer from responder       
        # max_utility_idx = max(utility_feedback_vector)
        # tied_max_idxs = [ind for ind, ele in enumerate(utility_feedback_vector) if ele == max_utility_idx]
        # for i,a in enumerate(A):
        #     print(f"first round offer:{a}, total cumulative utility: {utility_feedback_vector[i] +sum([utility_feedback_vector[len(A)*(i+1)+j] for j in range(len(A))]) }")

        # set up problem
        r_var = cp.Variable(len(A)+2*len(A)**2)
        objective = cp.Maximize(r_var@utility_feedback_vector - cp.norm(r_var, 2)**2/(2*M)) 

        # create summation constraints
        # constraints = [cp.min(r_var)>=0, cp.sum(r_var[:len(A)])==1] # >=0 and first offer probabilities sum to 1 (first extension of empty sequence for proposer)
        constraints = [r_var[i]>=0 for i in range(len(A)+2*len(A)**2)]
        constraints.append(cp.sum(r_var[:len(A)])==1)
        for i in range(len(A)):
            for j in range(len(A)):
                constraints.append(r_var[i] == cp.sum(r_var[(i+1)*len(A)+j]+r_var[(i+1)*len(A)+len(A)**2+j])) # mass on a_iAa_j and a_iRa_j (after responder rejects a_i) sums to mass on a_i


    else: # responder
        # create utility feedback vector
        utility_feedback_vector = [0.0 for a in range(len(A)+len(A)**2)] # Aa_i, Ra_ia_j low to high ordering for each

        for r_p in prev_not_i_strategies: # go through record of proposer strategies
            for i,a in enumerate(A):
                utility_feedback_vector[i] += a * r_p[f"{a:.2f}"][0] # Aa_i, acceptance of initial offer
                for j,b in enumerate(A):
                    utility_feedback_vector[len(A)*(i+1)+j] += delta * (1-b) * r_p[f"{a:.2f}"][1][f"A{b:.2f}"][0] # Ra_ia_j, proposer accepts second offer
        # max_utility_idx = max(utility_feedback_vector)
        # tied_max_idxs = [ind for ind, ele in enumerate(utility_feedback_vector) if ele == max_utility_idx]
        # set up problem
        r_var = cp.Variable(len(A)+len(A)**2)
        objective = cp.Maximize(r_var@utility_feedback_vector - cp.norm(r_var, 2)**2/(2*M))

        # create summation constraints
        # constraints = [cp.min(r_var)>=0] # all masses >=0
        constraints = [r_var[i]>=0 for i in range(len(A)+len(A)**2)]
        for i in range(len(A)):
            constraints.append(r_var[i] + cp.sum(r_var[len(A)*(i+1):len(A)*(i+2)]) == 1) # A_i, R_ia_j for each proposer offer a_i is an extension of the empty sequence for the responder
        

    # solve problem and populate strategy dictionary r_t_p_1
    problem = cp.Problem(objective,constraints)
    problem.solve()

    mass_values = [max(0,m) for m in r_var.value] # accounts for computational errors (cp "correct" up to 1e-8)
    # for cs in constraints:
    #     print(f"constraint: {cs}, dual variable value: {cs.dual_value}")

    # exit()
    
    if responder == False: # proposer
        r_t_p_1 = {f"{a:.2f}" : [mass_values[i],dict({f"A{b:.2f}": [mass_values[(i+1)*len(A)+j]] for j,b in enumerate(A)}.items(), **{f"R{b:.2f}":[mass_values[(i+1)*len(A)+len(A)**2+j]] for j,b in enumerate(A)})] for i,a in enumerate(A)}
    else: # responder
        r_t_p_1 = dict({f"A{a:.2f}":[mass_values[i]] for i,a in enumerate(A)}, **{f"R{a:.2f}{b:.2f}":[mass_values[len(A)*(i+1)+j]] for i,a in enumerate(A) for j,b in enumerate(A)})
    # print("max utility index: ", max_utility_idx)
    return r_t_p_1

    
def convert_max_u_idxs(p_idxs, r_idxs, A):
    max_p_strats = []
    max_r_strats = []
    for p_idx in p_idxs:
        if p_idx > len(A):
            p_j = p_idx%(len(A))
            p_i = (p_idx-p_j-len(A))//len(A)
            # print(p_i)
            max_p_strat = f"{A[p_i]:.2f}A{A[p_j]:.2f}"
        else:
            p_i = p_idx
            p_j = -1
            max_p_strat = f"{A[p_i]:.2f}"
        max_p_strats.append(max_p_strat)
    for r_idx in r_idxs:
        if r_idx > len(A):
            r_j = r_idx%(len(A))
            r_i = (r_idx-r_j-len(A))//len(A)
            max_r_strat = f"R{A[r_i]:.2f}{A[r_j]:.2f}"
        else:
            r_i = r_idx
            r_j = -1
            max_r_strat = f"A{A[r_i]:.2f}"
        max_r_strats.append(max_r_strat)
    return (max_p_strats, max_r_strats)


def NE_check_2_rounds(r_p,r_r, A, delta, eps):
    # utilities: 
        # proposers: offering a_i, accepting/rejecting a_j
        # responders: accepting/rejecting a_i, making counter offer a_j
    # Expected payoffs: stored in utility feedback vector

    # proposer's best responses: 
    # Given the responder's strategy, determine the offer a_i that maximizes the proposer's expected payoff.
        # any weight that the respodner puts on an offer less than it, it will accept
    p_r1_expected_utility = np.zeros(len(A))
    p_offer_masses = np.zeros(len(A))
    for i,a in enumerate(A):
        p_r1_expected_utility[i] += r_r[f"A{a:.2f}"][0]*(1-a)
        p_offer_masses[i] = r_p[f"{a:.2f}"][0]


    # responder's best response: 
        # knowing proposer gave the offer (pr 1), utility = a if accepted
    r_r1_expected_utility = [a for a in A] 

    # In the second round, decide whether to accept or reject a_j based on the responder's counteroffer strategy.
    p_r2_expected_utility = collections.defaultdict(list)
    r_r2_expected_utility = collections.defaultdict(list)
    p_cumulative_utility = collections.defaultdict(int)
    for i,a in enumerate(A):
        p_cumulative_utility[a] += p_r1_expected_utility[i]
        for j, b in enumerate(A):
            # responder second round utility
            if(r_p[f"{a:.2f}"][0] > eps):
                r_r2_expected_utility[a].append((r_p[f"{a:.2f}"][1][f"A{b:.2f}"][0])/(r_p[f"{a:.2f}"][0])*delta*(1-b))
            # proposer second round utility
            p_r2_expected_utility[a].append((r_r[f"R{a:.2f}{b:.2f}"][0]/1)*delta*b)
        p_cumulative_utility[a] += sum(p_r2_expected_utility[a])

    max_p_utility = 0
    max_p_utility_idxs = []
    for i,a in enumerate(A):
        if abs(p_cumulative_utility[a] - max_p_utility) < eps:
            max_p_utility_idxs.append(i)
        elif p_cumulative_utility[a] > max_p_utility:
            max_p_utility = p_cumulative_utility[a]
            max_p_utility_idxs = [i]

    # print(max_p_utility_idxs)
    # print(p_cumulative_utility)
    p_offer_idx = np.argmax(p_offer_masses)
    p_offer = A[p_offer_idx]
    print("Proposer's utility for offer ", p_offer, " = ", f"{p_cumulative_utility[p_offer]:.4f}", " with probability ", f"{p_offer_masses[p_offer_idx]:.4f}")
    print("Highest utility branch = offered: ", p_offer_idx in max_p_utility_idxs)
    # print(p_cumulative_utility)

    r_rej_offer_masses = np.zeros(len(A))
    for j,b in enumerate(A):
        r_rej_offer_masses[j] = r_r[f"R{p_offer:.2f}{b:.2f}"][0]
    
    max_r_utility = 0
    max_r_utility_idxs = []
    for i,b in enumerate(p_r2_expected_utility[p_offer]):
        if abs(b - max_r_utility) < eps:
            max_r_utility_idxs.append(i)
        elif b > max_r_utility:
            max_r_utility = b
            max_r_utility_idxs = [i]
    r_offer_idx = np.argmax(r_rej_offer_masses)
    r_offer = A[r_offer_idx]

    if p_offer_idx not in max_p_utility_idxs:
        print("Error: Branch with highest probability of being offered is not getting most expected utility")
        return (False, -1)
    
    if len(max_p_utility_idxs) == 1 and abs(1-p_offer_masses[p_offer_idx])<eps:
        print("Pure NE found for first round!")
    elif len(max_p_utility_idxs) > 1:
        # mixed NE
        print("Mixed nash equilibrium found!")
    else:
        # print(max_p_utility_idxs)
        # print(max_r_utility_idxs)
        # print(r_rej_offer_masses)
        # print(r_r2_expected_utility[p_offer])
        print("No nash NE found.")
        return (False, -1)
    
    if(r_r[f"A{p_offer:.2f}"][0]>r_r[f"R{p_offer:.2f}{r_offer:.2f}"][0]):
        print("Converge after first round")
        print("Proposer offers ", p_offer, f" with probability {p_offer_masses[p_offer_idx]:.4f}")
        print("Responder accepts: ", p_offer, f" with probability {1-sum(r_rej_offer_masses):.4f}") 
        return (True, p_offer)
    else:
        # verify responder's offer yield's maximum utility
        if r_offer_idx in max_r_utility_idxs:
            print("Responder's offer yield's maximum utility")
            print("Converge after second round, responder rejects initial offer: ", p_offer)
            print("Responder counteroffers: ", r_offer, " with probability ", f"{r_rej_offer_masses[r_offer_idx]:.4f}")
            print("Proposer accepts counteroffer: ", r_offer, " with probability ", f"{(r_p[f"{p_offer:.2f}"][1][f"A{r_offer:.2f}"][0])/(r_p[f"{p_offer:.2f}"][0]):.4f}")
            return (True, delta*(1-r_offer))
        else:
            print("Error: Responder's branch with highest probability of being counter-offered is not getting most expected utility")
            return (False, -1)


def check_pure_NE(r_p,r_r, A,delta, eps):
    ## checks for two possible pure NE corresponding to the proposer having an approx pure first round offer that is either accepted as a best response for both OR
    # first round offer is approx purely rejected by responder and a second round offer approx purely offered by responder is accepted as a best response

    # check first round offers 
    # eps = 1e-8 # tolerance for approx NE (?)

    first_round_offer = 0
    for a in A:
        if np.abs(1-r_p[f"{a:.2f}"][0]) < eps:
            if first_round_offer > 0:
                print("ERROR: two approximately pure first round offers")
                print(first_round_offer)
                print(r_p[f"{a:.2f}"][0])
                exit()
            first_round_offer = a
    
    if first_round_offer == 0: # didn't find the kind of NE we're looking for 
        print("no pure first round offer from the proposer found, manually check for other kinds of NE")
        # potentially mixed NE
        return (False, -2)

    # check best-response of this first round offer
    for a in A[:A.index(first_round_offer)]: # check all lower offers don't give more utility (assuming first_round_offer is pure)
        if np.abs(1-r_r[f"A{a:.2f}"][0]) <eps:
            print("smaller offer would be approx purely accepted, proposer first round offer is not best-response")
            return (False, -1)

    # get proposer strategy if offer is rejected (will check best-response of it below)
    proposer_rejection_strategy = r_p[f"{first_round_offer:.2f}"][1]

    # find responder's best response given proposer strategy
    # utility from first round offer
    responder_first_round_utility = first_round_offer * r_p[f"{first_round_offer:.2f}"][0]

    # utility of both agents if responder doesn't accept first round offer
    responder_second_round_utility = []
    proposer_second_round_utility = []
    for a in A:
        responder_second_round_utility.append((proposer_rejection_strategy[f"A{a:.2f}"][0])/(r_p[f"{first_round_offer:.2f}"][0])*delta*(1-a))
        proposer_second_round_utility.append(r_r[f"R{first_round_offer:.2f}{a:.2f}"][0]*delta*a)

    if max(responder_second_round_utility) == responder_first_round_utility:
        print("possibly mixed NE case - check final strategies manually")
        return (False, -2)

    best_strategy_idx = -1
    if max(responder_second_round_utility) > responder_first_round_utility:
        best_strategy_idx = np.argmax(responder_second_round_utility) # set this only if first round acceptance is not best

    # check responder best response
    if best_strategy_idx == -1: # first round best, need (approx) pure acceptance
        if np.abs(1-r_r[f"A{first_round_offer:.2f}"][0]) < eps:
            print(f"approx pure first round acceptance at {first_round_offer}")
            return (True, first_round_offer)
        else:
            print("responder approx pure acceptance is not best response")
            return (False, -1)
    else: # second round best, need (approx) pure offer at best_strategy_idx (todo - what if not unique?)
        if np.abs(1-r_r[f"R{first_round_offer:.2f}{A[best_strategy_idx]:.2f}"][0]) < eps and np.abs(1-proposer_rejection_strategy[f"A{A[best_strategy_idx]}"][0]) <eps:
            print(f"responder and proposer both best-responding in second round offer of {A[best_strategy_idx]}")
            return (True, delta*(1-A[best_strategy_idx]))
        else:
            print("check manually for other kinds of NE")
            return (False, -2)


def get_support(w_i,S_i):

    # print("----non-zero support----")
    non_zero_support = []
    for i,w in enumerate(w_i):
        if w>0:
            non_zero_support.append(S_i[i])

    
    return non_zero_support

def generate_new_betas(D):
    beta_p = {}
    for i in range(0,D + 1):
        key = f"{i/D:.2f}"
        inner_dict = {}
        for j in range(0,D + 1):
            inner_key_a = f"A{j/D:.2f}"
            inner_key_r = f"R{j/D:.2f}"
            inner_dict[inner_key_a] = [0, '-']
            inner_dict[inner_key_r] = [0, '-']
        beta_p[key] = [0, inner_dict]

    beta_r = {}
    for i in range(0,D + 1):
        key_a = f"A{i/D:.2f}"
        beta_r[key_a] = [0, '-']
        for j in range(0,D + 1):
            key_r = f"R{i/D:.2f}{j/D:.2f}"
            beta_r[key_r] = [0, '-']
    return (beta_p, beta_r)

def generate_imshow_two_round(M, D, T, delta, eps, alpha_c_i=None, alpha_f_i=None):
    # (D+1)x(D+1)x(D+1)x(D+1) matrix
    avg_payoffs = np.zeros((D+1, D+1))
    # payoffs_to_store = [[[] for _ in range(D+1)] for _ in range(D+1)]

    A = [round(i/D,2) for i in range(0,D+1)] # action list for offers {1/D, ... , 1}

    # S_f = list(itertools.product([i/D for i in range(D+1)], [i/D for i in range(D+1)]))
    # S_c = list(itertools.product([i/D for i in range(D+1)], [i/D for i in range(D+1)]))
    # print(A)
    for n_p1 in range(0,D+1):
        for n_r1 in range(0,D+1):
            payoffs = []
            for n_p2 in range(0,D+1):
                for n_r2 in range(0,D+1):
                    beta_p, beta_r = generate_new_betas(D)
                    # print(beta_p)
                    print(f"candidate strategy indices {(n_r1, n_r2)}, firm strategy indices {(n_p1, n_p2)}")

                    # generate pure strategy
                    # proposer 
                    first_offer_idx = n_p1 
                    second_response_idx = n_p2
                    beta_p[f"{A[first_offer_idx]:.2f}"][0] = 1
                    for b in A:
                        if b <= A[second_response_idx]:
                            beta_p[f"{A[first_offer_idx]:.2f}"][1][f"A{b:.2f}"][0] = 1
                        else:
                            beta_p[f"{A[first_offer_idx]:.2f}"][1][f"R{b:.2f}"][0] = 1
                    # responder
                    for a in A:
                        if a >= A[n_r1]:
                            beta_r[f"A{a:.2f}"][0] = 1
                        else:
                            beta_r[f"R{a:.2f}{A[n_r2]:.2f}"][0]=1
                        
                    prev_p = [beta_p]
                    prev_r = [beta_r]
                    for t in tqdm.tqdm(range(T)):

                        r_p_t_p_1 = agent_update(prev_r,A=A,delta=delta, M=M)
                        r_r_t_p_1 = agent_update(prev_p,A=A,delta=delta, M=M, responder=True)
                        prev_p.append(r_p_t_p_1)
                        prev_r.append(r_r_t_p_1)
        
                    print("----initial parameters----")
                    # print(f"beta_p: {beta_p}")
                    # print(f"beta_r: {beta_r}")

                    print("----final convergence----")
                    # print(f"w_f_T: {w_f_t_p_1}")
                    print("--proposer--")
                    print(r_p_t_p_1)
                    # print(f"w_c_T: {w_c_t_p_1}")
                    print("--responder--")
                    print(r_r_t_p_1)

                    NE_check, calc_payoff = check_pure_NE(r_p_t_p_1,r_r_t_p_1,A,delta, eps)
                    if calc_payoff == -2: 
                        NE_check, calc_payoff = NE_check_2_rounds(r_p_t_p_1,r_r_t_p_1, A, delta, eps)
                    print(f"NE: {NE_check}")
                    print(f"Payoff: {calc_payoff:.4f}")
                    print("--------------------------")
                    print()

                    if calc_payoff >=0 :
                        payoffs.append(calc_payoff)
                    else:
                        with open("imshows_2_round.txt", "a") as f:
                            # Write data to the file
                            f.write(f"candidate strat idxs {(n_r1, n_r2)}, firm strat idxs {(n_p1, n_p2)} did not converge to NE\n")
                            f.write("proposer: \n")
                            json.dump(r_p_t_p_1, f)
                            f.write("responder: \n")
                            json.dump(r_r_t_p_1, f)
                            f.write("\nv")

                            
                        
            if len(payoffs)>0:
                avg_payoffs[n_p1][n_r1] = sum(payoffs) / len(payoffs)
            else:
                avg_payoffs[n_p1][n_r1] = -1

    # Save data to file
    with open("imshows_2_round.txt", "a") as f:
        # Write data to the file
        msg = f'\nD: {D} M: {M} T: {T} delta: {delta} epsilon: {eps:.0e}\n' # later add alpha_c_idx: {alpha_c_idx} alpha_f_idx: {alpha_f_idx}
        f.write(msg)
        # for arr in payoffs:
        #     f.write(str(arr))
        #     f.write('\n')
        np.savetxt(f, avg_payoffs.reshape(-1, avg_payoffs.shape[-1]), fmt='%.4f')
        f.write('\n')

    # Create plots
    fig, ax = plt.subplots()
    title = f"Average Responder Payoff Value at NE for Initial Strategies\n(η={M:.2f}, D={D}, T={T}, δ={delta:.2f}, ɛ={eps:.0e})"
    fig.suptitle(title)

    im = ax.imshow(avg_payoffs, cmap='viridis', origin='lower')
    ax.set_xticks(np.arange(1, D+1, 1))
    ax.set_yticks(np.arange(1, D+1, 1))

    ax.set_xticklabels([f"{A[i]:.2f}" for i in range(1, D+1, 1)])
    ax.set_yticklabels([f"{A[i]:.2f}" for i in range(1, D+1, 1)])

    ax.set_xlabel("Responder Round 1 Strategy - Acceptance Threshold Values")
    ax.set_ylabel("Proposer Round 1 Strategy - Offer Values")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    cbar = plt.colorbar(im)
    # plt.show()
    plt.savefig(f'img_{title}.png') 

# generate_imshow_two_round(M=0.5, D=2, T=100, delta=0.9)
# generate_imshow_two_round(M=0.35, D=4, T=250, delta=0.9)
# generate_imshow_two_round(M=0.5, D=4, T=2000, delta=0.9, eps=1e-6)
generate_imshow_two_round(M=0.5, D=5, T=1000, delta=0.9, eps=1e-5)