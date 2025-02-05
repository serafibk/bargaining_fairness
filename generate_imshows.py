import cvxpy as cp
import numpy as np
import itertools
import tqdm
import matplotlib.pyplot as plt
import re

def agent_update(prev_not_i_strategies, A, delta, M, responder= False):
    # updating realization probabilities now

    if responder == False: # proposer
        # create utility feedback vector
        utility_feedback_vector = [0.0 for a in range(len(A)+2*len(A)**2)] # a_i, a_iAa_j,a_iRa_j low to high ordering (should never reject any offer)

        for r_r in prev_not_i_strategies: # go through record of responder strategies
            for i,a in enumerate(A): 
                utility_feedback_vector[i] += (1-a) * r_r[f"A{a:.2f}"][0] # a_i accepted
                for j,b in enumerate(A):
                    utility_feedback_vector[len(A)*(i+1)+j] += delta * b * r_r[f"R{a:.2f}{b:.2f}"][0] # accepts counter offer from responder       
        
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

    return r_t_p_1

    

def check_pure_NE(r_p,r_r, A,delta):
    ## checks for two possible pure NE corresponding to the proposer having an approx pure first round offer that is either accepted as a best response for both OR
    # first round offer is approx purely rejected by responder and a second round offer approx purely offered by responder is accepted as a best response

    # check first round offers 
    eps = 1e-2 # tolerance for approx NE (?)

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
        responder_second_round_utility.append(proposer_rejection_strategy[f"A{a:.2f}"][0]*delta*(1-a))
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
            return (True, 1-first_round_offer)
        else:
            print("responder approx pure acceptance is not best response")
            return (False, -1)
    else: # second round best, need (approx) pure offer at best_strategy_idx (todo - what if not unique?)
        if np.abs(1-r_r[f"R{first_round_offer:.2f}{A[best_strategy_idx]:.2f}"][0]) < eps and np.abs(1-proposer_rejection_strategy[f"A{A[best_strategy_idx]}"][0]) <eps:
            print(f"responder and proposer both best-responding in second round offer of {A[best_strategy_idx]}")
            return (True, delta*A[best_strategy_idx])
        else:
            print("check manually for other kinds of NE")
            return (False, -2)

    
# def generate_initial_points(beta_p, beta_r, A):
#     first_offer_idx = np.random.randint(len(A))
#     beta_p[f"{A[first_offer_idx]}"][0] = 1
#     for b in A:
#         second_response_idx = np.random.randint(2)
#         if second_response_idx == 0:
#             beta_p[f"{A[first_offer_idx]}"][1][f"A{b}"][0] = 1
#         else:
#             beta_p[f"{A[first_offer_idx]}"][1][f"R{b}"][0] = 1
    
#     # responder
#     for a in A:
#         response_idx = np.random.randint(len(A)+1)
#         if response_idx == 0:
#             beta_r[f"A{a}"][0] = 1
#         else:
#             beta_r[f"R{a}{A[response_idx-1]}"][0]=1

#     return beta_p, beta_r

# def check_NE(w_f,w_c, S,delta):

#     if len(get_support(w_f,S)) >1 or len(get_support(w_c,S)) >1:
#         return False
    
#     w_f_1 = get_support(w_f,S)[0][0]
#     w_f_2 = get_support(w_f,S)[0][1]

#     w_c_1 = get_support(w_c,S)[0][0]
#     w_c_2 = get_support(w_c,S)[0][1]

#     if w_f_1 == w_c_1:
#         if delta * (w_c_2) <= 1-w_f_1 and delta* (1-w_f_2) <= w_f_1:
#             return True
#         else:
#             return False
#     elif w_f_2 == w_c_2:
#         if delta * (w_c_2) >= 1-w_c_1 and delta* (1-w_c_2) >= w_f_1:
#             return True
#         else:
#             return False
#     else:
#         return False
    
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

def generate_imshow_two_round(M, D, T, delta, alpha_c_i=None, alpha_f_i=None):
    # (D+1)x(D+1)x(D+1)x(D+1) matrix
    avg_payoffs = np.zeros((D+1, D+1))
    A = [round(i/D,2) for i in range(0,D+1)] # action list for offers {1/D, ... , 1}

    # S_f = list(itertools.product([i/D for i in range(D+1)], [i/D for i in range(D+1)]))
    # S_c = list(itertools.product([i/D for i in range(D+1)], [i/D for i in range(D+1)]))

    for n_r1 in range(1,D+1):
        for n_r2 in range(1,D+1):
            payoffs = []
            for n_p1 in range(1,D+1):
                for n_p2 in range(1,D+1):
                    beta_p, beta_r = generate_new_betas(D)
                    # print(beta_p)
                    print(f"candidate strategy indices {(n_r1, n_r2)}, firm strategy indices {(n_p1, n_p2)}")

                    # generate pure strategy
                    # proposer 
                    first_offer_idx = n_p1 # np.random.randint(len(A))
                    second_response_idx = n_p2
                    beta_p[f"{A[first_offer_idx]:.2f}"][0] = 1
                    for b in A:
                        if b <= round(A[second_response_idx],2):
                            beta_p[f"{A[first_offer_idx]:.2f}"][1][f"A{b:.2f}"][0] = 1
                        else:
                            beta_p[f"{A[first_offer_idx]:.2f}"][1][f"R{b:.2f}"][0] = 1
                        # second_response_idx = np.random.randint(2)
                        # if second_response_idx == 0:
                        #     beta_p[f"{A[first_offer_idx]}"][1][f"A{b}"][0] = 1
                        # else:
                        #     beta_p[f"{A[first_offer_idx]}"][1][f"R{b}"][0] = 1
                    
                    # # responder
                    for a in A:
                        # response_idx = np.random.randint(len(A)+1)
                        if a >= A[n_r1]:
                            beta_r[f"A{a:.2f}"][0] = 1
                        else:
                            beta_r[f"R{a:.2f}{A[n_r2]:.2f}"][0]=1
                        # if response_idx == 0:
                        #     beta_r[f"A{a}"][0] = 1
                        # else:
                            # beta_r[f"R{a}{A[response_idx-1]}"][0]=1
                        

                    prev_p = [beta_p]
                    prev_r = [beta_r]

                    for t in tqdm.tqdm(range(T)):

                        r_p_t_p_1 = agent_update(prev_r,A=A,delta=delta, M=M)
                        r_r_t_p_1 = agent_update(prev_p,A=A,delta=delta, M=M, responder=True)
                        prev_p.append(r_p_t_p_1)
                        prev_r.append(r_r_t_p_1)
        
                    print("----initial parameters----")
                    print(f"beta_p: {beta_p}")
                    print(f"beta_r: {beta_r}")

                    print("----final convergence----")
                    # print(f"w_f_T: {w_f_t_p_1}")
                    print("--proposer--")
                    print(r_p_t_p_1)
                    # print(f"w_c_T: {w_c_t_p_1}")
                    print("--responder--")
                    print(r_r_t_p_1)

                    NE_check, calc_payoff = check_pure_NE(r_p_t_p_1,r_r_t_p_1,A,delta)
                    print(f"NE: {NE_check}")
                    print("--------------------------")
                    print()

                    # if calc_payoff == -2:
                        # check for mixed NE
                    # else:
                        # payoff = calc_payoff
                    payoffs.append(calc_payoff)
            avg_payoffs[n_r1][n_r2] = sum(payoffs) / len(payoffs)

    # Save data to file
    # with open("imshows_2_round.txt", "a") as f:
    #     msg = f'\nD: {D} M: {M} T: {T} delta: {delta} \n' # later add alpha_c_idx: {alpha_c_idx} alpha_f_idx: {alpha_f_idx}
    #     f.write(msg)
    #     np.savetxt(f, payoffs.reshape(-1, payoffs.shape[-1]), fmt='%.4f')
    #     f.write('\n')

    # Create plots
    fig, ax = plt.subplots()
    fig.suptitle(f"Initial strategies vs. average responder payoff value at NE\n(η={M:.4f}, D={D}, T={T}, δ={delta:.2f})")

    im = ax.imshow(avg_payoffs, cmap='viridis', origin='lower')
    ax.set_xticks(np.arange(0, D+1, D//5))
    ax.set_yticks(np.arange(0, D+1, D//5))
    # ax.set_xticks(np.arange(D/5), labels=[o for o in set(R_second_round_initial)])
    # ax.set_yticks(np.arange(len(indices)), labels=[at for at in set(R_first_round_initial)])

    ax.set_ylabel("Responder First Round Strategy - Acceptance Threshold Values")
    ax.set_xlabel("Responder Second Round Strategy - Offer Values")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.show()

generate_imshow_two_round(M=0.5, D=4, T=500, delta=0.9)