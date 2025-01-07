import cvxpy as cp
import numpy as np
import itertools
import tqdm
import matplotlib.pyplot as plt

# NO REFERENCE POINT CURRENTLY


def agent_update(prev_not_i_strategies, A, delta, M, responder= False):
    # updating realization probabilities now

    if responder == False: # proposer
        # create utility feedback vector
        utility_feedback_vector = [0.0 for a in range(len(A)+2*len(A)**2)] # a_i, Aa_j,Ra_j low to high ordering (should never reject any offer)

        for r_r in prev_not_i_strategies: # go through record of responder strategies
            for i,a in enumerate(A): 
                utility_feedback_vector[i] += (1-a) * r_r[f"A{a}"][0] # a_i accepted
                for j,b in enumerate(A):
                    utility_feedback_vector[len(A)*(i+1)+j] += delta * b * r_r[f"R{a}{b}"][0] # accepts counter offer from responder       

        # set up problem
        r_var = cp.Variable(len(A)+2*len(A)**2)
        objective = cp.Maximize(r_var@utility_feedback_vector - cp.norm(r_var, 2)/(2*M)) 

        # create summation constraints
        constraints = [cp.min(r_var)>=0, cp.sum(r_var[:len(A)])==1] # >=0 and first offer probabilities sum to 1
        for i in range(len(A)):
            for j in range(len(A)):
                constraints.append(r_var[i] == cp.sum(r_var[(i+1)*len(A)+j]+r_var[(i+1)*len(A)+len(A)**2+j])) # mass on Aa_j and Ra_j (after responder rejects a_i) sums to mass on a_i


    else: # responder
        # create utility feedback vector
        utility_feedback_vector = [0.0 for a in range(len(A)+len(A)**2)] # Aa_i, Ra_ia_j low to high ordering for each

        for r_p in prev_not_i_strategies: # go through record of proposer strategies
            for i,a in enumerate(A):
                utility_feedback_vector[i] += a * r_p[f"{a}"][0] # Aa_i, acceptance of initial offer
                for j,b in enumerate(A):
                    utility_feedback_vector[len(A)*(i+1)+j] += delta * (1-b) * r_p[f"{a}"][1][f"A{b}"][0] # Ra_ia_j, proposer accepts second offer
                
        # set up problem
        r_var = cp.Variable(len(A)+len(A)**2)
        objective = cp.Maximize(r_var@utility_feedback_vector - cp.norm(r_var, 2)**2/(2*M))

        # create summation constraints
        constraints = [cp.min(r_var)>=0]
        for i in range(len(A)):
            constraints.append(r_var[i] + cp.sum(r_var[len(A)*(i+1):len(A)*(i+2)]) == 1)
        

    # solve problem and populate strategy dictionary r_t_p_1
    problem = cp.Problem(objective,constraints)
    problem.solve()

    mass_values = [max(0,round(m, 9)) for m in r_var.value]
    
    if responder == False: # proposer
        r_t_p_1 = {f"{a}" : [mass_values[i],dict({f"A{b}": [mass_values[(i+1)*len(A)+j]] for j,b in enumerate(A)}.items(), **{f"R{b}":[mass_values[(i+1)*len(A)+len(A)**2+j]] for j,b in enumerate(A)})] for i,a in enumerate(A)}
    else: # responder
        # for i in range(len(A)):
        #     print(r_var.value[len(A)*(i+1):len(A)*(i+2)])
        #     # print(utility_feedback_vector[len(A)*(i+1):len(A)*(i+2)])
        #     # print(utility_feedback_vector[i])
        #     print(sum(r_var.value[len(A)*(i+1):len(A)*(i+2)])+r_var.value[i])
        # print("______________________")
        r_t_p_1 = dict({f"A{a}":[mass_values[i]] for i,a in enumerate(A)}, **{f"R{a}{b}":[mass_values[len(A)*(i+1)+j]] for i,a in enumerate(A) for j,b in enumerate(A)})
        # print(r_t_p_1)

    return r_t_p_1



def get_support(r_i,S_i):

    # print("----non-zero support----")
    non_zero_support = []
    for i,w in enumerate(r_i):
        if w>0:
            non_zero_support.append(S_i[i])

    
    return non_zero_support

    

def check_pure_NE(r_p,r_r, A,delta):

    # check first round offers 
    eps = 1e-2 # tolerance for approx NE (?)

    first_round_offer = 0
    for a in A:
        if np.abs(1-r_p[f"{a}"][0]) < eps:
            if first_round_offer > 0:
                print("ERROR: two approximately pure first round offers")
                print(first_round_offer)
                print(r_p[f"{a}"][0])
                exit()
            first_round_offer = a
    
    if first_round_offer == 0: # didn't find the kind of NE we're looking for 
        print("no pure first round offer from the proposer found, manually check for other kinds of NE")
        return False

    # check best-response of this first round offer
    for a in A[:A.index(first_round_offer)]: # check all lower offers don't give more utility (assuming first_round_offer is pure)
        if np.abs(1-r_r[f"A{a}"][0]) <eps:
            print("smaller offer would be approx purely accepted, proposer first round offer is not best-response")
            return False

    # get proposer strategy if offer is rejected (will check best-response of it below)
    proposer_rejection_strategy = r_p[f"{first_round_offer}"][1]

    # find responder's best response given proposer strategy
    # utility from first round offer
    responder_first_round_utility = first_round_offer * r_p[f"{first_round_offer}"][0]

    # utility of both agents if responder doesn't accept first round offer
    responder_second_round_utility = []
    proposer_second_round_utility = []
    for a in A:
        responder_second_round_utility.append(proposer_rejection_strategy[f"A{a}"][0]*delta*(1-a))
        proposer_second_round_utility.append(r_r[f"R{first_round_offer}{a}"][0]*delta*a)

    if max(responder_second_round_utility) == responder_first_round_utility:
        print("possibly mixed NE case - check final strategies manually")
        return False

    best_strategy_idx = -1
    if max(responder_second_round_utility) > responder_first_round_utility:
        best_strategy_idx = np.argmax(responder_second_round_utility) # set this only if first round acceptance is not best

    # check responder best response
    if best_strategy_idx == -1: # first round best, need (approx) pure acceptance
        if np.abs(1-r_r[f"A{first_round_offer}"][0]) < eps:
            print(f"approx pure first round acceptance at {first_round_offer}")
            return True
        else:
            print("responder approx pure acceptance is not best response")
            return False
    else: # second round best, need (approx) pure offer at best_strategy_idx (todo - what if not unique?)
        if np.abs(1-r_r[f"R{first_round_offer}{A[best_strategy_idx]}"][0]) < eps and np.abs(1-proposer_rejection_strategy[f"A{A[best_strategy_idx]}"][0]) <eps:
            print(f"responder and proposer both best-responding in second round offer of {A[best_strategy_idx]}")
            return True
        else:
            print("check manually for other kinds of NE")
            return False

            

            






    
def generate_initial_points(beta_p, beta_r, A):

    def random_probability_mass_split(mass, n):
        # splits the mass into n pieces randomly (for initialization) - returns vector of size n

        values = []
        remaining_mass = mass

        for i in range(n):
            if i == n-1: # ensure it adds up to mass
                values.append(remaining_mass)
            else:
                mass_value = remaining_mass * np.random.random()
                values.append(mass_value)
                remaining_mass = remaining_mass - mass_value
        
        return values

    # proposer
    mass_split_values = random_probability_mass_split(1, len(A)) # initial split

    for i,a in enumerate(A):
        beta_p[f"{a}"][0] = mass_split_values[i] 
        for j,b in enumerate(A):
            a_mass_split_values = random_probability_mass_split(mass_split_values[i],2) # split for each second round accept / reject extension of a first round offer 
            beta_p[f"{a}"][1][f"A{b}"][0] = a_mass_split_values[0]
            beta_p[f"{a}"][1][f"R{b}"][0] = a_mass_split_values[1]

            # debugging
            if beta_p[f"{a}"][1][f"R{b}"][1] != "-" or beta_p[f"{a}"][1][f"A{b}"][1] != "-" :
                print("ERROR")
                print(beta_p[f"{a}"][1][f"R{b}"])
                print(beta_p[f"{a}"][1][f"A{b}"])
                exit()
    
    # responder
    for i,a in enumerate(A):
        mass_split_values = random_probability_mass_split(1, len(A)+1)
        beta_r[f"A{a}"][0] = mass_split_values[0] # acceptance mass
        for j,b in enumerate(A):
            beta_r[f"R{a}{b}"][0] = mass_split_values[j+1]

        # debugging
        if beta_r[f"A{a}"][1] != "-" or beta_r[f"R{a}{b}"][1] != "-" :
                print("ERROR")
                print(beta_r[f"A{a}"])
                print(beta_r[f"R{a}{b}"])
                exit()
    
    return beta_p, beta_r



if __name__ == "__main__":
    intitial_strategies = []
    final_strategies = []
    T = 1000 # time steps
    delta = 0.9 # time discount factor
    M = 0.5
    D = 5

    A = [i/D for i in range(1,D+1)] # action list for offers

    for n in range(10):

        # initial strategies as realization plans now (or sequence-form polytopes)
        beta_p_zeroed = {f"{a}" : [0,dict({f"A{b}": [0,"-"] for b in A}.items(), **{f"R{b}":[0,"-"] for b in A})] for a in A} # code is float(offer)A/Rfloat(responder offer)
        # proposer constraints: beta_p[i][0] == sum([beta_p[i][1][j][0] for j in A]) for each i in A, sum([beta_p[i][0] for i in A]) == 1, all masses >= 0
        beta_r_zeroed = dict({f"A{a}":[0,"-"] for a in A}, **{f"R{a}{b}":[0,"-"] for a in A for b in A}) # code is A/Rfloat(proposer offer)float(responder offer)
        # responder constraints: beta_r[i][0] + sum([beta_r[ij][0] for j in A]) == 1 for  each i in A

        beta_p, beta_r = generate_initial_points(beta_p_zeroed, beta_r_zeroed, A)


        # beta_p = [1 if i == beta_p_idx else 0 for i in range(len(S_p))]
        # beta_r = [1 if i == beta_r_idx else 0 for i in range(len(S_r))]

        prev_p = [beta_p]
        prev_r = [beta_r]

        for t in tqdm.tqdm(range(int(T))):

            r_p_t_p_1 = agent_update(prev_r,A=A,delta=delta, M=M)
            r_r_t_p_1 = agent_update(prev_p,A=A,delta=delta, M=M, responder=True)

            # print(r_p_t_p_1)

            # print(f"r_f_t_p_1 support: {get_support(r_f_t_p_1, S_p)}")
            # print(f"r_c_t_p_1 support: {get_support(r_c_t_p_1, S_r)}")

            prev_p.append(r_p_t_p_1)
            prev_r.append(r_r_t_p_1)

            # if check_NE(r_f_t_p_1,r_c_t_p_1,S_p,delta):
            #     break

        print(f"RUN {n}/10 RESULTS")
        
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

        NE_check = check_pure_NE(r_p_t_p_1,r_r_t_p_1,A,delta)
        print(f"NE: {NE_check}")
        print("--------------------------")
        print()

    # intitial_strategies.append((beta_p,beta_r))
    # final_strategies.append((r_f_t_p_1, r_c_t_p_1))

    # f_final_1, f_final_2 = get_support(r_f_t_p_1,S_p)[0]
    # c_final_1, c_final_2 = get_support(r_c_t_p_1,S_r)[0]

    # if NE_check == False:
    #     c=-1
    # elif f_final_1 == c_final_1:
    #     c = f_final_1
    # else:
    #     c = delta * (1-c_final_2)

    # plotting initial vs. final conditions
    # P_first_round_initial = []
    # R_first_round_initial = []
    # P_second_round_initial = []
    # R_second_round_initial = []
    # P_first_round_final = []
    # R_first_round_final = []
    # P_second_round_final = []
    # R_second_round_final = []
    # P_alpha_first_round_initial = []
    # R_alpha_first_round_initial = []
    # P_alpha_second_round_initial = []
    # R_alpha_second_round_initial = []
    # proposer_payoffs = []

    # for initial, final, alpha in zip(intitial_strategies, final_strategies, alpha_points):
    #     R_first_round_initial.append(get_support(initial[1],S_r)[0][0])
    #     P_first_round_initial.append(get_support(initial[0],S_p)[0][0])
    #     R_alpha_first_round_initial.append(get_support(alpha[1],S_r)[0][0])
    #     P_alpha_first_round_initial.append(get_support(alpha[0],S_p)[0][0])

    #     R_second_round_initial.append(get_support(initial[1],S_r)[0][1])
    #     P_second_round_initial.append(get_support(initial[0],S_p)[0][1])
    #     R_alpha_second_round_initial.append(get_support(alpha[1],S_r)[0][1])
    #     P_alpha_second_round_initial.append(get_support(alpha[0],S_p)[0][1])

    #     R_first_round_final.append(get_support(final[1],S_r)[0][0])
    #     P_first_round_final.append(get_support(final[0],S_p)[0][0])

    #     R_second_round_final.append(get_support(final[1],S_r)[0][1])
    #     P_second_round_final.append(get_support(final[0],S_p)[0][1])



        # if check_NE(final[0],final[1],S_r,delta) == False:
        #     c=-1
        # elif P_first_round_final[-1] == R_first_round_final[-1]:
        #     c = 1-P_first_round_final[-1]
        # else:
        #     c = delta * R_second_round_final[-1]

        # proposer_payoffs.append(c)

    # computing average payoffs
    

    # colors_reshaped = np.zeros((5,5,5,5))
    # for n_f in range(D):
    #     for n_c in range(D):
    #         colors_reshaped[n_f][n_c] = colors[n_f*D + n_c]
        
    # # plotting
    # # color = plt.cm.rainbow(np.linspace(0, 1, N))
    # fig, ax = plt.subplots()
    # fig.suptitle("Initial strategies vs. average responder payoff value at NE")
    # values=ax.imshow(average_payoffs)
    # ax.set_xticks(np.arange(len(indices)), labels=[o for o in set(R_second_round_initial)])
    # ax.set_yticks(np.arange(len(indices)), labels=[at for at in set(R_first_round_initial)])

    # # ax2.imshow(colors_reshaped)
    # # ax2.set_xticks(np.arange(N), labels=[i/D for i in range(N)])
    # # ax2.set_yticks(np.arange(N), labels=[(N-i)/D for i in range(N)])

    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #      rotation_mode="anchor")

    # # plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
    # #      rotation_mode="anchor")

    # for i in range(len(indices)):
    #     for j in range(len(indices)):
    #         if round(average_payoffs[i, j],3) == -1:
    #             text = ax.text(j, i, "N/A",
    #                     ha="center", va="center", color="w")
    #         else:
    #             continue

    # fig.colorbar(values,ax=ax)
    #         # text = ax2.text(j, i, round(colors_reshaped[i, j],3),
    #         #             ha="center", va="center", color="w")
    # # for i, c in enumerate(color):
    # #     #first round
        
    #     # ax1.scatter(P_first_round_initial[i], R_first_round_initial[i], color=c, label= "Initial Strategies")
    #     # ax1.scatter(P_alpha_first_round_initial[i], R_alpha_first_round_initial[i], color=c, marker="+", label = "Reference Points")
    #     # ax1.scatter(P_first_round_final[i], R_first_round_final[i], color=c,marker = "x", label="Final Strategies")

    #     #second round
    #     # ax2.scatter(P_second_round_initial[i], R_second_round_initial[i], color=c, label= "Initial Strategies")
    #     # ax2.scatter(P_alpha_second_round_initial[i], R_alpha_second_round_initial[i], color=c, marker="+", label = "Reference Points")
    #     # ax2.scatter(P_second_round_final[i], R_second_round_final[i], color=c, marker="x", label="Final Strategies")
    
    # # ax1.plot(np.arange(0,1,0.1), np.arange(0,1,0.1), "k--") # agreement reference line
    # # ax.title.set_text("First round")
    # # ax1.legend()
    # ax.set_ylabel("Responder First Round Strategy - Acceptance Threshold Values")
    # ax.set_xlabel("Responder Second Round Strategy - Offer Values")
    # # ax1.set_xlim((0,1))
    # # ax1.set_ylim((0,1))

    # # ax2.plot(np.arange(0,1,0.1), np.arange(0,1,0.1), "k--") # agreement reference line
    # # ax2.title.set_text("Second round")
    # # # ax2.legend()
    # # ax2.set_xlabel("Proposer Strategy Values")
    # # # ax2.set_ylabel("Responder Strategy Values")
    # # # ax2.set_xlim((0,1))
    # # # ax2.set_ylim((0,1))

    # plt.show()

