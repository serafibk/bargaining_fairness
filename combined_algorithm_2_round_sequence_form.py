import cvxpy as cp
import numpy as np
import itertools
import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation

# NO REFERENCE POINT CURRENTLY


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
        
        for i,a in enumerate(A):
            print(f"first round offer:{a}, total cumulative utility: {utility_feedback_vector[i] +sum([utility_feedback_vector[len(A)*(i+1)+j] for j in range(len(A))]) }")

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
        r_t_p_1 = {f"{a:.2f}" : [mass_values[i],dict({f"A{b:.2f}": [mass_values[(i+1)*len(A)+j]] for j,b in enumerate(A)}.items(), **{f"R{b}":[mass_values[(i+1)*len(A)+len(A)**2+j]] for j,b in enumerate(A)})] for i,a in enumerate(A)}
        print("max first offer mass:",A[np.argmax([r_t_p_1[f"{a:.2f}"][0] for a in A])])
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

    # check best-response of this first round offer -- todo need to get total utility in each branch, not just acceptance
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
        responder_second_round_utility.append((proposer_rejection_strategy[f"A{a}"][0])/(r_p[f"{first_round_offer}"][0])*delta*(1-a))
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

            


    
def generate_initial_points(beta_p, beta_r, A, pure = False):

    def random_probability_mass_split(mass, n):
        # splits the mass into n pieces randomly (for initialization) - returns vector of size n
        values = []
        remaining_mass = mass

        for i in range(n-1):
            mass_value = remaining_mass * np.random.random()#/ n 
            values.append(mass_value)
            remaining_mass = remaining_mass - mass_value

        values.append(remaining_mass) # ensure it adds up to mass
        
        return values
    if pure == True: # randomly choose a pure strategy
        # proposer 
        first_offer_idx = np.random.randint(len(A))
        beta_p[f"{A[first_offer_idx]}"][0] = 1
        for b in A:
            second_response_idx = np.random.randint(2)
            if second_response_idx == 0:
                beta_p[f"{A[first_offer_idx]}"][1][f"A{b}"][0] = 1
            else:
                beta_p[f"{A[first_offer_idx]}"][1][f"R{b}"][0] = 1
        
        # responder
        for a in A:
            response_idx = np.random.randint(len(A)+1)
            if response_idx == 0:
                beta_r[f"A{a}"][0] = 1
            else:
                beta_r[f"R{a}{A[response_idx-1]}"][0]=1
    else: # randomly choose any strategy
        # proposer
        mass_split_values = random_probability_mass_split(1, len(A)) # initial split

        for i,a in enumerate(A):
            beta_p[f"{a}"][0] = mass_split_values[i] # mass for each first round offer
            for j,b in enumerate(A):
                a_mass_split_values = random_probability_mass_split(mass_split_values[i],2) # split for each second round accept / reject extension of a first round offer 
                beta_p[f"{a}"][1][f"A{b}"][0] = a_mass_split_values[0] # mass for accept for each possible second round offer proposer sees
                beta_p[f"{a}"][1][f"R{b}"][0] = a_mass_split_values[1] # mass for reject for each possible second round offer proposer sees

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
                beta_r[f"R{a}{b}"][0] = mass_split_values[j+1] # mass for each reject + new offer

            # debugging
            if beta_r[f"A{a}"][1] != "-" or beta_r[f"R{a}{b}"][1] != "-" :
                    print("ERROR")
                    print(beta_r[f"A{a}"])
                    print(beta_r[f"R{a}{b}"])
                    exit()
    
    return beta_p, beta_r

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



if __name__ == "__main__":
    intitial_strategies = []
    final_strategies = []
    T = 5000 # time steps
    delta = 0.9 # time discount factor
    M = 0.5 # learning rate
    D = 4 # discretization constant

    A = [round(i/D,2) for i in range(D+1)] # action list for offers {0,1/D, ... , 1}


    for n in range(4,5): # try 20 random initializations

        # initial strategies as realization plans now -- note that '-' could be used later on for recursive setting for n rounds of bargaining 
        # beta_p_zeroed = {f"{a}" : [0,dict({f"A{b}": [0,"-"] for b in A}.items(), **{f"R{b}":[0,"-"] for b in A})] for a in A} # code is float(offer)A/Rfloat(responder offer)
        # beta_r_zeroed = dict({f"A{a}":[0,"-"] for a in A}, **{f"R{a}{b}":[0,"-"] for a in A for b in A}) # code is A/Rfloat(proposer offer)float(responder offer)

        # generate possible initial realization plan from sequence-form polytope corresponding to action set A, Q(A)
        # beta_p, beta_r = generate_initial_points(beta_p_zeroed, beta_r_zeroed, A, pure=False)
        # beta_p= {'0.2': [0, {'A0.2': [0, '-'], 'A0.4': [0, '-'], 'A0.6': [0, '-'], 'A0.8': [0, '-'], 'A1.0': [0, '-'], 'R0.2': [0, '-'], 'R0.4': [0, '-'], 'R0.6': [0, '-'], 'R0.8': [0, '-'], 'R1.0': [0, '-']}], '0.4': [0, {'A0.2': [0, '-'], 'A0.4': [0, '-'], 'A0.6': [0, '-'], 'A0.8': [0, '-'], 'A1.0': [0, '-'], 'R0.2': [0, '-'], 'R0.4': [0, '-'], 'R0.6': [0, '-'], 'R0.8': [0, '-'], 'R1.0': [0, '-']}], '0.6': [1, {'A0.2': [0, '-'], 'A0.4': [1, '-'], 'A0.6': [1, '-'], 'A0.8': [0, '-'], 'A1.0': [0, '-'], 'R0.2': [1, '-'], 'R0.4': [0, '-'], 'R0.6': [0, '-'], 'R0.8': [1, '-'], 'R1.0': [1, '-']}], '0.8': [0, {'A0.2': [0, '-'], 'A0.4': [0, '-'], 'A0.6': [0, '-'], 'A0.8': [0, '-'], 'A1.0': [0, '-'], 'R0.2': [0, '-'], 'R0.4': [0, '-'], 'R0.6': [0, '-'], 'R0.8': [0, '-'], 'R1.0': [0, '-']}], '1.0': [0, {'A0.2': [0, '-'], 'A0.4': [0, '-'], 'A0.6': [0, '-'], 'A0.8': [0, '-'], 'A1.0': [0, '-'], 'R0.2': [0, '-'], 'R0.4': [0, '-'], 'R0.6': [0, '-'], 'R0.8': [0, '-'], 'R1.0': [0, '-']}]}
        # beta_r= {'A0.2': [0, '-'], 'A0.4': [0, '-'], 'A0.6': [0, '-'], 'A0.8': [0, '-'], 'A1.0': [0, '-'], 'R0.20.2': [0, '-'], 'R0.20.4': [0, '-'], 'R0.20.6': [0, '-'], 'R0.20.8': [1, '-'], 'R0.21.0': [0, '-'], 'R0.40.2': [0, '-'], 'R0.40.4': [1, '-'], 'R0.40.6': [0, '-'], 'R0.40.8': [0, '-'], 'R0.41.0': [0, '-'], 'R0.60.2': [0, '-'], 'R0.60.4': [0, '-'], 'R0.60.6': [1, '-'], 'R0.60.8': [0, '-'], 'R0.61.0': [0, '-'], 'R0.80.2': [0, '-'], 'R0.80.4': [0, '-'], 'R0.80.6': [1, '-'], 'R0.80.8': [0, '-'], 'R0.81.0': [0, '-'], 'R1.00.2': [0, '-'], 'R1.00.4': [0, '-'], 'R1.00.6': [0, '-'], 'R1.00.8': [1, '-'], 'R1.01.0': [0, '-']}

        beta_p, beta_r = generate_new_betas(D)
        y_index = 3 * (D+1) + 3
        x_index = 3 * (D+1) + 3
        
        # generate pure strategy
        # proposer 
        first_offer_idx = 3 
        second_response_idx = 3
        beta_p[f"{A[first_offer_idx]:.2f}"][0] = 1
        for b in A:
            if b <= A[second_response_idx]:
                beta_p[f"{A[first_offer_idx]:.2f}"][1][f"A{b:.2f}"][0] = 1
            else:
                beta_p[f"{A[first_offer_idx]:.2f}"][1][f"R{b:.2f}"][0] = 1
        # responder
        for a in A:
            if a >= A[3]:
                beta_r[f"A{a:.2f}"][0] = 1
            else:
                beta_r[f"R{a:.2f}{A[3]:.2f}"][0]=1


        prev_p = [beta_p]
        prev_r = [beta_r]

        # # initial check
        # for i,a in enumerate(A):
        #         counter =0
        #         for b in A[1:]:
        #             if beta_p[f"{a}"][1][f"A{A[0]}"][0] > ((1-b)/(1-A[0]))*beta_p[f"{a}"][1][f"A{b}"][0]:
        #                 counter = counter + 1
        #         if counter == len(A)-1:
        #             print(f"Condition met for first round offer {a} in initial conditions.")

        # accumulation_condition = [0 for a in A]

        proposer_mass = []
        responder_mass = []

        for t in tqdm.tqdm(range(int(T))):

            r_p_t_p_1 = agent_update(prev_r,A=A,delta=delta, M=M)
            r_r_t_p_1 = agent_update(prev_p,A=A,delta=delta, M=M, responder=True)
            # checking for responder acumulation condition
            # for i,a in enumerate(A):
            #     if accumulation_condition[i] == 1:
            #         continue 
            #     counter =0
            #     for b in A[1:]:
            #         if r_p_t_p_1[f"{a}"][1][f"A{A[0]}"][0] > ((1-b)/(1-A[0]))*r_p_t_p_1[f"{a}"][1][f"A{b}"][0]:
            #             # print(r_p_t_p_1[f"{a}"][1][f"A{A[0]}"][0])
            #             # print(r_p_t_p_1[f"{a}"][1][f"A{b}"][0])
            #             counter = counter + 1
            #     if counter == len(A)-1:
            #         print(f"Condition met for first round offer {a} at time step {t}.")
            #         accumulation_condition[i] = 1

            if t > 4950:
                print(f"0.0 - 0.25: ",r_p_t_p_1["0.00"][0]-r_p_t_p_1["0.25"][0])
                print(f"0.25 - 0.5: ", r_p_t_p_1["0.25"][0]-r_p_t_p_1["0.50"][0])
                print(f"sum of gaps: ", np.abs(r_p_t_p_1["0.00"][0]-r_p_t_p_1["0.25"][0] )+np.abs(r_p_t_p_1["0.25"][0]-r_p_t_p_1["0.50"][0]))
            # else:
            #     exit()
            prev_p.append(r_p_t_p_1)
            prev_r.append(r_r_t_p_1)

            # mass_max_a_idx = np.argmax([r_p_t_p_1[f"{a}"][0] for a in A])
            # max_mass_a = 0.6#A[mass_max_a_idx]

            # responder_mass_t = [r_r_t_p_1[f"R{max_mass_a}{a}"][0] for a in A]
            # responder_mass_t.append(r_r_t_p_1[f"A{max_mass_a}"][0])
            # responder_mass.append(responder_mass_t)


            # proposer_mass_t = [r_p_t_p_1[f"{0.6}"][1][f"A{a}"][0] for a in A]
            # proposer_mass.append(proposer_mass_t)

            # if check_NE(r_f_t_p_1,r_c_t_p_1,S_p,delta):
            #     break

        print(f"RUN {n+1}/10 RESULTS")
        
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
        
        # todo plot if we find set up that converges to different NE
        time_start = 1
        time_end = T

        #proposer
        # all_f_cdfs = []   
        # for w_f in prev_f[time_start:time_end]:
        #     cdf_at_t = [1-get_cdf(w_f,i) for i in range(len(w_f))]
        #     all_f_cdfs.append(cdf_at_t)

        # fig, ax = plt.subplots(1, 1, figsize = (6, 6))
        # def animate(t):
        #     ax.cla() # clear the previous image
        #     # ax.set_title(f"Proposer initial={get_support(beta_f,S_f)}, Responder initial={get_support(beta_c,S_c)}, M={M}, T={T}")
        #     # ax.plot(S_c,all_cdfs[t], label="Responder",color="blue")
        #     ax.scatter(A,proposer_mass[t], label="Proposer",color="blue")
        #     ax.scatter(A+[0], responder_mass[t], label="Responder",color="red") # 0 == Accept condition
        #     # ax.scatter([proposer_final_most_mass], [1], label=f"NE point: {proposer_final_most_mass}", color="black")
        #     ax.legend()

        # anim = animation.FuncAnimation(fig, animate, frames = time_end-time_start, interval = 5, blit = False)
        # anim.save(f'n={n}_responder_max_mass_response_just_at_0.6_plus_proposer.gif', writer='Pillow', fps=30)
        # plt.show()

