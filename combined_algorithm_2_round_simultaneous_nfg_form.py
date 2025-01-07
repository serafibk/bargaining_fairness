import cvxpy as cp
import numpy as np
import itertools
import tqdm
import matplotlib.pyplot as plt


def agent_update(prev_not_i_strategies, S_i, alpha_i, delta, M, responder= False):

    utility_feedback_vector = [0.0 for s in S_i]


    if responder == True: # candidate
        for w_f in prev_not_i_strategies:
            for j, s_c in enumerate(S_i):
                for k, w_f_supp in enumerate(S_i):
                    if w_f_supp[0] >= s_c[0]:
                        utility_feedback_vector[j] += w_f[k] *  w_f_supp[0]
                    elif s_c[1] >= w_f_supp[1]:
                        utility_feedback_vector[j] += w_f[k] * delta *(1-s_c[1])
    else: # firm
        for w_c in prev_not_i_strategies:
            for j, s_f in enumerate(S_i):
                for k, w_c_supp in enumerate(S_i):
                    if w_c_supp[0] <= s_f[0]:
                        utility_feedback_vector[j] += w_c[k] *  (1-s_f[0])
                    elif s_f[1] <= w_c_supp[1]:
                        utility_feedback_vector[j] += w_c[k] * delta *(w_c_supp[1])
    

    w_var = cp.Variable(len(S_i))
    objective = cp.Maximize(w_var@utility_feedback_vector - cp.norm(w_var, 2)/(2*M))
    constraints = [cp.sum(w_var)==1, cp.min(w_var)>=0]
    problem = cp.Problem(objective,constraints)

    problem.solve()

    w_t_p_1_all = []
    for w_p in w_var:
        probability = max(0.0,round(w_p.value,8))
        w_t_p_1_all.append(probability)
    
    # keep largest non-zero support
    # largest = 0
    # for i, w_supp in enumerate(w_t_p_1_all):
    #     if w_supp > 0:
    #         # print(S_i[i])
    #         largest = i

    # w_t_p_1 = [1 if i == largest else 0 for i in range(len(w_t_p_1_all))]    
    return w_t_p_1_all



def get_support(w_i,S_i):

    # print("----non-zero support----")
    non_zero_support = []
    for i,w in enumerate(w_i):
        if w>0:
            non_zero_support.append(S_i[i])

    
    return non_zero_support

    

def check_NE(w_f,w_c, S,delta):

    if len(get_support(w_f,S)) >1 or len(get_support(w_c,S)) >1:
        return False
    
    w_f_1 = get_support(w_f,S)[0][0]
    w_f_2 = get_support(w_f,S)[0][1]

    w_c_1 = get_support(w_c,S)[0][0]
    w_c_2 = get_support(w_c,S)[0][1]

    if w_f_1 == w_c_1:
        if delta * (w_c_2) <= 1-w_f_1 and delta* (1-w_f_2) <= w_f_1:
            return True
        else:
            return False
    elif w_f_2 == w_c_2:
        if delta * (w_c_2) >= 1-w_c_1 and delta* (1-w_c_2) >= w_f_1:
            return True
        else:
            return False
    else:
        return False





if __name__ == "__main__":
    intitial_strategies = []
    final_strategies = []
    alpha_points = []
    sample_points = np.arange(0.04,1,0.04, dtype=float)
    sample_strategies = [(float(round(s,2)), float(round(sample_points[len(sample_points)-1-i],2))) for i,s in enumerate(sample_points)]
    T = 100 # time steps
    delta = 0.9 # time discount factor
    M = 0.5
    D = 16

    indices = [1,3,5]

    # R_f = 8 # fixed first round or second round index
    # R_c = 3 # fixed first round or second round index
    S_f = list(itertools.product([i/D for i in range(1,D+1)],[i/D for i in range(1,D+1)]))
    S_c = list(itertools.product([i/D for i in range(1,D+1)],[i/D for i in range(1,D+1)]))

    def choose_supp_idx(s, S_i):
        return S_i.index(s)

    average_payoffs = np.zeros((len(indices),len(indices)))

    for i1, n_f1 in enumerate(indices): # all the firm's first round strategies
        for i2, n_f2 in enumerate(indices):# all the firm's second round strategies
            payoffs = []
            for j1, n_c1 in enumerate(indices): # all the candidate's first round strategies
                for j2, n_c2 in enumerate(indices): # all the candidate's second round strategies

                    print(f"candidate strategy indices {(n_f1, n_f2)}, firm strategy indices {(n_c1, n_c2)}")
                
                    beta_c_idx = n_f1*D + n_f2 #np.random.randint(len(S_f))#choose_supp_idx((0.5, 0.7), S_f)#choose_supp_idx(sample_strategies[n],S_f)#
                    beta_f_idx = n_c1*D + n_c2 #np.random.randint(len(S_c))#choose_supp_idx(sample_strategies[N-1-n],S_c)#np.random.randint(len(S_c))
                    alpha_f_idx = 21#np.random.randint(len(S_f))# choose_supp_idx(sample_strategies[np.random.randint(N)],S_f)#np.random.randint(len(S_f))
                    alpha_c_idx = 93#np.random.randint(len(S_c)) #choose_supp_idx(sample_strategies[np.random.randint(N)],S_c)#np.random.randint(len(S_c))

                    beta_f = [1 if i == beta_f_idx else 0 for i in range(len(S_f))]
                    beta_c = [1 if i == beta_c_idx else 0 for i in range(len(S_c))]
                    alpha_f = [1 if i == alpha_f_idx else 0 for i in range(len(S_f))]
                    alpha_c = [1 if i == alpha_c_idx else 0 for i in range(len(S_c))]

                    prev_f = [beta_f]
                    prev_c = [beta_c]

                    for t in tqdm.tqdm(range(int(T))):

                        w_f_t_p_1 = agent_update(prev_c,S_i=S_f,alpha_i=alpha_f,delta=delta, M=M)
                        w_c_t_p_1 = agent_update(prev_f,S_i=S_c,alpha_i=alpha_c,delta=delta, M=M, responder=True)

                        print(f"w_f_t_p_1 support: {get_support(w_f_t_p_1, S_f)}")
                        print(f"w_c_t_p_1 support: {get_support(w_c_t_p_1, S_c)}")
                        # print(w_f_t_p_1)
                        # print(w_c_t_p_1)
                        # break
                        # if max(w_f_t_p_1) < 0.99999 or max(w_c_t_p_1) < 0.99999: # my check for justification of rounding to avoid drift from pure
                        #     print("FOUND")
                        #     print(w_f_t_p_1)
                        #     print(w_c_t_p_1)

                        prev_f.append(w_f_t_p_1)
                        prev_c.append(w_c_t_p_1)

                        if check_NE(w_f_t_p_1,w_c_t_p_1,S_f,delta):
                            break
                    
                    print("----initial parameters----")
                    print(f"beta_f: {get_support(beta_f,S_f)}")
                    print(f"beta_c: {get_support(beta_c,S_c)}")
                    print(f"alpha_f: {get_support(alpha_f,S_f)}")
                    print(f"alpha_c: { get_support(alpha_c,S_c)}")
                

                    print("----final convergence----")
                    # print(f"w_f_T: {w_f_t_p_1}")
                    print("--non-zero support--")
                    print(get_support(w_f_t_p_1, S_f))
                    # print(f"w_c_T: {w_c_t_p_1}")
                    print("--non-zero support--")
                    print(get_support(w_c_t_p_1, S_c))

                    NE_check = check_NE(w_f_t_p_1,w_c_t_p_1,S_f,delta)

                    print(f"NE: {NE_check}")

                    intitial_strategies.append((beta_f,beta_c))
                    final_strategies.append((w_f_t_p_1, w_c_t_p_1))
                    alpha_points.append((alpha_f,alpha_c))

                    f_final_1, f_final_2 = get_support(w_f_t_p_1,S_f)[0]
                    c_final_1, c_final_2 = get_support(w_c_t_p_1,S_c)[0]

                    if NE_check == False:
                        c=-1
                    elif f_final_1 == c_final_1:
                        c = f_final_1
                    else:
                        c = delta * (1-c_final_2)

                    payoffs.append(c)
            average_payoffs[i1][i2] = sum(payoffs) / len(payoffs)

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
    #     R_first_round_initial.append(get_support(initial[1],S_c)[0][0])
    #     P_first_round_initial.append(get_support(initial[0],S_f)[0][0])
    #     R_alpha_first_round_initial.append(get_support(alpha[1],S_c)[0][0])
    #     P_alpha_first_round_initial.append(get_support(alpha[0],S_f)[0][0])

    #     R_second_round_initial.append(get_support(initial[1],S_c)[0][1])
    #     P_second_round_initial.append(get_support(initial[0],S_f)[0][1])
    #     R_alpha_second_round_initial.append(get_support(alpha[1],S_c)[0][1])
    #     P_alpha_second_round_initial.append(get_support(alpha[0],S_f)[0][1])

    #     R_first_round_final.append(get_support(final[1],S_c)[0][0])
    #     P_first_round_final.append(get_support(final[0],S_f)[0][0])

    #     R_second_round_final.append(get_support(final[1],S_c)[0][1])
    #     P_second_round_final.append(get_support(final[0],S_f)[0][1])



        # if check_NE(final[0],final[1],S_c,delta) == False:
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

