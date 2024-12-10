import cvxpy as cp
import numpy as np
import itertools
import tqdm


def agent_update(prev_not_i_strategies, S_i, alpha_i, M, responder= False,t=0):

    utility_feedback_vector = [0.0 for s in S_i]


    if responder == True: # candidate
        for w_f in prev_not_i_strategies:
            for j, s_c in enumerate(S_i):
                for k, w_f_supp in enumerate(S_i):
                    if w_f_supp >= s_c:
                        # if s_c < get_support(prev_not_i_strategies[0], S_i)[0]: # ignore strategies below starting point
                        #     utility_feedback_vector[j] += 0
                        
                        utility_feedback_vector[j] += w_f[k] *  w_f_supp
    else: # firm
        for w_c in prev_not_i_strategies:
            for j, s_f in enumerate(S_i):
                for k, w_c_supp in enumerate(S_i):
                    if w_c_supp <= s_f:
                        utility_feedback_vector[j] += w_c[k] *  (1-s_f)
    

    w_var = cp.Variable(len(S_i))
    # if not responder and t > 370:
        # print(utility_feedback_vector)
    objective = cp.Maximize(w_var@utility_feedback_vector - cp.norm(w_var, 2)**2/(2*M))
    constraints = [cp.sum(w_var)==1, cp.min(w_var)>=0]#cp.sum_largest(w_var,8) <= 0.98]
    problem = cp.Problem(objective,constraints)

    problem.solve()

    w_t_p_1_all = []
    for i,w_p in enumerate(w_var):
        probability = max(0.0, round(w_p.value,4))
        w_t_p_1_all.append(probability)
    # print(w_t_p_1_all)
    
    # keep largest non-zero support
    # if responder == True:
    #     largest = 0
    #     for i, w_supp in enumerate(w_t_p_1_all):
    #         if w_supp >0:
    #             w_supp_prev = w_supp
    #             largest = i


        # w_t_p_1_all = [1 if i == largest else 0 for i in range(len(w_t_p_1_all))]    
    return w_t_p_1_all


def check_mixed_NE(w_c,w_f,S):

    if 1 not in w_f:
        print("here 1")
        return False


    if np.argmax(get_support(w_c,S)) != np.argmax(w_f):
        print("here 2")
        return False
    
    utilities = []
    for i in range(len(S)):
        if i <= np.argmax(w_f):
            utilities.append(sum([w_c[j] for j in range(i+1)])*(1-S[i]))
    
    print(utilities)
    if np.argmax(utilities) == np.argmax(w_f):
        return True
    else:
        print("here 3")
        return False

    



def get_support(w_i,S_i):

    # print("----non-zero support----")
    non_zero_support = []
    for i,w in enumerate(w_i):
        if w>0:
            non_zero_support.append(S_i[i])

    
    return non_zero_support

    






if __name__ == "__main__":

    T =300 # time steps
    M = 0.5#T**(1/4) # regularizer constant
    D = 100

    S_f = [i/D for i in range(1,D)]
    S_c = [i/D for i in range(1,D)]


    beta_f_idx = 50#np.random.randint(len(S_f))
    beta_c_idx = 28#np.random.randint(len(S_c))
    alpha_f_idx = 0#np.random.randint(len(S_f))
    alpha_c_idx = 0#np.random.randint(len(S_c))

    beta_f = [1 if i == beta_f_idx else 0 for i in range(len(S_f))]
    beta_c = [1 if i == beta_c_idx else 0 for i in range(len(S_c))]
    alpha_f = [1 if i == alpha_f_idx else 0 for i in range(len(S_f))]
    alpha_c = [1 if i == alpha_c_idx else 0 for i in range(len(S_c))]

    prev_f = [beta_f]
    prev_c = [beta_c]

    for t in tqdm.tqdm(range(T)):

        w_f_t_p_1 = agent_update(prev_c,S_i=S_f,alpha_i=alpha_f, M=M, t=t)
        w_c_t_p_1 = agent_update(prev_f,S_i=S_c,alpha_i=alpha_c, M=M, responder=True,t=t)
    

        # 
        if t < 10:
            print(w_c_t_p_1)
            print(f"w_c_t_p_1 support: {get_support(w_c_t_p_1, S_c)}")
            print(w_f_t_p_1)
            print(f"w_f_t_p_1 support: {get_support(w_f_t_p_1, S_f)}")
        else:
            break
        # print(f"w_f_t_p_1 support: {get_support(w_f_t_p_1, S_f)}")
        # print(f"w_c_t_p_1 support: {get_support(w_c_t_p_1, S_c)}")

        

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

    w_c_t_p_1 = [np.float64(0.0807), np.float64(0.0807), np.float64(0.0807), np.float64(0.0807), np.float64(0.0807), np.float64(0.0807), np.float64(0.0807), np.float64(0.0807), np.float64(0.0807), np.float64(0.0807), np.float64(0.0731), np.float64(0.0651), np.float64(0.0552), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w_f_t_p_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.float64(1.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    print(check_mixed_NE(w_c_t_p_1,w_f_t_p_1, S_f))


