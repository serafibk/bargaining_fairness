import cvxpy as cp
import numpy as np
import itertools
import tqdm


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
    objective = cp.Maximize(w_var@utility_feedback_vector - cp.norm(w_var-alpha_i, 1)/cp.sqrt(M))
    constraints = [cp.sum(w_var)==1, cp.min(w_var)>=0]
    problem = cp.Problem(objective,constraints)

    problem.solve()

    w_t_p_1_all = []
    for w_p in w_var:
        probability = max(0.0,round(w_p.value,5))
        w_t_p_1_all.append(probability)
    
    # keep largest non-zero support
    largest = 0
    for i, w_supp in enumerate(w_t_p_1_all):
        if w_supp > 0:
            largest = i

    w_t_p_1 = [1 if i == largest else 0 for i in range(len(w_t_p_1_all))]    
    return w_t_p_1



def get_support(w_i,S_i):

    # print("----non-zero support----")
    non_zero_support = []
    for i,w in enumerate(w_i):
        if w>0:
            non_zero_support.append(S_i[i])

    
    return non_zero_support

    






if __name__ == "__main__":

    T = 50 # time steps
    M = 1000.0 # regularizer constant
    delta = 0.95 # time discount factor

    S_f = list(itertools.product([i/T for i in range(T+1)],[i/T for i in range(T+1)]))
    S_c = list(itertools.product([i/T for i in range(T+1)],[i/T for i in range(T+1)]))

    beta_f_idx = np.random.randint(len(S_f))
    beta_c_idx = np.random.randint(len(S_c))
    alpha_f_idx = np.random.randint(len(S_f))
    alpha_c_idx = np.random.randint(len(S_c))

    beta_f = [1 if i == beta_f_idx else 0 for i in range(len(S_f))]
    beta_c = [1 if i == beta_c_idx else 0 for i in range(len(S_c))]
    alpha_f = [1 if i == alpha_f_idx else 0 for i in range(len(S_f))]
    alpha_c = [1 if i == alpha_c_idx else 0 for i in range(len(S_c))]

    prev_f = [beta_f]
    prev_c = [beta_c]

    for t in tqdm.tqdm(range(int(T/10))):

        w_f_t_p_1 = agent_update(prev_c,S_i=S_f,alpha_i=alpha_f,delta=delta, M=M)
        w_c_t_p_1 = agent_update(prev_f,S_i=S_c,alpha_i=alpha_c,delta=delta, M=M, responder=True)

        print(f"w_f_t_p_1 support: {get_support(w_f_t_p_1, S_f)}")
        print(f"w_c_t_p_1 support: {get_support(w_c_t_p_1, S_c)}")

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


