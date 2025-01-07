import cvxpy as cp
import numpy as np
import itertools
import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation


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
        probability = max(0.0, round(w_p.value,9))
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

    # if 1 not in w_f:
    #     print("here 1")
    #     return False


    if np.argmax(get_support(w_c,S)) != np.argmax(w_f):
        print("here 2")
        return False
    
    utilities = []
    for i in range(len(S)):
        if i <= np.argmax(w_f):
            utilities.append(sum([w_c[j] for j in range(i+1)])*(1-S[i]))
    
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

    

def get_cdf(w,idx): # sum probability mass up until and including idx

    return sum([w[i] for i in range(idx+1)])





if __name__ == "__main__":

    T =500 # time steps
    M = 0.5#T**(1/4) # regularizer constant
    D = 100

    N = 2

    S_f = [i/D for i in range(1,D)]
    S_c = [i/D for i in range(1,D)]

    for n in range(N):

        beta_f_idx = np.random.randint(len(S_f))
        beta_c_idx = np.random.randint(len(S_c))
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
            # if t < 2:
            #     print(w_c_t_p_1)
            #     print(f"w_c_t_p_1 support: {get_support(w_c_t_p_1, S_c)}")
            #     print(w_f_t_p_1)
            #     print(f"w_f_t_p_1 support: {get_support(w_f_t_p_1, S_f)}")
            # else:
            #     break
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

        # w_c_t_p_1 = [np.float64(0.07703922429802273), np.float64(0.07703922429802272), np.float64(0.07698215627273045), np.float64(0.07691660819823715), np.float64(0.07689143433146955), np.float64(0.07689143318836383), np.float64(0.07689143184548435), np.float64(0.07689143007740834), np.float64(0.07689142766327332), np.float64(0.07689142424489294), np.float64(0.07689141910476761), np.float64(0.07689140973033365), np.float64(0.07689137781497941), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)]
        # w_f_t_p_1 = [np.float64(9.491926414417362e-11), np.float64(1.0315089809815776e-10), np.float64(1.1249598798160589e-10), np.float64(1.2331513028795063e-10), np.float64(1.3612064655073943e-10), np.float64(1.5162059862379025e-10), np.float64(1.7083772986024806e-10), np.float64(1.9553628976589555e-10), np.float64(2.2989593011453287e-10), np.float64(2.878814186817747e-10), np.float64(4.3420966673719127e-10), np.float64(1.210868151586972e-09), np.float64(0.9999999856927584), np.float64(3.5095726150451397e-09), np.float64(1.205034827310276e-09), np.float64(6.510527076791693e-10), np.float64(4.5160500799310025e-10), np.float64(3.575431753592871e-10), np.float64(3.04547465577603e-10), np.float64(2.705683134042068e-10), np.float64(2.465207953506935e-10), np.float64(2.2818618328099668e-10), np.float64(2.1342527735133304e-10), np.float64(2.0107286183640352e-10), np.float64(1.9045216475985665e-10), np.float64(1.8706475394553033e-10), np.float64(1.7792923833050114e-10), np.float64(1.6980569539527661e-10), np.float64(1.6251357160631234e-10), np.float64(1.559196813774348e-10), np.float64(1.4992145694652726e-10), np.float64(1.444371591931847e-10), np.float64(1.3939993058962654e-10), np.float64(1.3475402939571746e-10), np.float64(1.3045233785346668e-10), np.float64(1.2645463724901362e-10), np.float64(1.2272636007935647e-10), np.float64(1.1923765007811648e-10), np.float64(1.1596262891795884e-10), np.float64(1.1287880752245476e-10), np.float64(1.0996660278897348e-10), np.float64(1.0720893413618866e-10), np.float64(1.0459088255231593e-10), np.float64(1.020993999495645e-10), np.float64(9.972305989917838e-11), np.float64(9.745184296782184e-11), np.float64(9.527695133613351e-11), np.float64(9.319064840971106e-11), np.float64(9.118611988811622e-11), np.float64(8.925735333134053e-11), np.float64(8.739903371382946e-11)]

        print(check_mixed_NE(w_c_t_p_1,w_f_t_p_1, S_f))

        # plotting cdf of each agent over time
        time_start = 1
        time_end = T
        # responder
        all_cdfs = []
        for w_c in prev_c[time_start:time_end]:
            cdf_at_t = [get_cdf(w_c,i) for i in range(len(w_c))]
            all_cdfs.append(cdf_at_t)

        # for t in range(len(prev_c[time_start:time_end])):
        #     plt.plot(S_c,all_cdfs[t], label=t,color="blue")
        
        # plt.legend()
        # plt.show()


        #proposer
        all_f_cdfs = []   
        for w_f in prev_f[time_start:time_end]:
            cdf_at_t = [1-get_cdf(w_f,i) for i in range(len(w_f))]
            all_f_cdfs.append(cdf_at_t)

        proposer_final_most_mass = S_f[np.argmax(w_f_t_p_1)]


        # for t in range(len(prev_f[time_start:time_end])):
        #     plt.plot(S_f,all_f_cdfs[t], label=t,color="red")
        fig, ax = plt.subplots(1, 1, figsize = (6, 6))
        def animate(t):
            ax.cla() # clear the previous image
            ax.set_title(f"Proposer initial={get_support(beta_f,S_f)}, Responder initial={get_support(beta_c,S_c)}, M={M}, T={T}")
            ax.plot(S_c,all_cdfs[t], label="Responder",color="blue")
            ax.plot(S_f,all_f_cdfs[t], label="Proposer",color="red")
            ax.scatter([proposer_final_most_mass], [1], label=f"NE point: {proposer_final_most_mass}", color="black")
            ax.legend()
            # ax.plot(x[:i], y[:i]) # plot the line
            # ax.set_xlim([x0, tfinal]) # fix the x axis
            # ax.set_ylim([1.1*np.min(y), 1.1*np.max(y)]) # fix the y axis

        anim = animation.FuncAnimation(fig, animate, frames = len(prev_c[time_start:time_end]), interval = 5, blit = False)
        anim.save(f'n={n}_cdf_plots_proposer_reversed.gif', writer='Pillow', fps=30)
        plt.show()


