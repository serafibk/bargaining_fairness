import cvxpy as cp
import numpy as np
import itertools
import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation

def agent_update(prev_not_i_strategies, A, delta, M, responder= False):
    # updating realization probabilities now

    if responder == False: # proposer
        # create utility feedback vector
        utility_feedback_vector = [0.0 for a in range(len(A)+2*len(A)**2)] # a_i, a_iAa_j,a_iRa_j low to high ordering (should never reject any offer)

        for t,r_r in enumerate(prev_not_i_strategies): # go through record of responder strategies
            for i,a in enumerate(A):
                utility_feedback_vector[i] += (1-a) * r_r[f"A{a}"][0] # a_i accepted
                for j,b in enumerate(A):
                    if t == 0:
                        if a != 0.2 and a !=0.4:
                            if b == 0.2:
                                utility_feedback_vector[len(A)*(i+1)+j] = 10 # make sure firm starts by purely accepting worker's offer
                        else: # lower cumulative utilities
                            if a== 0.2 and b == 0.6: # worker's pure offer
                                utility_feedback_vector[len(A)*(i+1)+j] = 20 # make sure other branches do not get mass 
                            if a== 0.4 and b == 0.6: # worker's pure offer
                                utility_feedback_vector[len(A)*(i+1)+j] = 80 # make sure other branches do not get mass 
                            if a == 0.2 and b == 0.2:
                                utility_feedback_vector[len(A)*(i+1)+j] = 0.01 
                            if a == 0.2 and b == 0.4:
                                utility_feedback_vector[len(A)*(i+1)+j] = 0.6
                            if a == 0.4 and b == 0.2:
                                utility_feedback_vector[len(A)*(i+1)+j] = 0.01 
                            if a == 0.4 and b == 0.4:
                                utility_feedback_vector[len(A)*(i+1)+j] = 0.7
                    utility_feedback_vector[len(A)*(i+1)+j] += delta * b * r_r[f"R{a}{b}"][0] # accepts counter offer from responder       
        
        # for i,a in enumerate(A):
        #     print(f"first round offer:{a}, total cumulative utility: {utility_feedback_vector[i] +sum([utility_feedback_vector[len(A)*(i+1)+j] for j in range(len(A))]) }")

        # set up problem
        r_var = cp.Variable(len(A)+2*len(A)**2)
        objective = cp.Maximize(r_var@utility_feedback_vector - cp.norm(r_var, 2)**2/(2*M)) 

        # create summation constraints
        constraints = [r_var[i]>=0 for i in range(len(A)+2*len(A)**2)]
        constraints.append(cp.sum(r_var[:len(A)])==1)
        for i in range(len(A)):
            for j in range(len(A)):
                constraints.append(r_var[i] == cp.sum(r_var[(i+1)*len(A)+j]+r_var[(i+1)*len(A)+len(A)**2+j])) # mass on a_iAa_j and a_iRa_j (after responder rejects a_i) sums to mass on a_i


    else: # responder
        # create utility feedback vector
        utility_feedback_vector = [0.0 for a in range(len(A)+len(A)**2)] # Aa_i, Ra_ia_j low to high ordering for each

        for t,r_p in enumerate(prev_not_i_strategies): # go through record of proposer strategies
            for i,a in enumerate(A):
                utility_feedback_vector[i] += a * r_p[f"{a}"][0] # Aa_i, acceptance of initial offer
                for j,b in enumerate(A):
                    if t == 0:
                        if a != 0.2 and a != 0.4:
                            if b == 0.2:
                                utility_feedback_vector[len(A)*(i+1)+j] = 100 # make sure worker stays at this strategy
                        else:
                            if a == 0.2 and b == 0.2:
                                utility_feedback_vector[len(A)*(i+1)+j] = 100
                            if a == 0.2 and b == 0.4:
                                utility_feedback_vector[len(A)*(i+1)+j] = 100
                            if a == 0.2 and b == 0.6:
                                utility_feedback_vector[len(A)*(i+1)+j] = 100 + (0.99/M)
                            if a == 0.4 and b == 0.2:
                                utility_feedback_vector[len(A)*(i+1)+j] = 100
                            if a == 0.4 and b == 0.4:
                                utility_feedback_vector[len(A)*(i+1)+j] = 100 
                            if a == 0.4 and b == 0.6:
                                utility_feedback_vector[len(A)*(i+1)+j] = 100 + (0.99/M)
                    utility_feedback_vector[len(A)*(i+1)+j] += delta * (1-b) * r_p[f"{a}"][1][f"A{b}"][0] # Ra_ia_j, proposer accepts second offer
                
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
    
    if responder == False: # proposer
        r_t_p_1 = {f"{a}" : [mass_values[i],dict({f"A{b}": [mass_values[(i+1)*len(A)+j]] for j,b in enumerate(A)}.items(), **{f"R{b}":[mass_values[(i+1)*len(A)+len(A)**2+j]] for j,b in enumerate(A)})] for i,a in enumerate(A)}
    else: # responder
        r_t_p_1 = dict({f"A{a}":[mass_values[i]] for i,a in enumerate(A)}, **{f"R{a}{b}":[mass_values[len(A)*(i+1)+j]] for i,a in enumerate(A) for j,b in enumerate(A)})

    return r_t_p_1



beta_p= {'0.2': [0.2, {'A0.2': [0.105, '-'], 'A0.4': [0.2, '-'], 'A0.6': [0.2, '-'], 'A0.8': [0.1, '-'], 'A1.0': [0.1, '-'], 'R0.2': [0.095, '-'], 'R0.4': [0, '-'], 'R0.6': [0, '-'], 'R0.8': [0.1, '-'], 'R1.0': [0.1, '-']}], '0.4': [0.8, {'A0.2': [0.405, '-'], 'A0.4': [0.75, '-'], 'A0.6': [0.8, '-'], 'A0.8': [0.4, '-'], 'A1.0': [0.4, '-'], 'R0.2': [0.395, '-'], 'R0.4': [0.05, '-'], 'R0.6': [0, '-'], 'R0.8': [0.4, '-'], 'R1.0': [0.4, '-']}], '0.6': [0, {'A0.2': [0, '-'], 'A0.4': [0, '-'], 'A0.6': [0, '-'], 'A0.8': [0, '-'], 'A1.0': [0, '-'], 'R0.2': [0, '-'], 'R0.4': [0, '-'], 'R0.6': [0, '-'], 'R0.8': [0, '-'], 'R1.0': [0, '-']}], '0.8': [0, {'A0.2': [0, '-'], 'A0.4': [0, '-'], 'A0.6': [0, '-'], 'A0.8': [0, '-'], 'A1.0': [0, '-'], 'R0.2': [0, '-'], 'R0.4': [0, '-'], 'R0.6': [0, '-'], 'R0.8': [0, '-'], 'R1.0': [0, '-']}], '1.0': [0, {'A0.2': [0, '-'], 'A0.4': [0, '-'], 'A0.6': [0, '-'], 'A0.8': [0, '-'], 'A1.0': [0, '-'], 'R0.2': [0, '-'], 'R0.4': [0, '-'], 'R0.6': [0, '-'], 'R0.8': [0, '-'], 'R1.0': [0, '-']}]}
beta_r= {'A0.2': [0, '-'], 'A0.4': [0, '-'], 'A0.6': [0, '-'], 'A0.8': [0, '-'], 'A1.0': [0, '-'], 'R0.20.2': [0.003333, '-'], 'R0.20.4': [0.003333, '-'], 'R0.20.6': [0.993333, '-'], 'R0.20.8': [0, '-'], 'R0.21.0': [0, '-'], 'R0.40.2': [0.003333, '-'], 'R0.40.4': [0.003333, '-'], 'R0.40.6': [0.993333, '-'], 'R0.40.8': [0, '-'], 'R0.41.0': [0, '-'], 'R0.60.2': [1, '-'], 'R0.60.4': [0, '-'], 'R0.60.6': [0, '-'], 'R0.60.8': [0, '-'], 'R0.61.0': [0, '-'], 'R0.80.2': [1, '-'], 'R0.80.4': [0, '-'], 'R0.80.6': [0, '-'], 'R0.80.8': [0, '-'], 'R0.81.0': [0, '-'], 'R1.00.2': [1, '-'], 'R1.00.4': [0, '-'], 'R1.00.6': [0, '-'], 'R1.00.8': [0, '-'], 'R1.01.0': [0, '-']}

prev_p = [beta_p]
prev_r = [beta_r]

T = 3000
eta = 0.01
delta = 1
D=5
A = [round(i/D,2) for i in range(1,D+1)]

for t in tqdm.tqdm(range(int(T))):

            r_p_t_p_1 = agent_update(prev_r,A=A,delta=delta, M=eta)
            r_r_t_p_1 = agent_update(prev_p,A=A,delta=delta, M=eta, responder=True)

            if t % 100 == 0:

                print("FIRM")
                print("0.2 probability mass ", r_p_t_p_1["0.2"][0])
                print("0.4 probability mass ", r_p_t_p_1["0.4"][0])
                
                print()
                print("WORKER")
                print("0.2 branch")
                print("R0.2 0.2 ", r_r_t_p_1["R0.20.2"][0])
                print("R0.2 0.4 ", r_r_t_p_1["R0.20.4"][0])
                print("R0.2 0.6 ", r_r_t_p_1["R0.20.6"][0])
                print("0.4 branch")
                print(f"R0.4 0.2 ", r_r_t_p_1["R0.40.2"][0])
                print(f"R0.4 0.4 ", r_r_t_p_1["R0.40.4"][0])
                print(f"R0.4 0.6 ", r_r_t_p_1["R0.40.6"][0])
                print(f"A0.4", r_r_t_p_1["A0.4"][0])

            prev_p.append(r_p_t_p_1)
            prev_r.append(r_r_t_p_1)



# p_star = 0.3
# p_1 = 0.1
# p_2 = 0.2
# # p_3 = 0.3
# p = [p_1,p_2]#,p_3]

# #starting firm masses
# f_p1 = 0.5
# f_p2 = 0.5
# # f_p3 = 0.25
# # f_p4 = 0.25
# f_p = [f_p1,f_p2]#,f_p3,f_p4]

# # starting worker cumulative differences for each branch 
# U_w_p1_star = 0.9
# U_w_p1_1 = 0.2
# U_w_p1_2 = 0.1
# # U_w_p1_3 = 0.05
# U_f_p1_1 = 0.01
# U_f_p1_2 = 0.99
# # U_f_p1_3 = 0.21

# U_w_p1 = {p_1:U_w_p1_1, p_2:U_w_p1_2}#,p_3:U_w_p1_3}
# U_f_p1 = {p_1:U_f_p1_1, p_2:U_f_p1_2}#,p_3:U_f_p1_3}

# U_w_p2_star = 0.92
# U_w_p2_1 = 0.1
# U_w_p2_2 = 0.15
# # U_w_p2_3 = 0.05
# U_f_p2_1 = 0.99
# U_f_p2_2 = 0.01
# # U_f_p2_3 = 0.5

# U_w_p2 = {p_1:U_w_p2_1, p_2:U_w_p2_2}#,p_3:U_w_p2_3}
# U_f_p2 = {p_1:U_f_p2_1, p_2:U_f_p2_2}#,p_3:U_f_p2_3}

# k=3

# def calc_u_p_start(U_p_star,U_w_p,eta,delta,p_star,p,k):
#     u_star_term = (1/k)* (1+eta*np.sum([U_p_star - U_w_p[p2] for p2 in p])) * eta * delta *p_star
#     u_p_term = (1/k)*np.sum([(1+np.sum([U_w_p[p2] - U_w_p[q2] for q2 in p if q2 is not p2]))*eta*delta*p2 for p2 in p])

#     return u_star_term + u_p_term

# def calc_u_p_t(u_p_prev, p, U_f_p, p_star,fp,eta,delta,k):
#     p_star_sum = sum([(p_star-p2)*((1-p_star)*fp-(1-p2)*min(fp,(1/2)*(fp+U_f_p[p2]))) for p2 in p])
#     p_2_sum = sum([sum([(p2 - q2)*((1-p2)*min(fp,(1/2)*(fp+U_f_p[p2])) - (1-q2)*min(fp,(1/2)*(fp+U_f_p[q2]))) for q2 in p if q2 < p2])for p2 in p])

#     return u_p_prev + (eta**2 * delta**2)/k * (p_star_sum + p_2_sum)

# print("f_p1 - 1")
# u_p1_start = calc_u_p_start(U_w_p1_star,U_w_p1,eta,delta,p_star,p,k)
# print(f"u_p1_start = {u_p1_start}")
# u_p1_next = calc_u_p_t(u_p1_start, p,U_f_p1,p_star,f_p1,eta,delta,k)
# print(f"u_p1_t = {u_p1_next}")

# print("f_p2 - 1 ")
# u_p2_start = calc_u_p_start(U_w_p2_star,U_w_p2,eta,delta,p_star,p,k)
# print(f"u_p1_start = {u_p2_start}")
# u_p2_next = calc_u_p_t(u_p2_start, p,U_f_p2,p_star,f_p2,eta,delta,k)
# print(f"u_p1_t = {u_p2_next}")

# for t in range(100):
#     f_p1 = f_p1 + (u_p1_next - u_p2_next)/2
#     f_p2 = f_p2 + (u_p2_next - u_p1_next)/2

#     print(f"f_p1 - {t}")
#     u_p1_next = calc_u_p_t(u_p1_start, p,U_f_p1,p_star,f_p1,eta,delta,k)
#     print(f"u_p1_t = {u_p1_next}")

#     print(f"f_p2 - {t}")
#     u_p2_next = calc_u_p_t(u_p2_next, p,U_f_p2,p_star,f_p2,eta,delta,k)
#     print(f"u_p1_t = {u_p2_next}")


    
