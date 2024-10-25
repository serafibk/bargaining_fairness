from scipy.optimize import linprog
import numpy as np

# shared possible actions
S_i = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
joint_strategies = [(a_i,a_j) for a_i in S_i for a_j in S_i]
       
# payoff matrices
E_p = [1-S_i[i] if j <= i else 0 for i in range(len(S_i)) for j in range(len(S_i))]
E_r = [S_i[i] if j <= i else 0 for i in range(len(S_i)) for j in range(len(S_i))]


# constraint matrix construction, |S_i|-1 constraints for each possible signal (action) per agent
constraint_matrix = []

# proposer
for i_a in range(len(S_i)):
    for i_a_alt in range(len(S_i)): # need expected value for a to be greater than all other actions, given the signal a
        cons_vec = [0 for i in range(len(S_i)*len(S_i))] # track all possible probability variables
        if i_a == i_a_alt:
            continue # trivial constraint
        for j_resp_a in range(len(S_i)):
            prob_index = i_a * len(S_i) + j_resp_a
            utility_index_alt = i_a_alt * len(S_i) + j_resp_a

            cons_vec[prob_index] = E_p[utility_index_alt] - E_p[prob_index] # negated
        
        constraint_matrix.append(cons_vec)

# responder
for j_a in range(len(S_i)):
    for j_a_alt in range(len(S_i)): # need expected value for a to be greater than all other actions, given the signal a
        cons_vec = [0 for i in range(len(S_i)*len(S_i))] # track all possible probability variables
        if j_a == j_a_alt:
            continue # trivial constraint
        for i_prop_a in range(len(S_i)):
            prob_index = i_prop_a * len(S_i) + j_a
            utility_index_alt = i_prop_a * len(S_i) + j_a_alt

            cons_vec[prob_index] = E_r[utility_index_alt] - E_r[prob_index] # negated
        
        constraint_matrix.append(cons_vec)

constraint_ub = [0 for i in range(np.shape(constraint_matrix)[0])] # upper bound of each constraint is 0


# TODO -- add some constraints for not pure and not mixed (also just use gaussian elimination? - no that's for equalities)

# probability constraints
prob_constraint = [[1 for i in range(len(S_i)*len(S_i))]] # sum to 1
p_ij_bounds = (0,1) # valid probability values


# (negative) objective function try a range of objective functions
# for a_i in range(len(S_i)):
#     for a_j in range(len(S_i)):
social_welfare = [-1 if (i == 6 and j == 7) or (i == 5 and j == 6) else 0 for i in range(len(S_i)) for j in range(len(S_i))] # maximize probabilities of each possible combination
social_welfare[7*len(S_i)+8] =  -1
print(social_welfare)

result = linprog(social_welfare, A_ub=constraint_matrix, b_ub=constraint_ub, A_eq=prob_constraint, b_eq=1, bounds=[p_ij_bounds for i in range(len(S_i)*len(S_i))])
# print(f"A_i = {S_i[a_i]}, A_j = {S_i[a_j]}")
print(result.fun)
print(result.message)
for i,p in enumerate(result.x):
    if p > 0:
        print(f"Strategy: {joint_strategies[i]}, probability: {p}")
print()
        # print(result.x)
        


    




        



