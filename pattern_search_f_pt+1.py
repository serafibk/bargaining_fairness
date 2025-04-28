import numpy as np


def split_p_2s(u, U_thresholds):
    # u is the upper bound threshold s.t. f_p is greater than all u's strictly less than this threshold
    P2 = []
    for u_thresh in U_thresholds:
        if u_thresh < u:
            P2.append((1/2)*u_thresh) # collect the values of U's that f_p must be greater than
    
    return P2


def calculate_f_1_values_and_bounds(U_1,U_2,U_3,d12,d13,P):

    f_1_values = []
    f_2_values = []
    f_3_values = []
    f_1_lower_bounds = []
    f_1_upper_bounds = []
    region=[]



    for i1,u1 in enumerate(U_1):
        P1_2 = split_p_2s(u1,U_1)
        P1_1 = P-len(P1_2)
        P1 = 1+P1_1+(1/2)*len(P1_2)
        if i1 ==0:
            u1_lower= 0
        else:
            u1_lower = U_1[int(i1-1)]

        for i2,u2 in enumerate(U_2):
            P2_2 = split_p_2s(u2,U_2)
            P2_1 = P-len(P2_2)
            P2 = 1+P2_1+(1/2)*len(P2_2)
            if i2 ==0:
                u2_lower= 0
            else:
                u2_lower = U_2[int(i2-1)]

            for i3,u3 in enumerate(U_3):
                P3_2 = split_p_2s(u3,U_3)
                P3_1 = P-len(P3_2)
                P3 = 1+P3_1+(1/2)*len(P3_2)
                if i3 ==0:
                    u3_lower= 0
                else:
                    u3_lower = U_3[int(i3-1)]

                f_1 = (1+(1/P2)*(d12-(1/2)*sum(P1_2)+(1/2)*sum(P2_2))+(1/P3)*(d13-(1/2)*sum(P1_2)+(1/2)*sum(P3_2)))/(1+P1/P2 + P1/P3)
                f_12_lower = (P2/P1)*(u2_lower+(1/P2)*(d12-(1/2)*sum(P1_2)+(1/2)*sum(P2_2)))
                f_13_lower = (P3/P1)*(u3_lower+(1/P3)*(d13-(1/2)*sum(P1_2)+(1/2)*sum(P3_2)))
                f_12_upper = (P2/P1)*(u2+(1/P2)*(d12-(1/2)*sum(P1_2)+(1/2)*sum(P2_2)))
                f_13_upper = (P3/P1)*(u3+(1/P3)*(d13-(1/2)*sum(P1_2)+(1/2)*sum(P3_2)))

                f_1_values.append(f_1)
                region.append(f"{i1},{i2},{i3}")
            
                f_1_lower_bounds.append(max([u1_lower,f_12_lower,f_13_lower]))
                f_1_upper_bounds.append(min([u1,f_12_upper,f_13_upper]))
                
                f_2_values.append((P1/P2)*f_1-(1/P2)*(d12-(1/2)*sum(P1_2)+(1/2)*sum(P2_2)))
                f_3_values.append((P1/P3)*f_1-(1/P3)*(d13-(1/2)*sum(P1_2)+(1/2)*sum(P3_2)))

    return f_1_values,f_1_lower_bounds,f_1_upper_bounds, f_2_values,f_3_values,region


if __name__ == "__main__":

    U_1 = [0.89,0.99,1]
    U_2 = [0.001,0.01,1]
    U_3 = [0.001,0.001,1]

    d12 = .01
    d13 = -.01

    P=4 # total number of offers

    f_1_values,f_1_lower_bounds,f_1_upper_bounds,f_2_v,f_3_v,regions = calculate_f_1_values_and_bounds(U_1,U_2,U_3,d12,d13,P)

    print("possible solutions")
    for v,l,u,f_2,f_3,r in zip(f_1_values,f_1_lower_bounds,f_1_upper_bounds,f_2_v,f_3_v,regions):
        # if l < v and v <= u:
        print(f"f_1 value: {v} and {l:.3f}<= f_1 < {u:.3f} in region {r}")
        # print(f"f_2: {f_2}, f_3: {f_3}")





