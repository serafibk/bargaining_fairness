# x^t = x^t-1 -a + a* y^t-1, y^t = y^t-1 + b*x^t-1-c
import numpy as np


def generate_x_t_p_1(x_t,y_t,a):
    return x_t  + a*y_t-a

def generate_y_t_p_1(x_t,y_t,b,c):
    return y_t + b*x_t - c


if __name__=="__main__":
    x_1 = 0.6
    y_1 = 0.7
    c = 0.1
    a = 0.2
    b = 0.3

    print(1/2 * (x_1-(c+1-y_1)/b))
    print(1/(2*np.sqrt(a*b))*(x_1+c/b -a*y_1 +a-2*x_1))

    x_t_p_1 = generate_x_t_p_1(x_1,y_1,a)
    y_t_p_1 = generate_y_t_p_1(x_1,y_1,b,c)

    for n in range(10):
        print(f"x_t = {x_t_p_1}")
        print(f"y_t = {y_t_p_1}")

        x_t = generate_x_t_p_1(x_t_p_1,y_t_p_1,a)
        y_t = generate_y_t_p_1(x_t_p_1,y_t_p_1,b,c)

        x_t_p_1 = x_t
        y_t_p_1 = y_t
