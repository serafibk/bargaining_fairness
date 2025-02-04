# x^t = x^t-1 - c + a* y^t-1, y^t = y^t-1 + b*x^t-1


def generate_x_t_p_1(x_t,y_t,c,a):
    return x_t - c + a*y_t

def generate_y_t_p_1(x_t,y_t,b):
    return y_t + b*x_t


if __name__=="__main__":
    x_1 = 0.3
    y_1 = 0.05
    c = 0.1
    a = 0.2
    b = 0.3

    x_t_p_1 = generate_x_t_p_1(x_1,y_1,c,a)
    y_t_p_1 = generate_y_t_p_1(x_1,y_1,b)

    for n in range(10):
        print(f"x_t = {x_t_p_1}")
        print(f"y_t = {y_t_p_1}")

        x_t = generate_x_t_p_1(x_t_p_1,y_t_p_1,c,a)
        y_t = generate_y_t_p_1(x_t_p_1,y_t_p_1,b)

        x_t_p_1 = x_t
        y_t_p_1 = y_t
