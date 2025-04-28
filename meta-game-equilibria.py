import numpy as np
import nashpy as nash
import warnings

filename = "meta_game_input_test"

with open(f"{filename}.txt", "r") as f:
    # read D, M, T, alpha_c_idx, alpha_f_idx from the file and store
    first_line = f.readline().strip().split()
    D = int(first_line[1])
    M = float(first_line[3])
    T = int(first_line[5])
    delta = float(first_line[7])
    eps = float(first_line[9])
    # alpha_c_idx = int(first_line[7])
    # alpha_f_idx = int(first_line[9])
    A = [i / (D) for i in range(D + 1)]

    loaded_data = np.loadtxt(f'{filename}.txt', skiprows=1, dtype=np.float64)
    # print(loaded_data.shape)
    # original_shape = ((D+1)**2, (D+1)**2)
    # payoff_matrix = loaded_data.reshape(original_shape)

# worker_payoffs = [[0.25,0.3,0.32],[0.31,0.3,0.25], [0.4,0.35,0.2]] # 3x3 game example, firm row player
N = 3 # number of strategies per player
worker_payoffs = loaded_data


A = np.array([[(1-w)-0.5 for c,w in enumerate(row)]for r,row in enumerate(worker_payoffs)]) # input row player values

meta_game = nash.Game(A)

# minimax solution
f, w = meta_game.linear_program()# returns row strategy, col. strategy

print("Minimax solution")
print(f"firm: {f}")
print(f"worker: {w}")
print(f"Value of game: {np.transpose(f)@A@w}")

# support enumeration solution - finds some equilibria of degenerate games
print("vertex enumeration")
equilibria = meta_game.vertex_enumeration()
for eq in equilibria:
    f,w = eq
    print(f"firm: {f}")
    print(f"worker: {w}")
    print(f"Value of game: {np.transpose(f)@A@w}")

