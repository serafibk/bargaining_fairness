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

ROUND = 1 # change to match number of rounds

worker_payoffs = loaded_data
if ROUND == 1:
    A = np.array([[(1-w)-0.5 for c,w in enumerate(row)]for r,row in enumerate(worker_payoffs)]) # input row player values

    meta_game = nash.Game(A)

    # minimax solution
    f, w = meta_game.linear_program()# returns row strategy, col. strategy

    print("Minimax solution")
    print(f"firm equilibrium strategy: {f}")
    print(f"worker equilibrium strategy: {w}")
    print(f"Value of game: {np.transpose(f)@A@w}")

if ROUND == 2:
    firm_payoffs = np.array([[(1-w) for c,w in enumerate(row)]for r,row in enumerate(worker_payoffs)])

    meta_game = nash.Game(firm_payoffs, worker_payoffs)

# support enumeration solution - finds some equilibria of degenerate games
print("vertex enumeration")
equilibria = meta_game.vertex_enumeration()
for eq in equilibria:
    f,w = eq
    print(f"firm equilibrium strategy: {f}")
    print(f"worker equilibrium strategy: {w}")
    # print(f"Value of game: {np.transpose(f)@A@w}")

