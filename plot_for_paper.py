import matplotlib.pyplot as plt
import numpy as np

pr = np.arange(0,1,0.01)

delta = 0.9
tau = delta*delta/(1+delta)

gap_size = [(delta-tau)/(1-tau * p) for p in pr]

plt.plot(pr, gap_size)
plt.xlabel("Probability of f matching wtih c_1: p")
plt.ylabel("Gap Size: w_2 - w_1")
plt.title("Growth of gap size vs. p with delta = 0.9 and tau = delta^2/(1+delta)")
plt.show()

p = 0.5

deltas = np.arange(0.01,0.99,0.01)
taus = [delta*delta/(1+delta) for delta in deltas]

gap_size = [(delta-tau)/(1-tau * p) for delta,tau in zip(deltas,taus)]

plt.plot(deltas, gap_size)
plt.xlabel("Discount factor: delta")
plt.ylabel("Gap Size: w_2 - w_1")
plt.title("Growth of gap size vs. delta with p=0.5 and tau = delta^2/(1+delta)")
plt.show()
