import numpy as np
import matplotlib.pyplot as plt

# define the standard exponential distribution
x = np.linspace(-1, 5, 3000)
y = np.exp(-x)
# when x is negative, y is zero
y[x < 0] = 0
y[x > 2.5] = 0
y[1000:1008] = 0.8
plt.plot(x, y)
plt.fill_between(x, y, 0, alpha=0.3)
plt.xticks([x[1004]], [r'$J(\theta_r)$'])
plt.axvline(x=x[1004], color='grey', linestyle='-', linewidth=0.75)
plt.yticks([])
# remove upper and right spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.ylim(0, 1)
plt.xlim(-1, 5)
plt.show()