# make dummy data from a simple logistic regression model
# n: sample size
# beta: parameter vector for fixed effect
# gamma: parameter vector for random effect
# x: design matrix for fixed effect
# z: design matrix for random effect
# eta: linear predictor
# p: parameter for distribution followed by observation
# y: observation

import numpy as np
import scipy.stats as stat

n = 100
beta = np.array([0.1, 0.5])
gamma = np.array([0, 0.1])
x1 = stat.norm.rvs(1, 1, size=n).reshape(n, 1)
x2 = stat.norm.rvs(-1, 1, size=n).reshape(n, 1)
x = np.concatenate((x1, x2), axis=1)
z = stat.norm.rvs(gamma[0], gamma[1], size=[n,1])

eta = np.dot(x, beta).reshape(n, 1) + z

p = 1/(1+np.exp(-eta))

y = stat.binom.rvs(1, p)

data = np.concatenate((y,x,z), axis=1)

