import math

import pandas as pd

OPTIMIZATION_THRESHOLD = 10

# p(y|x,d,alpha)
def p_y(y, x, d, alpha):
    pass

# product from i=1 to m of p(a_j|a_1,...a_{j-1},a)
def p_a(d, a):
    sizes = {}
    for size in a.sizes():
        sizes[size] = 0

    prod = 1.0
    for i in range(d.length()):
        antLen = d.getAntecedentByIndex(i).length()
        prod *= 1.0 / (a.lengthsBySize[antLen] - sizes[antLen])
        sizes[antLen] += 1
    return prod

# product from i=1 to m of p(c_j|c_1,...c_{j-1},a,eta)
def p_c(d, a, eta):
    normalizer = 0
    for size in a:
        normalizer += eta ** size / math.factorial(size)

    prod = 1.0
    for i in range(d.length()):
        antLen = d.getAntecedentByIndex(i).length()
        prod *= (eta ** antLen / (math.factorial(antLen) * normalizer))
    return prod

# p(m|A,lmda)
def p_m(m, a, lmda):
    if a.length() >= OPTIMIZATION_THRESHOLD * m
        return (lmda ** m + 0.0) / (math.factorial(m) * math.e ** m)

    den = 0
    for i in range(a.length() + 1):
        den += (lmda ** i + 0.0) / (math.factorial(i))
    return (lmda ** m + 0.0) / (math.factorial(m) * den)

# p(d|a,lmda,eta)
def p_d(d, a, lmda, eta):
    return p_m(d.length(), a, lmda) * p_c(d, a, eta) * p_a(d, a)

# p(d|x,y,a,alpha,lmda,eta)
def p_d_given_data(d, x, y, a, alpha, lmda, eta):
    return p_d(d, a, lmda, eta) * p_y(y, x, d, alpha)
