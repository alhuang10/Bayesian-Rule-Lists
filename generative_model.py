import math

import pandas as pd

def getNumAntecedents(A):
    count = 0
    for size in A:
        count += len(A[size])
    return count

# p(m|A,lmda)
def p_m(m, A, lmda):
    return (lmda ** m + 0.0) / (math.factorial(m) * math.e ** m)

# product from i=1 to m of p(c_j|c_1,...c_{j-1},A,eta)
def p_c(d, A, eta):
    normalizer = 0
    for size in A:
        normalizer += eta ** size / math.factorial(size)

    prod = 1.0
    for i in range(len(d)):
        prod *= eta ** len(d[i]) / (math.factorial(len(d[i])) * normalizer)
    return prod

# product from i=1 to m of p(a_j|a_1,...a_{j-1},A)
def p_a(d, A):
    sizes = {}
    for size in A:
        sizes[size] = 0

    prod = 1.0
    for i in range(len(d)):
        prod *= 1.0 / (len(A[len(d[i])]) - sizes[len(d[i])])
        sizes[len(d[i])] += 1
    return prod

# p(d|A,lmda,eta)
def p_d(d, A, lmda, eta):
    return p_m(len(d), A, lmda) * p_c(d, A, eta) * p_a(d, A)

# p(y|x,d,alpha)
def p_y(y, x, d, alpha):
    pass

# p(d|x,y,A,alpha,lmda,eta)
def p_d_given_data(d, x, y, A, alpha, lmda, eta):
    return p_d(d, A, lmda, eta) * p_y(y, x, d, alpha)
