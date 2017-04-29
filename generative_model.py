import math
import pandas as pd

OPTIMIZATION_THRESHOLD = 10

# p(y|x,d)
def p_y(y, x, d):
    prob = 1.0
    for i in range(len(x)):
        prob *= d.evaluate(x[i], y[i])
    return prob

# product from j=1 to m of p(a_j|a_1,...a_{j-1},a)
def p_a(d, a):
    """
    Sampling of antecedent a_j after size c_j is selected, uniform distribution antecedents that haven't been chosen (Page 6)

    d - antecedent list, length m, type BayesianRuleList
    a - complete, pre-mined collection of antecedents, contains R antecedents, up to C cardinality for each antecedent
    """
    sizes = {}
    for size in a.sizes():
        sizes[size] = 0

    prod = 1.0
    for i in range(d.length()):
        ant_len = d.getAntecedentByIndex(i).length()
        prod *= 1.0 / (a.lengthsBySize[ant_len] - sizes[ant_len])
        sizes[ant_len] += 1
    return prod

# product from i=1 to m of p(c_j|c_1,...c_{j-1},a,eta)
def p_c(d, a, eta):
    """
    Sampling of antecedent cardinalities, c_j represents the cardinality of the j-th antecedent in the list d, (Page 6)

    d - antecedent list, length m, type BayesianRuleList
    a - complete, pre-mined collection of antecedents, contains R antecedents, up to C cardinality for each antecedent
    eta - hyperparameter, models prior belief of the required antecedent cardinality, should be small compared to C
    """ 
    normalizer = 0  
    for size in a:
        normalizer += pow(eta, size) / math.factorial(size)

    prod = 1.0
    for i in range(d.length()):
        ant_len = d.getAntecedentByIndex(i).length()
        prod *= (pow(eta,ant_len) / (math.factorial(ant_len) * normalizer))
    return prod

# p(m|A,lmda)
def p_m(m, a, lmda):
    """
    Truncated Poisson for distribution over the antecedent list length (Page 5)

    m - length of the antecedent list
    a -  complete, pre-mined collection of antecedents, contains R antecedents, up to C cardinality for each antecedent
    lmda - hyperparameter that models the prior belief of the antecedent list length required to model the data
        - expected value of distribution when len(a) >> lmda
    """
    R = a.length()

    # Closed form approximation when length of pre-mind antecedents is sufficently larger
    if R >= OPTIMIZATION_THRESHOLD * m
        return (pow(lmda, m)) / (math.factorial(m) * pow(math.e,m))
    
    denominator = 0
    for i in range(R+1):
        denominator += (pow(lmda,i)) / (math.factorial(i))
    return (pow(lmda, m) / math.factorial(m)) / (denominator)

# p(d|a,lmda,eta)
def p_d(d, a, lmda, eta):
    """
    Prior probability for antecedent lists (Page 5)

    d - antecedent list, length m, type BayesianRuleList
    a - complete, pre-mined collection of antecedents, contains R antecedents, up to C cardinality for each antecedent
    """ 
    return p_m(d.length(), a, lmda) * p_c(d, a, eta) * p_a(d, a)

# p(d|x,y,a,alpha,lmda,eta)
def p_d_given_data(d, x, y, a, alpha, lmda, eta):
    """
    Proportional to posterior over antecedent lists for use in MCMC

    alpha: hyperparameter for Dirichlet prior
    """
    return p_d(d, a, lmda, eta) * p_y(y, x, d)
