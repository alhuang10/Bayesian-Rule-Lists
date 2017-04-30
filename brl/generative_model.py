import math
import pandas as pd
import random
from .utils import *

OPTIMIZATION_THRESHOLD = 10

def generate_default_antecedent_list(all_antecedents, lmda, eta):

    p = random.random()
    sum = 0.0
    sampled_antecedent_list_length = None

    # Sample the length of the antecedent list
    for m in range(1, all_antecedents.length()+1):
        sum += math.exp(p_m(m, all_antecedents, lmda))
        if sum > p:
            sampled_antecedent_list_length = m
            break

    print(sum, p, sampled_antecedent_list_length, all_antecedents.length())

    antecedent_list = []

    iterations = 0
    while True:
        # For each antecedent in the list, sample the cardinality and get from all_antecedents
        for i in range(1,sampled_antecedent_list_length+1):
            p = random.random()
            sum = 0.0
            sampled_ant_cardinality = None

            denominator = 0.0
            # Potential change if A runs out of antecedents
            for k in all_antecedents.sizes():
                denominator += (pow(eta, k) / math.factorial(k))

            for c in all_antecedents.sizes():
                sum += (pow(eta, c) / math.factorial(c)) / denominator
                if sum > p:
                    sampled_ant_cardinality = c

            correct_length_antecedents = all_antecedents.get_antecedents_by_length(sampled_ant_cardinality)
            antecedent_list.append(correct_length_antecedents[random.randint(0, len(correct_length_antecedents)-1)])


        # Some temporary shit to check that we have no duplicates
        if len(antecedent_list) == len(set(antecedent_list)):
            break

        iterations += 1

        if iterations > 1000:
            print("Failed to sample unique elements, proceeding")
            antecedent_list = [ant for ant in iter(set(antecedent_list))]
            break

    return AntecedentList(antecedent_list)

# proportional to p(y|x,d,alpha)
def p_y(y, x, d, alpha):
    n = [[0 for col in range(len(alpha))] for row in range(d.length() + 1)]
    for i in range(len(x)):
        n[d.get_first_antecedent_index(x[i])][y[i]] += 1

    log_prod = 0
    for i in range(d.length() + 1):
        sum_i = 0
        for j in range(len(alpha)):
            sum_i += n[i][j] + alpha[j]
            log_prod += math.log(math.factorial(n[i][j] + alpha[j] - 1))
        log_prod -= math.log(math.factorial(sum_i - 1))
    return log_prod

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
        ant_len = d.get_antecedent_by_index(i).length()

        # print(ant_len, a.lengths_by_size()[ant_len], sizes[ant_len])
        
        prod *= 1.0 / (a.lengths_by_size()[ant_len] - sizes[ant_len])
        sizes[ant_len] += 1
    return math.log(prod)

# product from i=1 to m of p(c_j|c_1,...c_{j-1},a,eta)
def p_c(d, a, eta):
    """
    Sampling of antecedent cardinalities, c_j represents the cardinality of the j-th antecedent in the list d, (Page 6)

    d - antecedent list, length m, type BayesianRuleList
    a - complete, pre-mined collection of antecedents, contains R antecedents, up to C cardinality for each antecedent
    eta - hyperparameter, models prior belief of the required antecedent cardinality, should be small compared to C
    """ 
    normalizer = 0  
    for size in a.sizes():
        normalizer += pow(eta, size) / math.factorial(size)

    prod = 1.0
    for i in range(d.length()):
        ant_len = d.get_antecedent_by_index(i).length()
        prod *= (pow(eta,ant_len) / (math.factorial(ant_len) * normalizer))
    return math.log(prod)

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
    if R >= OPTIMIZATION_THRESHOLD * m:
        # print("OPTIMIZATION_THRESHOLD reached")
        return math.log((pow(lmda, m)) / (math.factorial(m) * (pow(math.e, lmda) - 1)))
    
    denominator = 0
    for i in range(1, R+1):
        denominator += (pow(lmda,i)) / (math.factorial(i))
    return math.log((pow(lmda, m) / math.factorial(m)) / (denominator))

# p(d|a,lmda,eta)
def p_d(d, a, lmda, eta):
    """
    Prior probability for antecedent lists (Page 5)

    d - antecedent list, length m, type BayesianRuleList
    a - complete, pre-mined collection of antecedents, contains R antecedents, up to C cardinality for each antecedent
    """ 
    return p_m(d.length(), a, lmda) + p_c(d, a, eta) + p_a(d, a)

# p(d|x,y,a,alpha,lmda,eta)
def p_d_given_data(d, x, y, a, alpha, lmda, eta):
    """
    Proportional to posterior over antecedent lists for use in MCMC

    alpha: hyperparameter for Dirichlet prior
    """
    return p_d(d, a, lmda, eta) + p_y(y, x, d, alpha)
