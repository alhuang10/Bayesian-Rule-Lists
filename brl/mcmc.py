import copy
import random

from .utils import *
from .generative_model import *

MOVE_TYPE = 0
REMOVE_TYPE = 1
ADD_TYPE = 2

NUM_CHAINS = 3

def generate_move_proposal(current_d):
    proposed_d = copy.deepcopy(current_d)

    i, j = -1, -1
    while i == j:
        i = random.randint(0, proposed_d.length() - 1)
        j = random.randint(0, proposed_d.length() - 1)

    proposed_d.move_antecedents(i, j)

    return proposed_d, 1.0

def generate_remove_proposal(current_d, all_antecedents):
    proposed_d = copy.deepcopy(current_d)

    proposed_d.remove_antecedent(random.randint(0, proposed_d.length() - 1))

    prob_backward = 1.0 / ((all_antecedents.length() - proposed_d.length()) * current_d.length())
    prob_forward = 1.0 / current_d.length()

    return proposed_d, prob_backward / prob_forward

def generate_add_proposal(current_d, all_antecedents):
    '''
    all_antecedents - AntecedentGroup
    current_d - current list, AntecedentList
    '''
    proposed_d = copy.deepcopy(current_d)

    antecedent = all_antecedents.get_random_antecedent()
    while proposed_d.contains(antecedent):
        antecedent = all_antecedents.get_random_antecedent()

    proposed_d.add_antecedent(random.randint(0, proposed_d.length()), antecedent)

    prob_backward = 1.0 / proposed_d.length()
    prob_forward = 1.0 / ((all_antecedents.length() - current_d.length()) * proposed_d.length())
    return proposed_d, prob_backward / prob_forward

def generate_proposal(current_d, all_antecedents):
    while True:
        proposal_type = random.randint(0, 2)
        if proposal_type == MOVE_TYPE:
            if current_d.length() == 1:
                continue
            return generate_move_proposal(current_d)
        if proposal_type == REMOVE_TYPE:
            if current_d.length() == 1:
                continue
            return generate_remove_proposal(current_d, all_antecedents)
        if proposal_type == ADD_TYPE:
            if current_d.length() == all_antecedents.length():
                continue
            return generate_add_proposal(current_d, all_antecedents)

def check_accepted(proposed_d, current_d, all_antecedents, x, y, move_ratio, alpha, lmda, eta):
    threshold = min(1.0, (move_ratio * math.exp(p_d_given_data(proposed_d, x, y, all_antecedents, alpha, lmda, eta)) /
        math.exp(p_d_given_data(current_d, x, y, all_antecedents, alpha, lmda, eta))))
    return random.random() < threshold

def check_gelman_rubin(chains, means, variances, threshold, x, y, all_antecedents, alpha, lmda, eta):
    samples = len(chains[0])
    if samples < 2:
        return False

    for i in range(NUM_CHAINS):
        val = p_d_given_data(chains[i][samples - 1], x, y, all_antecedents, alpha, lmda, eta)
        updated_mean = ((samples - 1) * means[i] + val) / samples
        updated_variance = ((samples - 2) * variances[i] + (val - updated_mean) * (val - means[i])) / (samples - 1)
        means[i] = updated_mean
        variances[i] = updated_variance

    w = 0
    for i in range(NUM_CHAINS):
        w += variances[i]
    w /= NUM_CHAINS

    mean_of_means = sum(means) / NUM_CHAINS
    b = 0
    for i in range(NUM_CHAINS):
        b += pow(mean_of_means - means[i], 2)
    b *= samples / (NUM_CHAINS - 1)

    v = ((samples - 1) * w + (NUM_CHAINS + 1) * b / NUM_CHAINS) / samples

    psrf = v / w
    print(psrf)
    
    return psrf < threshold

def brl_metropolis_hastings(min_num_iterations, burn_in, convergence_threshold, x, y, all_antecedents, alpha, lmda, eta):
    '''
    all_antecedents - AntecedentGroup
    min_num_iterations - minimum number of iterations to run the chains for
    burn_in - amount of time to let the chain burn in
    '''
    current_ds = []
    all_ds = []
    means = []
    variances = []
    for i in range(NUM_CHAINS):
        current_ds.append(generate_default_antecedent_list(all_antecedents, lmda, eta))
        all_ds.append([])
        means.append(0)
        variances.append(0)

    i = 0
    while not check_gelman_rubin(all_ds, means, variances, convergence_threshold, x, y, all_antecedents, alpha, lmda, eta) or i < min_num_iterations:
        if i % 100 == 0:
            print("Iteration: %d" % (i))

        for j in range(NUM_CHAINS):
            proposed_d, proposal_prob_ratio = generate_proposal(current_ds[j], all_antecedents)

            if check_accepted(proposed_d, current_ds[j], all_antecedents, x, y, proposal_prob_ratio, alpha, lmda, eta):
                current_ds[j] = proposed_d

            if i >= burn_in:
                all_ds[j].append(current_ds[j])

        i += 1

    print("Iterations Run:", i)
    return all_ds[0]
