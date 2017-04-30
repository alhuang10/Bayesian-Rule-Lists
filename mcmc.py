import copy
import random

from brl.utils import *
from generative_model import *

MCMC_TOTAL_ITERATIONS = 1000

MOVE_TYPE = 0
REMOVE_TYPE = 1
ADD_TYPE = 2

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
        assert False

def check_accepted(proposed_d, current_d, all_antecedents, x, y, move_ratio, alpha, lmda, eta):
    return random.random() < math.min(1.0, (move_ratio * p_d_given_data(proposed_d, x, y, all_antecedents, alpha, lmda, eta) /
        p_d_given_data(current_d, x, y, all_antecedents, alpha, lmda, eta)))

def brl_metropolis_hastings(x, y, all_antecedents, alpha, lmda, eta):
    current_d = generate_default_brl()
    all_ds = []

    for i in range(MCMC_TOTAL_ITERATIONS):
        proposed_d, proposal_prob_ratio = generate_proposal(current_d)
        if check_accepted(proposed_d, current_d, proposal_prob_ratio, alpha, lmda, eta):
            current_d = proposed_d
        all_ds.append(current_d)

