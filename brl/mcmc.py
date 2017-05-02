import copy
import random

from .utils import *
from .generative_model import *

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

    assert current_d.length() == proposed_d.length()

    return proposed_d, 1.0

def generate_remove_proposal(current_d, all_antecedents):

    assert current_d.length() > 1

    proposed_d = copy.deepcopy(current_d)

    proposed_d.remove_antecedent(random.randint(0, proposed_d.length() - 1))

    prob_backward = 1.0 / ((all_antecedents.length() - proposed_d.length()) * current_d.length())
    prob_forward = 1.0 / current_d.length()

    assert current_d.length() - 1 == proposed_d.length()

    return proposed_d, prob_backward / prob_forward

def generate_add_proposal(current_d, all_antecedents):
    '''
    all_antecedents - AntecedentGroup
    current_d - current list, AntecedentList
    '''

    proposed_d = copy.deepcopy(current_d)

    antecedent = all_antecedents.get_random_antecedent()
    while proposed_d.contains(antecedent):
        print("Antecedent already present, trying again")
        antecedent = all_antecedents.get_random_antecedent()

    proposed_d.add_antecedent(random.randint(0, proposed_d.length()), antecedent)

    prob_backward = 1.0 / proposed_d.length()
    prob_forward = 1.0 / ((all_antecedents.length() - current_d.length()) * proposed_d.length())

    assert current_d.length() + 1 == proposed_d.length()

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

            # print(current_d.length())

            return generate_remove_proposal(current_d, all_antecedents)
        if proposal_type == ADD_TYPE:
            if current_d.length() == all_antecedents.length():
                continue
            return generate_add_proposal(current_d, all_antecedents)
        assert False

def check_accepted(proposed_d, current_d, all_antecedents, x, y, move_ratio, alpha, lmda, eta):

    threshold = min(1.0, (move_ratio * math.exp(p_d_given_data(proposed_d, x, y, all_antecedents, alpha, lmda, eta)) /
        math.exp(p_d_given_data(current_d, x, y, all_antecedents, alpha, lmda, eta))))

    # print(proposed_d.length(), current_d.length(), move_ratio)

    # print(proposed_d.length(), current_d.length(), move_ratio)
    # print("Proposed probability:", math.exp(p_d_given_data(proposed_d, x, y, all_antecedents, alpha, lmda, eta)), 
    #     "Current state probability", math.exp(p_d_given_data(current_d, x, y, all_antecedents, alpha, lmda, eta)))
    # print("\n")

    # print(proposed_d.length(), current_d.length(), move_ratio, math.exp(p_d_given_data(proposed_d, x, y, all_antecedents, alpha, lmda, eta)) /
    #     math.exp(p_d_given_data(current_d, x, y, all_antecedents, alpha, lmda, eta)), threshold)

    return random.random() < threshold


# TODO - burn in
# TODO - convergence
def brl_metropolis_hastings(num_iterations, x, y, all_antecedents, alpha, lmda, eta):

    '''
    all_antecedents - AntecedentGroup
    '''
    current_d = generate_default_antecedent_list(all_antecedents, lmda, eta)

    assert current_d.length() != 0

    all_ds = []

    def all_unique(test_list):
        for i in range(len(test_list)):
            for j in range(i+1, len(test_list)):
                if test_list[i] == test_list[j]:
                    return False
        return True


    for i in range(num_iterations):
        if i % 100 == 0:
            print("Iteration: %d" % (i))
        proposed_d, proposal_prob_ratio = generate_proposal(current_d, all_antecedents)
        assert all_unique(current_d.antecedents)
        if check_accepted(proposed_d, current_d, all_antecedents, x, y, proposal_prob_ratio, alpha, lmda, eta):
            current_d = proposed_d
        all_ds.append(current_d)

    return all_ds
