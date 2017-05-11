from collections import defaultdict
import random
import math
from collections import defaultdict
from brl.mcmc import *
from brl.generative_model import *
from brl.antecedent_mining import *
from operator import add
import scipy.stats as st
from sklearn.metrics import auc

def print_posterior_antecedent_list_results(N_posterior, brl_point_list, confidence_interval_width, alpha):

    for i, antecedent in enumerate(brl_point_list.antecedents):
        if i==0:
            print("If", end=" ")
        else:
            print("Else if", end=" ")
        posterior_dirichlet_parameter = list(map(add, alpha, N_posterior[i+1]))
        print(antecedent.print_antecedent())

        for j in range(len(N_posterior[i + 1])):
            print("\tLabel {} probability: {}".format(j, posterior_dirichlet_parameter[j] / sum(posterior_dirichlet_parameter)), end=" ")
            confidence_interval = compute_dirichlet_confidence_interval(posterior_dirichlet_parameter, j, confidence_interval_width)
            print(confidence_interval)

    # For no antecedent matching
    if 0 in N_posterior:
        posterior_dirichlet_parameter = list(map(add, alpha, N_posterior[0]))
        print("Else,")
        for j in range(len(N_posterior[i + 1])):
            print("\tLabel {} probability: {}".format(j, posterior_dirichlet_parameter[j] / sum(posterior_dirichlet_parameter)), end=" ")
            confidence_interval = compute_dirichlet_confidence_interval(posterior_dirichlet_parameter, j, confidence_interval_width)
            print(confidence_interval)

def generate_N_bold_posterior(data_matrix, outcomes, brl_point_list, number_of_possible_labels):

    N_posterior = defaultdict(lambda: [0]*number_of_possible_labels)

    for sample_features, label in zip(data_matrix, outcomes):

        # Returns 0 if no antecedents apply
        antecedent_index = brl_point_list.get_first_applying_antecedent(sample_features)
        N_posterior[antecedent_index][label] += 1 # Increment the appropriate label count by one

    return N_posterior

def compute_dirichlet_confidence_interval(concentration_parameters, index_to_compute, confidence_interval_width):

    param_sum = sum(concentration_parameters)
    index_value = concentration_parameters[index_to_compute]

    variance = (index_value*(param_sum-index_value)) / (pow(param_sum, 2)*(param_sum+1))
    standard_deviation = math.sqrt(variance)

    standard_error = standard_deviation / math.sqrt(param_sum)

    mean = index_value / param_sum

    z_value = st.norm.ppf(1.0 - (1.0 - confidence_interval_width)/2)
    # print(z_value)

    lower_bound = mean - z_value*standard_error
    upper_bound = mean + z_value*standard_error

    # print("95% Confidence Interval for Survival:", lower_bound, upper_bound)
    return (lower_bound, upper_bound)


def find_brl_point(generated_mcmc_samples, data_matrix, outcomes, all_antecedents, alpha, lmda, eta):

    antecedent_list_lengths = [ant_list.length() for ant_list in generated_mcmc_samples]
    average_antecedent_list_length = sum(antecedent_list_lengths) / len(antecedent_list_lengths)

    num_antecedents = sum(antecedent_list_lengths)
    total_cardinality = 0

    for ant_list in generated_mcmc_samples:
        for ant in ant_list.antecedents:
            total_cardinality += ant.length() 

    average_cardinality = total_cardinality / num_antecedents

    print("Average Antecedent List Length:", average_antecedent_list_length)
    print("Average Antecedent Cardinality:", average_cardinality)

    antecedent_list_length_bounds = [math.floor(average_antecedent_list_length), math.ceil(average_antecedent_list_length)]
    antecedent_cardinality_bounds = [math.floor(average_cardinality), math.ceil(average_cardinality)]

    print("List Length Bounds", antecedent_list_length_bounds)

    # Go through all the lists to get the highest posterior list 
    brl_point_list = None
    highest_posterior_probability = 0.0

    for antecedent_list in generated_mcmc_samples:

        average_cardinality = antecedent_list.get_average_cardinality()
        list_length = antecedent_list.length()
        
        if (average_cardinality >= antecedent_cardinality_bounds[0] and average_cardinality <= antecedent_cardinality_bounds[1]) and (list_length >= antecedent_list_length_bounds[0] and list_length <= antecedent_list_length_bounds[1]):

            # Take exponential because we do all probabilities using logs
            posterior_probability = math.exp(p_d_given_data(antecedent_list, data_matrix, outcomes, all_antecedents, alpha, lmda, eta))

            if posterior_probability > highest_posterior_probability:
                highest_posterior_probability = posterior_probability
                brl_point_list = antecedent_list

    return brl_point_list, highest_posterior_probability

def brl_point_predict(x_test_sample, N_posterior, antecedent_list, alpha, probability_threshold=0.5):

    antecedent_index = antecedent_list.get_first_applying_antecedent(x_test_sample)
    posterior_dirichlet_parameter = list(map(add,alpha,N_posterior[antecedent_index]))

    if len(posterior_dirichlet_parameter) == 2:
        positive_probability = posterior_dirichlet_parameter[1] / sum(posterior_dirichlet_parameter)

        if positive_probability > probability_threshold:
            return 1
        else:
            return 0

    # For multilabel
    else:
        return posterior_dirichlet_parameter.index(max(posterior_dirichlet_parameter))


def make_brl_test_set_predictions(data_test, outcome_test, N_posterior, brl_point_list, alpha, threshold, verbose=True):

    predictions = [brl_point_predict(test_sample, N_posterior, brl_point_list, alpha, probability_threshold=threshold) for test_sample in data_test]

    correct = 0
    incorrect = 0

    true_positive_count = 0
    total_positive_outcomes = 0

    false_positive_count = 0
    total_negative_outcomes = 0


    for i,val in enumerate(predictions):

        if outcome_test[i] == 1:
            total_positive_outcomes += 1

            if val == 1:
                true_positive_count += 1
        else:
            total_negative_outcomes += 1
            if val == 1:
                false_positive_count += 1


        if val == outcome_test[i]:
            correct += 1
        else:
            incorrect += 1

    fpr = false_positive_count / total_negative_outcomes
    tpr = true_positive_count / total_positive_outcomes
    
    if verbose:
        print("Correct: {}".format(correct))
        print("Incorrect: {}".format(incorrect))
        print("Percentage: {}".format(correct/(correct+incorrect)))
        print("False Positive Count, Total Negative Outcomes", false_positive_count, total_negative_outcomes)
        print("False Positive Rate: {}".format(fpr))
        print("True Positive Rate: {}".format(tpr))
        print("True Positive Count, Total Positive Outcomes", true_positive_count, total_positive_outcomes)
        print("Confusion Matrix:")
        print(total_negative_outcomes - false_positive_count, false_positive_count)
        print(total_positive_outcomes - true_positive_count, true_positive_count)

    return fpr, tpr, correct/(correct+incorrect)

# Finds AUC using fpr,tpr points and trapezoidal area method
def find_auc(data_test, outcome_test, N_posterior, brl_point_list, alpha):

    fprs = []
    tprs = []
    for i in range(0, 101):
        threshold = i/100.0
        fpr, tpr, accuracy = make_brl_test_set_predictions(data_test, outcome_test, N_posterior, brl_point_list, alpha, threshold, verbose=False)
        fprs.append(fpr)
        tprs.append(tpr)

    auc_value = auc(fprs, tprs)
    print("AUC: {}".format(auc_value))
    return auc_value






