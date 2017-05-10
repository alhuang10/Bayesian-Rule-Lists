#!/usr/bin/env python
from brl.antecedent_mining import *
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from collections import defaultdict
from brl.mcmc import *
from brl.generative_model import *
from brl.utils import *
from brl.brl_methods import *
import time
import math
from sklearn.model_selection import train_test_split

def find_brl():

    # ***Titanic-Specific Data Processing Start***
    def convert_class(passenger_class):
        if passenger_class == 1:
            return "First Class"
        elif passenger_class == 2:
            return "Second Class"
        else: 
            return "Third Class"
        
    def convert_age(age):
        adult_age_cutoff = 18
        if age >= adult_age_cutoff:
            return "Adult"
        else:
            return "Child"

    test_data = pd.DataFrame.from_csv("data/titanic_dataset/test.csv")
    test_data_only_relevant_features =  test_data[['Pclass', 'Sex', 'Age']]
    test_data_only_relevant_features = test_data_only_relevant_features.dropna()

    # test_outcomes = test_data_only_relevant_features['Survived'].values.flatten()

    test_data_only_complete = test_data_only_relevant_features[['Pclass', 'Sex', 'Age']]
    # Convert data into categorical variables
    test_data_only_complete['Pclass'] = test_data_only_complete['Pclass'].apply(convert_class)
    test_data_only_complete['Age'] = test_data_only_complete['Age'].apply(convert_age)
    test_data_only_complete['Sex'] = test_data_only_complete['Sex'].apply(lambda x: str.title(x))   

    test_data_matrix = test_data_only_complete.as_matrix()



    data = pd.DataFrame.from_csv("data/titanic_dataset/train.csv")
    data_only_relevant_features =  data[['Survived','Pclass', 'Sex', 'Age']]
    data_only_relevant_features = data_only_relevant_features.dropna() # Remove rows with missing information

    outcomes_all = data_only_relevant_features['Survived'].values.flatten()

    data_only_complete = data_only_relevant_features[['Pclass', 'Sex', 'Age']]
    # Convert data into categorical variables
    data_only_complete['Pclass'] = data_only_complete['Pclass'].apply(convert_class)
    data_only_complete['Age'] = data_only_complete['Age'].apply(convert_age)
    data_only_complete['Sex'] = data_only_complete['Sex'].apply(lambda x: str.title(x))

    # ***Titanic-Specific Data Processing End***

    data_matrix_all = data_only_complete.as_matrix()

    # If train wtih all
    # data_matrix = data_matrix_all
    # outcomes = outcomes_all

    # If train with only a subset
    data_matrix, data_test, outcomes, outcome_test = train_test_split(data_matrix_all, outcomes_all, test_size=.25)

    num_samples = len(data_matrix)

    # FP-Growth Parameters
    min_support_threshold = .1 # Elements that do not meet the support threshold are excluded
    max_antecedent_length = 3 # Max length of antecedent lists to retrieve
    number_of_possible_labels = 2

    # MCMC Parameters
    alpha = [1,1]
    lmda = 1
    eta = 1
    num_iterations = 600
    burn_in = 0
    convergence_threshold = 1.05
    confidence_interval_width = 0.95

    # Frequent-Pattern (FP) Growth Algorithm: (brl.antecedent_mining)
    all_antecedents = generate_antecedent_list(data_matrix, num_samples, min_support_threshold, max_antecedent_length)

    # MCMC - Metropolis Hastings
    print("\nMCMC Parameters:")
    print("Alpha", alpha, "\n", "Lambda:", lmda, "\n", "Eta:", eta, "\n", "Min Number Iterations:", num_iterations, "\n", "Burn In", burn_in, "\n", "Convergence Threshold:", convergence_threshold, "\n")

    start = time.clock()
    generated_mcmc_samples = brl_metropolis_hastings(num_iterations, burn_in, convergence_threshold, data_matrix, outcomes, all_antecedents, alpha, lmda, eta)
    brl_point_list, highest_posterior_probability = find_brl_point(generated_mcmc_samples, data_matrix, outcomes, all_antecedents, alpha, lmda, eta)

    end = time.clock()
    print("Runtime in seconds:", end - start)

    # Generate the N's for each posterior using the data
    N_posterior = generate_N_bold_posterior(data_matrix, outcomes, brl_point_list, number_of_possible_labels)
 
    print("N_posterior:")
    print(N_posterior)
    print("\n")

    # Display the characteristics of the BRL point list
    print_posterior_antecedent_list_results(N_posterior, brl_point_list, confidence_interval_width, alpha)

    # Evaluate the BRL on the test set
    make_brl_test_set_predictions(data_test, outcome_test, N_posterior, brl_point_list, alpha)

    return N_posterior, brl_point_list


if __name__=='__main__':
    N_posterior, brl_point_list = find_brl()
