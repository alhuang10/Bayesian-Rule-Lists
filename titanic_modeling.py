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

def find_brl(train_with_all):

    # ***Titanic-Specific Data Processing Start***
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

    data_matrix = None
    outcomes = None
    data_test = None
    outcomes_test = None

    if train_with_all:
        data_matrix = data_matrix_all
        outcomes = outcomes_all
        _, data_test, _, outcome_test = train_test_split(data_matrix_all, outcomes_all, test_size=.25)


    else:
        # If train with only a subset
        data_matrix, data_test, outcomes, outcome_test = train_test_split(data_matrix_all, outcomes_all, test_size=.25)

    num_samples = len(data_matrix)

    # FP-Growth Parameters
    min_support_threshold = .1 # Elements that do not meet the support threshold are excluded
    max_antecedent_length = 3 # Max length of antecedent lists to retrieve
    number_of_possible_labels = 2

    print("\nFP-Growth Parameters")
    print("Number of Training Samples: {}".format(num_samples))
    print("Minimum Support Threshold: {}".format(min_support_threshold))
    print("Max Antecedent Length: {}".format(max_antecedent_length))

    # MCMC Parameters
    alpha = [1,1]
    lmda = 1
    eta = 1
    num_iterations = 2000
    burn_in = 500
    convergence_threshold = 1.05
    confidence_interval_width = 0.95

    # Frequent-Pattern (FP) Growth Algorithm: (brl.antecedent_mining)
    all_antecedents = generate_antecedent_list(data_matrix, num_samples, min_support_threshold, max_antecedent_length)

    print("Number of Antecdents Mined: {}".format(len(all_antecedents.antecedents)))
    # MCMC - Metropolis Hastings
    print("\nMCMC Parameters:")
    print("Alpha", alpha)
    print("Lambda:", lmda)
    print("Eta:", eta)
    print("Min Number Iterations:", num_iterations)
    print("Burn In", burn_in)
    print("Convergence Threshold:", convergence_threshold, "\n")

    input("Press Enter to Begin MCMC...")

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
    make_brl_test_set_predictions(data_test, outcome_test, N_posterior, brl_point_list, alpha, 0.5)
    find_auc(data_test, outcome_test, N_posterior, brl_point_list, alpha)
    return N_posterior, brl_point_list

# Generate file for kaggle upload
def generate_titanic_kaggle_prediction(N_posterior, brl_point_list, alpha):

    output_file = open("titanic_test_set_results.txt", 'w')
    output_file.write("PassengerId,Survived\n")

    test_data = pd.DataFrame.from_csv("data/titanic_dataset/test.csv")
    test_data = test_data.reset_index()
    test_data =  test_data[['PassengerId','Pclass', 'Sex', 'Age']]
    test_data.fillna(30) # Replace missing age with 30

    test_data['Pclass'] = test_data['Pclass'].apply(convert_class)
    test_data['Age'] = test_data['Age'].apply(convert_age)
    test_data['Sex'] = test_data['Sex'].apply(lambda x: str.title(x))

    test_data_matrix = test_data.as_matrix()

    for i, features in enumerate(test_data_matrix):
        passenger_id = features[0]
        feats = features[1:]
        prediction = brl_point_predict(feats, N_posterior, brl_point_list, alpha)

        output_file.write(str(passenger_id))
        output_file.write(",")
        output_file.write(str(prediction))
        output_file.write("\n")

    output_file.close()



if __name__=='__main__':
    
    train_with_all = False
    N_posterior, brl_point_list = find_brl(train_with_all)
    alpha = [1,1]
    generate_titanic_kaggle_prediction(N_posterior, brl_point_list, alpha)
