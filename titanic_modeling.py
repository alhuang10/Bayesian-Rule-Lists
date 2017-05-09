#!/usr/bin/env python
from brl.antecedent_mining import *
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from collections import defaultdict
from brl.mcmc import *
from brl.generative_model import *
import time
import math

def find_brl():
    # Titanic-Specific Data Processing Start
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

    data = pd.DataFrame.from_csv("data/titanic_dataset/train.csv")
    data_only_relevant_features =  data[['Survived','Pclass', 'Sex', 'Age']]
    data_only_relevant_features = data_only_relevant_features.dropna() # Remove rows with missing information

    outcomes = data_only_relevant_features['Survived'].values.flatten()

    data_only_complete = data_only_relevant_features[['Pclass', 'Sex', 'Age']]
    # Convert data into categorical variables
    data_only_complete['Pclass'] = data_only_complete['Pclass'].apply(convert_class)
    data_only_complete['Age'] = data_only_complete['Age'].apply(convert_age)
    data_only_complete['Sex'] = data_only_complete['Sex'].apply(lambda x: str.title(x))

    # Titanic-Specific Data Processing End


    # Frequent-Pattern (FP) Growth Algorithm:

    # Parameters
    min_support_threshold = .1 # Elements that do not meet the support threshold are excluded
    max_antecedent_length = 3 # Max length of antecedent lists to retrieve

    data_matrix = data_only_complete.as_matrix()
    num_samples = len(data_matrix)

    counts = defaultdict(int)
    attribute_index =  {} # Keeps track of the index of an attribute in the data

    # Construct the counts list
    for person_features in data_matrix:
        for i, attribute in enumerate(person_features):
            counts[attribute] += 1

            if attribute not in attribute_index:
            	attribute_index[attribute] = i

    print("Feature Counts:", list(counts.items()))

    # Prune the counts list, remove every element that has support lower than the threshold
    pruned_counts = {k: v for k,v in counts.items() if v/num_samples > min_support_threshold}

    # Sort the pruned counts and iterate through each element in this order
    sorted_attributes = sorted(pruned_counts, key=pruned_counts.get, reverse=True)

    # Create the FP-Tree
    fp_tree = FP_Tree()

    # For each example, sort according to the order of the sorted_attributes list and add to fp_tree
    for sample in data_matrix:
        
        # Prune for minimum support before adding to tree
        sample = list(filter(lambda x: x in sorted_attributes, sample))    
        sample = sorted(sample, key=lambda x: sorted_attributes.index(x))
        fp_tree.insert_list(sample, fp_tree.root)

    # Run FP-Growth algorithm to find the frequent itemsets
    output_itemsets = []

    find_itemsets(fp_tree, [], output_itemsets, num_samples, min_support_threshold, max_antecedent_length, attribute_index)

    def all_unique(test_list):
        for i in range(len(test_list)):
            for j in range(i+1, len(test_list)):
                if test_list[i] == test_list[j]:
                    return False
        return True

    print("Number of Samples:", num_samples, "Minimum Support Threshold:", min_support_threshold, "Max Antecedent Length:", max_antecedent_length, "Antecedents Mined:", len(output_itemsets))


    # for item in output_itemsets:
    #     print(item)

    def create_expressions(ant_list):
    	expression = [Expression(attribute_index[ant], operator.eq, ant) for ant in ant_list]
    	return expression

    expressions_list = [create_expressions(ant_list) for ant_list in output_itemsets]
    antecedent_list = [Antecedent(expression) for expression in expressions_list]
    all_antecedents = AntecedentGroup(antecedent_list)

    # MCMC Parameters
    start = time.clock()

    alpha = [1,1]
    lmda = 3
    eta = 1
    num_iterations = 1000
    burn_in = 100

    print("\nMCMC Parameters:")
    print("Alpha", alpha, "\n", "Lambda:", lmda, "\n", "Eta:", eta, "\n", "Number Iterations:", num_iterations, "\n", "Burn In", burn_in, "\n")

    generated_samples = brl_metropolis_hastings(num_iterations, burn_in, data_matrix, outcomes, all_antecedents, alpha, lmda, eta)

    antecedent_list_lengths = [ant_list.length() for ant_list in generated_samples]
    average_antecedent_list_length = sum(antecedent_list_lengths) / len(antecedent_list_lengths)

    num_antecedents = sum(antecedent_list_lengths)
    total_cardinality = 0

    for ant_list in generated_samples:
        for ant in ant_list.antecedents:
            total_cardinality += ant.length() 

    average_cardinality = total_cardinality/num_antecedents

    print("Average Antecedent List Length:", average_antecedent_list_length)
    print("Average Antecedent Cardinality:", average_cardinality)

    antecedent_list_length_bounds = [math.floor(average_antecedent_list_length), math.ceil(average_antecedent_list_length)]
    antecedent_cardinality_bounds = [math.floor(average_cardinality), math.ceil(average_cardinality)]

    print("List Length Bounds", antecedent_list_length_bounds)

    # Go through all the lists to get the highest posterior list 
    highest_posterior_list = None
    highest_posterior_probability = 0.0

    for antecedent_list in generated_samples:

        average_cardinality = antecedent_list.get_average_cardinality()
        list_length = antecedent_list.length()
        
        if (average_cardinality >= antecedent_cardinality_bounds[0] and average_cardinality <= antecedent_cardinality_bounds[1]) and (list_length >= antecedent_list_length_bounds[0] and list_length <= antecedent_list_length_bounds[1]):

            # Take exponential because we do all probabilities using logs
            posterior_probability = math.exp(p_d_given_data(antecedent_list, data_matrix, outcomes, all_antecedents, alpha, lmda, eta))

            if posterior_probability > highest_posterior_probability:
                highest_posterior_probability = posterior_probability
                highest_posterior_list = antecedent_list


    print("Highest Posterior List:", highest_posterior_list)
    print("Highest Posterior Probability:", highest_posterior_probability)

    end = time.clock()

    print("Runtime in seconds:", end - start)

    print("\nBRL Point Estimate")
    highest_posterior_list.print_antecedent_list()

    return highest_posterior_list

if __name__=='__main__':
    brl_point = find_brl()
