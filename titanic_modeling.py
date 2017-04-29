from antecedent_mining import *
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from collections import defaultdict

# Titanic Data Processing
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
data_only_relevant_features =  data[['Pclass', 'Sex', 'Age']]

data_only_complete = data_only_relevant_features.dropna() # Remove rows with missing information

# Convert data into categorical variables
data_only_complete['Pclass'] = data_only_complete['Pclass'].apply(convert_class)
data_only_complete['Age'] = data_only_complete['Age'].apply(convert_age)
data_only_complete['Sex'] = data_only_complete['Sex'].apply(lambda x: str.title(x))

# 714 samples
print("Number of Samples:", len(data_only_complete))



# Frequent-Pattern (FP) Growth Algorithm for Titanic:

# Algorithm Parameters
min_support_threshold = .05 # Elements that do not meet the support threshold are excluded
antecedent_length = 2 # Length of antecedent lists to retrieve

data_matrix = data_only_complete.as_matrix()
num_samples = len(data_matrix)

counts = defaultdict(int)

# Construct the counts list
for person_features in data_matrix:
    for attribute in person_features:
        counts[attribute] += 1

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
print(num_samples, min_support_threshold, antecedent_length)

find_itemsets(fp_tree, [], output_itemsets, num_samples, min_support_threshold, antecedent_length)

print("Output Itemsets:\n", output_itemsets)




