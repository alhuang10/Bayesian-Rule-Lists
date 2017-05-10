# FP-Growth classes and methods
# from utils import *
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import operator
from .utils import *
from collections import defaultdict

def generate_antecedent_list(data_matrix, num_samples, min_support_threshold, max_antecedent_length):
    
    counts = defaultdict(int)
    attribute_indices =  {} # Keeps track of the index of an attribute in the data

    # Construct the counts list
    for person_features in data_matrix:
        for i, attribute in enumerate(person_features):
            counts[attribute] += 1

            if attribute not in attribute_indices:
                attribute_indices[attribute] = i

    # print("Feature Counts:", list(counts.items()))

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

    find_itemsets(fp_tree, [], output_itemsets, num_samples, min_support_threshold, max_antecedent_length, attribute_indices)

    # print("Number of Samples:", num_samples, "Minimum Support Threshold:", min_support_threshold, "Max Antecedent Length:", max_antecedent_length, "Antecedents Mined:", len(output_itemsets))

    def create_expressions(raw_ant_list):
        expression = [Expression(attribute_indices[ant], operator.eq, ant) for ant in raw_ant_list]
        return expression

    expressions_list = [create_expressions(raw_ant_list) for raw_ant_list in output_itemsets]
    antecedent_list = [Antecedent(expression) for expression in expressions_list]
    all_antecedents = AntecedentGroup(antecedent_list)

    return all_antecedents

# Finds itemsets of all lengths, can add functionality to support min_length or certain length only
def find_itemsets(current_tree, suffixes_found, output_list, total_transactions, min_support, max_antecedent_length, attribute_indices):
    reverse_ordering = sorted(list(current_tree.item_counts.keys()), key=current_tree.item_counts.get)
        
    for attribute in reverse_ordering:
            
        nodes = current_tree.item_nodes[attribute]

        # Check for support
        support = current_tree.item_counts[attribute] / total_transactions
        
        if support >= min_support and attribute not in suffixes_found:
            new_suffix_set = [attribute]
            new_suffix_set.extend(suffixes_found)
            
            # Only get specific length antecedents
            if len(new_suffix_set) <= max_antecedent_length:
                output_list.append(new_suffix_set)
            
            conditional_tree = create_conditional_tree(current_tree.get_prefix_paths(attribute))
            find_itemsets(conditional_tree, new_suffix_set, output_list, total_transactions, min_support, max_antecedent_length, attribute_indices)
    
# Create a conditional tree using paths generated from get_prefix_paths
def create_conditional_tree(paths):
    tree = FP_Tree()
    conditional_attribute = paths[0][-1].attribute # Paths are from root to end
    # For each path, parent starts as tree.root 
    current_parent = tree.root
    
    for path in paths:
        for node in path:
            attr = node.attribute
            # count is 0 unless it is the attribute of interest
            count = 0
            if attr == conditional_attribute:
                count = node.count
            
            # Check if current_parent already has a child with given attr, otherwise create a new node to mimic it
            attr_node = next( (node for node in current_parent.child_list if node.attribute == attr), None)

            # All we have to do is switch because we want 0 counts
            if attr_node is not None:
                current_parent = attr_node
            else:
                new_node = FP_Node(attr, count, current_parent)
                current_parent.child_list.append(new_node)
                
                # Add the new node to the tree record
                if attr in tree.item_nodes[attr]:
                    tree.item_nodes[attr][-1].neighbor = attr
                    
                tree.item_nodes[attr].append(new_node)
                current_parent = new_node
            
            # Will modify only for the conditional
            tree.item_counts[attr] += count
                
        # reset for next root
        current_parent = tree.root
        
    # Once constructed, get the prefix paths and add one to each element in each prefix path
    paths = tree.get_prefix_paths(conditional_attribute)
    for path in paths:
        
        count_to_add = path[-1].count
        
        for node in path:
            cur_attr = node.attribute
            if cur_attr != conditional_attribute:
                node.count += count_to_add
                tree.item_counts[cur_attr] += count_to_add
                
    # After getting the conditional tree, prune all the nodes of the conditional attribute
    
    for cond_node in tree.item_nodes[conditional_attribute]:
        parent = cond_node.parent
        parent.child_list = []
        
    del tree.item_counts[conditional_attribute]
    del tree.item_nodes[conditional_attribute]
    
    # Might want to prune nodes that don't meet support but can probably just do that externally with checks    
    return tree

class FP_Node:
    
    # attribute
    # count
    # child list
    
    def __init__(self, attribute, count, parent):
        
        self.attribute = attribute
        self.count = count
        self.parent = parent
        self.child_list = []
        self.neighbor = None
        
class FP_Tree:
    
    # item_table - attribute to (count, pointer_to_tree)
    # root - null node
    
    def __init__(self):
        self.root = FP_Node(None, 0, None)
        self.item_counts = defaultdict(int)
        self.item_nodes = defaultdict(list) # Ordered by insertion into tree
    
    # Helper method for constructing prefix path trees
    def add_node(self, attribute, count, parent_node):
        
        new_node = FP_Node(attribute, count, parent_node)    
        parent_node.child_list.append(new_node)
    
        self.item_counts[attribute] += count
        
        if attribute in self.item_nodes:
            # Mark neighbor of most recent node of same attribute
            # Only time we set neighbors
            self.item_nodes[attribute][-1].neighbor = new_node
                
        self.item_nodes[attr_to_insert].append(attr_node) # Add new node to item_nodes
      

    # Starting from an end node get the path of the node
    # Helper method for getting all paths ending with a given attribute
    # Each path starts with the root
    def collect_path(self, node):
        
        path = []
        
        current_node = node
        while current_node.parent is not None: # While not root
            path.append(current_node)
            current_node = current_node.parent # move up to parent
        
#         path.append(current_node) # Append the root
        path.reverse()
        
        return path
        
    def get_prefix_paths(self, attribute):
        
        attribute_nodes = self.item_nodes[attribute]
        all_paths = [self.collect_path(node) for node in attribute_nodes]
        
        return all_paths
        
    def insert_list(self, item_list, parent_node):
                
        # Base case
        if len(item_list) == 0:
            return
        else:
            
            attr_to_insert = item_list[0]
            
            # If parent_node has a direct child with same attribute name then increment
            attr_node = next( (node for node in parent_node.child_list if node.attribute == attr_to_insert), None)
            
            # Else create a new node with provided parent node (link both ways)
            if attr_node is None:
                
                attr_node = FP_Node(attr_to_insert, 1, parent_node)
                parent_node.child_list.append(attr_node)
                
                if attr_to_insert in self.item_nodes:
                    # Mark neighbor of most recent node of same attribute
                    # Only time we set neighbors
                    self.item_nodes[attr_to_insert][-1].neighbor = attr_node
                
                self.item_nodes[attr_to_insert].append(attr_node) # Add new node to item_nodes
              
            else:
                attr_node.count += 1
            
            self.item_counts[attr_to_insert] += 1 # Increment item count            
                
            # Recursively call the rest of the list
            # Parent node either a newly created node or an existing child
            self.insert_list(item_list[1:], attr_node)        