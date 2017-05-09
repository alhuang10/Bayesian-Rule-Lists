from collections import defaultdict
import random
import math

class Expression(object):
    def __init__(self, index, op, value):
        self.index = index
        self.op = op
        self.value = value

    def evaluate(self, x):
        return self.op(x[self.index], self.value)

    def __eq__(self, other_exp):
        return self.index == other_exp.index and self.value == other_exp.value and self.op == other_exp.op

class Antecedent(object):
    def __init__(self, expressions):
        self.expressions = expressions

    def evaluate(self, x):
        return all(exp.evaluate(x) for exp in self.expressions)

    def length(self):
        return len(self.expressions)

    def __eq__(self, other):

        if self.length() != other.length():
            return False

        for i in range(self.length()):
            if self.expressions[i] != other.expressions[i]:
                return False
        return True

    def print_antecedent(self):
        exp_string = ""
        for exp in self.expressions:
            exp_string += exp.value
            exp_string +=", "
        return exp_string
            

class AntecedentList(object):
    def __init__(self, antecedents):
        self.antecedents = antecedents

    def get_antecedents(self):
        return self.antecedents

    def length(self):
        return len(self.antecedents)

    def get_antecedent_by_index(self, idx):
        return self.antecedents[idx]

    def get_first_antecedent_index(self, x):
        for i in range(len(self.antecedents)):
            if self.antecedents[i].evaluate(x):
                return i
        return self.length()

    def contains(self, antecedent):
        return any(antecedent == current_ant for current_ant in self.antecedents)

    def move_antecedents(self, i , j):
        temp = self.antecedents.pop(i)
        self.antecedents.insert(j, temp)

    def remove_antecedent(self, i):
        del self.antecedents[i]

    def add_antecedent(self, i, antecedent):
        self.antecedents.insert(i, antecedent)

    def get_average_cardinality(self):
        cardinalities = [antecedent.length() for antecedent in self.antecedents]
        return sum(cardinalities) / len(cardinalities)

    def print_antecedent_list(self):
        for antecedent in self.antecedents:
            print(antecedent.print_antecedent())

    def get_first_applying_antecedent(self, x_sample):

        for i, ant in enumerate(self.antecedents):
            if ant.evaluate(x_sample):
                # 1-indexing into antecedent list
                return i+1 
        # Return a 0 if none of the antecedents in the list apply to the sample
        return 0

class AntecedentGroup(object):
    def __init__(self, antecedents):
        self.antecedents_by_size = defaultdict(list)
        self.antecedents = antecedents
        for antecedent in antecedents:
            self.antecedents_by_size[antecedent.length()].append(antecedent)

    def sizes(self):
        return self.antecedents_by_size.keys()

    def lengths_by_size(self):
        return {k: len(v) for k, v in self.antecedents_by_size.items()}

    def length(self):
        return sum(self.lengths_by_size().values())

    def get_random_antecedent(self):
        return self.antecedents[random.randint(0, len(self.antecedents) - 1)]

    def get_antecedents_by_length(self, length):
        return self.antecedents_by_size[length]


