from collections import defaultdict

class Expression(object):
    def __init__(self, index, op, value):
        self.index = index
        self.op = op
        self.value = value

    def evaluate(self, x):
        return self.op(x[self.index], self.value)

class Antecedent(object):
    def __init__(self, expressions):
        self.expressions = expressions

    def evaluate(self, x):
        return all(exp.evaluate(x) for exp in expressions)

    def length(self):
        return len(self.expressions)

class BayesianRuleList(object):
    def __init__(self, antecedents):
        self.antecedents = antecedents

    def length(self):
        return len(self.antecedents)

    def get_antecedent_by_index(self, idx):
        return self.antecedents[idx]

    def get_first_antecedent_index(self, x):
        for i in range(len(antecedents)):
            if antecedents[i].evaluate(x):
                return i
        return self.length()

class AntecedentGroup(object):
    def __init__(self, antecedents):
        self.antecedents_by_size = defaultdict(list)
        for antecedent in antecedents:
            self.antecedents_by_size[antecedent.length()].append(antecedent)

    def sizes(self):
        return self.antecedents_by_size.keys()

    def lengths_by_size(self):
        return {k: len(v) for k, v in self.antecedents_by_size.items()}
