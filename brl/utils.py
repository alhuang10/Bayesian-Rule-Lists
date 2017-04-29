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

class Consequent(object):
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def evaluate(self, y):
        return self.probabilities[y]

class IfThenBlock(object):
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent

    def checkIf(self, x):
        return self.antecedent.evaluate(x)

    def evaluate(self, y):
        return self.consequent.evaluate(y)

    def getAntecedent(self):
        return self.antecedent

class BayesianRuleList(object):
    def __init__(self, ifThenBlocks, defaultConsequent):
        self.ifThenBlocks = ifThenBlocks
        self.defaultConsequent = defaultConsequent

    def length(self):
        return len(self.ifThenBlocks)

    def getAntecedentByIndex(self, idx):
        return self.ifThenBlocks[idx].getAntecedent()

    def evaluate(self, x, y):
        for block in self.ifThenBlocks:
            if block.checkIf(x):
                return block.evaluate(y)
        return self.defaultConsequent.evaluate(y)

class AntecedentGroup(object):
    def __init__(self, antecedents):
        self.antecedentsBySize = defaultdict(list)
        for antecedent in antecedents:
            self.antecedentsBySize[antecedent.length()].append(antecedent)

    def sizes(self):
        return self.antecedentsBySize.keys()

    def lengthsBySize(self):
        return {k: len(v) for k, v in self.antecedentsBySize.items()}
