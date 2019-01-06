import math

def occurences(examples):
    o = {}
    for example in examples:
        x, y = example
        if y not in o:
            o[y] = 0
        o[y] += 1
    return o

def entropy(examples):
    o = occurences(examples)
    entropy = 0
    for c in o:
        p = o[c] / len(examples)
        entropy += p * math.log(p)
    return entropy * -1

def information_gain(examples, attribute):
    if attribute is None:
        return 0
    o = occurences(examples)
    r = entropy(examples)
    for value, branch in split(examples, attribute).items():
        r -= (float(len(branch)) / float(len(examples))) * entropy(branch)
    return r

class HoeffdingAnytimeTree:
    
    classes = [0, 1]
    
    def __init__(self, examples, attributes, measure, delta):
        self.attributes = attributes
        self.examples = examples
        self.measure = measure
        self.delta = delta
        
        self.children = []
        self.split = None  # attribute that is splitted on this node
        self.value = None  # value of previous split
        self.label = majority_class(examples)[0]
                
    def __str__(self, prefix=""):
        if len(self.children) > 0:
            string = "{0}{1}: {2}\n".format(prefix, self.value, self.split)
        else:
            string = "{0}{1}: {2}\n".format(prefix, self.value, self.label)
        for child in self.children:
            string += prefix + child.__str__(prefix + "\t")
        return string
    
    def start(self):
        self.attributes += [None]
        
        for j, sample in enumerate(self.examples):
            print("Considering new example")
            x, y = sample
            path = self.sort(x)
            for index, node in enumerate(path):
                for attr in node.attributes:
                    pass
                    # self.n[attr][j][y][self] += 1
                # TODO: why is there a tab here ? (removed for now)
                if index == len(path) - 1:
                    node.attempt_to_split()
                else:
                    node.re_evaluate_best_split()
        
    def attempt_to_split(self):       
        label, all_same = majority_class(self.examples)
        
        if not all_same:
            
            print("Splitting", id(self))
            
            # Compute each G_l(X_i)
            measures = {}
            for attr in self.attributes:
                measures[attr] = measure(self, attr)
            
            # Find best attribute
            best = max(measures.items(), key=lambda item: item[1])[0]
            
            # Compute epsilon
            epsilon = 0  # TODO
            
            if measures[best] - measures[None] > epsilon and best is not None:
                self.split = best
                attributes = [attr for attr in self.attributes if attr != best]
                for value, examples in split(self.examples, best).items():
                    print("Adding leave for split", best, "with value", value)
                    leaf = HoeffdingAnytimeTree(
                        examples,
                        attributes,
                        self.measure,
                        self.delta
                    )
                    self.children.append(leaf)
                    leaf.value = value
    
    def re_evaluate_best_split(self):
        pass
        
    def sort(self, x):
        for child in self.children:
            if self.split is not None and x[self.split] == child.value:
                    return [self] + child.sort(x)
        if len(self.children) == 0:
            return [self]
        return []

def majority_class(examples):
    occurences = {}
    for x, y in examples:
        if y not in occurences:
            occurences[y] = 0
        occurences[y] += 1
    l = max(occurences.items(), key=lambda item: item[1])[0]
    return l, occurences[l] == len(examples)

def measure(node, attribute):
    """ If attribute is None, should be the G using 
    the most frequent label
    """
    return information_gain(node.examples, attribute)

def split(examples, attribute):
    children = {}
    for example in examples:
        x, y = example
        if x[attribute] not in children:
            children[x[attribute]] = []
        children[x[attribute]] += [example]
    return children
    
or_data = [
    ({ "a": "zero", "b": "zero"}, 0),
    ({ "a": "one", "b": "zero"}, 1),
    ({ "a": "zero", "b": "one"}, 1),
    ({ "a": "one", "b": "one"}, 1)
]

or_data += or_data

attributes = ["a", "b"]

h = HoeffdingAnytimeTree(or_data, attributes, measure, 0.9)
h.start()
print(h)