import pandas as pd
import numpy as np
import scipy.stats
import copy
import math
import random
#from collections import defaultdict

LOW = "low"
HIGH = "high"
CLASS = "class"

################# entropoy
# Input:
# data_frame -- pandas data frame
#
# Output:
# answer -- float indicating the empirical entropy of tyhe data in data_frame
##############################################
def entropy(data_frame : pd.DataFrame):


    # need to vectorize

    '''
    positive = 0
    negative = 0
    for i in data_frame.index:
        if data_frame[CLASS][i] == HIGH:
            positive += 1
        else:
            negaive += 1
    '''

    positive = len(data_frame[data_frame[CLASS] == HIGH])
    negative = len(data_frame[data_frame[CLASS] == LOW])
    
    total = positive + negative
    
    return scipy.stats.entropy(np.array([positive / total, negative / total]), base = 2)

################# info_gain
#
# Inputs:
# data_frame -- pandas data frame
# attribute -- string indicating the attribute for which we wish to compute the information gain
# domain -- set of values (strings) that the attribute can take
#
# Output:
# answer -- float indicating the information gain
################################################3
def info_gain(data_frame : pd.DataFrame, attribute : str, domain : set):

    # need to vectorize
    remainder = 0
    
    for d in domain:

        '''
        positive = 0
        negative = 0

        for i in range(len(data_frame.index)):
            #print(f"i = {i}")
            if data_frame.iloc[i][attribute] == d and data_frame.iloc[i][CLASS] == HIGH:
                positive += 1
            elif data_frame.iloc[i][attribute] == d and data_frame.iloc[i][CLASS] == LOW:
                negative += 1
        
        '''
        domain_df = data_frame[data_frame[attribute] == d]
        positive = domain_df[domain_df[CLASS] == HIGH]
        negative = domain_df[domain_df[CLASS] == LOW]
        total = len(positive) + len(negative)

        #print(f"postive:{positive}, negative:{negative}, size:{len(data_frame.index)}, attribute:{attribute}, d:{d}")

        '''
        # to ensure we dont count entropy for v_i not in dataframe
        if total > 0:
            remainder += ((positive + negative) / (len(data_frame.index))) * (scipy.stats.entropy(np.array([positive / total, negative / total]), 2))
        '''
        if total > 0:
            remainder += ((total) / (len(data_frame.index))) * entropy(domain_df)
    
    return entropy(data_frame) - remainder

######## Decision_tree class
#
# This class defines the data structure of the decision tree to be learnt
############################
class Decision_Tree:

    # constructor
    def __init__(self,attribute,branches,label):
        self.attribute = attribute
        self.branches = branches
        self.label = label

    # leaf constructor
    def make_leaf(label : str):
        return Decision_Tree('class', {}, label)
    
    # node constructor
    def make_node(attribute : str, branches : dict):
        return Decision_Tree(attribute, branches, None)

    # string representation
    def __repr__(self):
        return self.string_repr(0)
        
    # decision tree string representation
    def string_repr(self,indent):
        indentation = '\t'*indent
        
        # leaf string representation
        if self.attribute == 'class':
            return f'\n{indentation}class = {self.label}'

        # node string representation
        else:
            representation = ''
            for value in self.branches:
                representation += f'\n{indentation}{self.attribute} = {value}:'
                representation += self.branches[value].string_repr(indent+1)
            return representation

    # classify a data point
    def classify(self, data_point : pd.DataFrame):

        # leaf
        if self.attribute == 'class':
            return self.label

        # node
        else:
            return self.branches[data_point[self.attribute]].classify(data_point)

############# choose attribute
#
# Inputs:
# attributes_with_domains -- dictionary with attributes as keys and domains as values
# data_frame -- pandas data_frame
#
# Output:
# best_score -- float indicting the information gain score of the best attribute
# best_attribute -- string indicating the best attribute
#################################
def choose_attribute(attributes_with_domains : dict, data_frame : pd.DataFrame):

    # default value until this function is filled in
    best_score = -1
    best_attribute = None

    for attribute, domain in attributes_with_domains.items():
        score = info_gain(data_frame, attribute, domain)
        #print(f"attribute:{attribute}, score:{score}")
        if score > best_score:
            best_attribute = attribute
            best_score = score

    return best_score, best_attribute

############# train decision tree
# choose_attribute,  Decision_Tree.make_leaf, Decision_Tree.make_node
#
# Inputs:
# data_frame -- pandas data frame
# attributes_with_domains -- dictionary with attributes as keys and domains as values
# default_class -- string indicating the class to be assigned when data_frame is empty
# threshold -- integer indicating the minimum number of data points in data_frame to allow
#              the creation of a new node that splits the data with some attribute
#
# Output:
# decision_tree -- Decision_Tree object
########################
def train_decision_tree(data_frame : pd.DataFrame, attributes_with_domains : dict, default_class : str, threshold : int):

    # empty base case
    if data_frame.empty:
        return Decision_Tree.make_leaf(default_class)
    

    # check if all classifications are the same, can possibly vectorize
    allSame = True
    for i in range(len(data_frame.index) - 1):
        if data_frame.iloc[i][CLASS] != data_frame.iloc[i + 1][CLASS]:
            allSame = False
            break
    
    mode_series = data_frame[CLASS].mode()
    mode = mode_series[random.randint(0, mode_series.size - 1)]
    # attributes are empty, ask about mode
    if len(attributes_with_domains) == 0 or len(data_frame.index) < threshold:
        return Decision_Tree.make_leaf(mode)
    elif allSame:
        #print(data_frame)
        return Decision_Tree.make_leaf(data_frame.iloc[0][CLASS])
    else:
        best_score, best_attribute = choose_attribute(attributes_with_domains, data_frame)
        #print(f"BEST ATTRIBUTE:({best_attribute}, {best_score})")


        new_attr = copy.deepcopy(attributes_with_domains)
        new_attr.pop(best_attribute)

        subtrees = {}

        '''
        values = defaultdict(list)
        for idx in range(len(data_frame.index)):
            values[data_frame.iloc[idx][best_attribute]].append(data_frame.iloc[idx])
        
        for key, value in values.items():
            df = pd.concat(value, axis = 'columns')
            print(df)
            subtrees.append(train_decision_tree(df, attributes_with_domains, mode, threshold))
        '''

        for d in attributes_with_domains[best_attribute]:
            subtrees[d] = (train_decision_tree(data_frame[data_frame[best_attribute] == d], new_attr, mode, threshold))
        
        return Decision_Tree.make_node(best_attribute, subtrees)

    
    



######### eval decision tree
#
# Inputs:
# decision tree -- Decision_Tree object
# data_frame -- pandas data frame
#
# Output:
# accuracy -- float indicating the accuracy of the decision tree
#############
def eval_decision_tree(decision_tree : Decision_Tree, data_frame : pd.DataFrame):

    correct = 0
    for row in range(len(data_frame.index)):
        # need to vectorize, not sure I can
        estimated = decision_tree.classify(data_frame.iloc[row])
        #print(estimated)
        if data_frame.iloc[row][CLASS] == estimated:
            correct += 1

    # return the accuracy
    accuracy = correct / len(data_frame.index)
    #print(f"accuracy = {accuracy}")
    return accuracy

########### k-fold cross-validation
#
# Inputs:
# train_data -- pandas data frame
# test_data -- pandas data frame
# attributes_with_domains -- dictionary with attributes as keys and domains as values
# k -- integer indicating the number of folds
# threshold_list -- list of thresholds to be evaluated
#
# Outputs:
# best_threshold -- integer indiating the best threshold found by cross validation
# test_accuracy -- float indicating the accuracy based on the test set
#####################################
def cross_validation(train_data : pd.DataFrame, test_data : pd.DataFrame, attributes_with_domains : dict, k : int, threshold_list : list):

    avg_accuracy = []
    # split by rows to cross validate
    parts = np.array_split(train_data, k)

    for threshold in threshold_list:
        accuracy = []
        # cross validation
        for i in range(k):
            # split by rows to cross validate
            validation_data = parts[i]

            pieces = []
            for j in range(len(parts)):
                if i != j:
                    pieces.append(parts[j])
            
            train_data_split = pd.concat(pieces)

            #print(validation_data)
            # ask about default classification, case where we're under threshold(I think we pick the mode)
            decision_tree = train_decision_tree(train_data_split, attributes_with_domains, LOW, threshold)
            #print(decision_tree)
            #print(f"Eval for {i+1}th fold, threshold = {threshold}")
            accuracy.append(eval_decision_tree(decision_tree, validation_data))
        
        #print(f"accuracies for theshold {theshold} is: {accuracy}")
        #print(f"average is: {sum(accuracy) / len(accuracy)}")
        avg_accuracy.append(math.fsum(accuracy) / len(accuracy))

    
    print("Average Accuracies")
    for i in range(len(avg_accuracy)):
        print(f"Accuracy of threshold = {threshold_list[i]} is {avg_accuracy[i]}")
    
    #print(f"Average accuracies: {avg_accuracy}")
    # find the best accuracy idx 
    best_accuracy_idx = 0
    for i in range(1, len(avg_accuracy)):
        if avg_accuracy[i] > avg_accuracy[best_accuracy_idx]:
            best_accuracy_idx = i
    
    # print(avg_accuracy)

    # test data
    best_threshold = threshold_list[best_accuracy_idx]
    tree = train_decision_tree(train_data, attributes_with_domains, LOW, best_threshold)
    test_accuracy = eval_decision_tree(tree, test_data)

    print(tree)

    return best_threshold, test_accuracy

############################ main
# This code performs the following operations:
# 1) Load the data
# 2) create a list of attributes
# 3) create a dictionary that maps each attribute to its domain of values
# 4) split the data into train and test sets
# 5) train a decision tree while optimizing the threshold hyperparameter by
#    10-fold cross validation 
#####################################

#load data
data_frame = pd.read_csv("categorical_real_estate.csv")
data_frame = data_frame.fillna('NA')
print(data_frame)

# get attributes
attributes = list(data_frame.columns)
attributes.remove('class')

# create dictionary that maps each attribute to its domain of values
attributes_with_domains = {}
for attr in attributes:
    attributes_with_domains[attr] = set(data_frame[attr])


#split data in to train and test
train_data = data_frame.iloc[0:1000]
test_data = data_frame.iloc[1000:]

# perform 10-fold cross-validation
best_threshold, accuracy = cross_validation(train_data, test_data, attributes_with_domains, 10, [10,20,40,80,160])
print(f'Best threshold {best_threshold}: accuracy {accuracy}')

