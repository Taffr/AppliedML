from collections import Counter
from graphviz import Digraph
import math


class ID3DecisionTreeClassifier:

    def __init__(self, minSamplesLeaf=1, minSamplesSplit=2):

        self.nodeCounter = 0

        # the graph to visualise the tree
        self.dot = Digraph(comment='The Decision Tree')
        self.target = None
        self.data = None
        # suggested attributes of the classifier to handle training parameters
        self.minSamplesLeaf = minSamplesLeaf
        self.minSamplesSplit = minSamplesSplit
        self.usedAttributes = []

    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def newID3Node(self, label=None, attribute=None, entropy=None, samples=None, classes=[], nodes=[]):
        node = {'id': self.nodeCounter, 'label': label, 'attribute': attribute, 'entropy': entropy, 'samples': samples,
                'classes': classes, 'nodes': nodes}

        self.nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def addNodeToGraph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if (node[k] != None) and (k != 'nodes'):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])




        # print(nodeString)

        return

    # make the visualisation available
    def makeDotData(self):
        return self.dot

    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def findSplitAttr(self, stateEntropy, samples, target, attributes):
        # Attributes is e.g:
        # {'color': ['y', 'g'], 'size': ['s', 'l'], 'shape': ['r', 'i']}

        # Samples is e.g.:
        # [('y', 's', 'r'), ('y', 's', 'r'), ('g', 's', 'i'), ('g', 'l', 'i'), ('y', 'l', 'r'), ('y', 's', 'r'), ('y', 's', 'r'), ('y', 's', 'r'), ('g', 's', 'r'), ('y', 'l', 'r'), ('y', 'l', 'r'), ('y', 'l', 'r'), ('y', 'l', 'r'), ('y', 'l', 'r'), ('y', 's', 'i'), ('y', 'l', 'i')]

        # print('***** SAMPLES ******')
        # print(samples)

        attributeMapMapMap = {}
        # Create each map that should go in the MapMapMap
        for attr in attributes:
            attributeMapMapMap[attr] = {}
            for value in attributes[attr]:
                attributeMapMapMap[attr][str(value)] = {}
                # for classification in self.classes:
                #     attributeMapMapMap[attr][str(value)][classification] = 0

        for row in range(len(samples)):
            for col in range(len(samples[0])):  # What if missing values?
                for key in attributes.keys():
                    if samples[row][col] in attributes[key]:
                        if target[row] in attributeMapMapMap[key][samples[row][col]]:
                            attributeMapMapMap[key][samples[row][col]][target[row]] += 1
                        else:
                            attributeMapMapMap[key][samples[row][col]][target[row]] = 1
        print('***** attributeMapMapMap ******')
        print(attributeMapMapMap)
        print('**********************')
        n_samples = len(samples)
        attrEntropy = {}
        for attr in attributeMapMapMap:
            # color
            # attrEntropy[attr] = stateEntropy
            attribute_entropy = stateEntropy
            for value in attributeMapMapMap[attr]:
                # y, g
                value_entropy = self.__entropy(attributeMapMapMap[attr][value])
                n_values = sum(attributeMapMapMap[attr][value].values())
                attribute_entropy -= n_values/n_samples * value_entropy

            attrEntropy[attr] = attribute_entropy

        # for attr in attributeMapMapMap:
        #     attrEntropy[attr] = 0
        #     for value in attributeMapMapMap[attr].keys():
        #         for classification in attributeMapMapMap[attr][value]:
        #             attrEntropy[attr] = attributeMapMapMap[attr][value][classification] / sum(
        #                 attributeMapMapMap[attr][value].values()) * self.__entropy(attributeMapMapMap[attr][value])

        # print('***** attrEntropy ******')
        # print(attrEntropy)
        # print('**********************')

        # Pick attribute with highest IG.
        highest_IG = -1
        best_attribute = ''
        index = 0
        best_index = -1
        for attr in attrEntropy:
            if attrEntropy[attr] > highest_IG:
                highest_IG = attrEntropy[attr]
                best_attribute = attr
                best_index = index
            index += 1

        return {'i': best_index, 'attribute': best_attribute, 'entropy': attrEntropy[best_attribute], 'IG': highest_IG, 'values': attributeMapMapMap[best_attribute]}

    def __entropy(self, classCount):
        # print('****** CLASS COUNT ******')
        # print(classCount)
        # print('**************')
        entropy = 0
        if len(classCount) is 1:
            return 0

        total = sum(classCount.values())
        for key in classCount.keys():
            probability = classCount[key] / total
            entropy += probability * math.log(probability, 2)
        return -entropy

    # def __findHighestInfo(self, entropy, usedAttributes):

    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):
        # Add the different target classifications to a list

        # fill in something more sensible here... root should become the output of the recursive tree creation
        # ID3 Algorithm will perform following tasks recursively:

        # Create a root node for the tree

        # If all examples are positive, return leaf node ‘positive’

        # Else if all examples are negative, return leaf node ‘negative’

        # Calculate the entropy of current state E(S)

        # For each attribute, calculate the entropy with respect to the attribute ‘A’ denoted by E(S, A)

        # Select the attribute which has the maximum value of IG(S, A) and split the current (parent) node on the selected attribute

        # Remove the attribute that offers highest IG from the set of attributes

        self.attribute_index = {}
        index = 0
        for key in attributes:
            self.attribute_index[key] = index
            index += 1

        self.classes = classes
        root = self.__buildTree(data, target, None, attributes)
        self.addNodeToGraph(root)

        print('------- ROOT\'S NODES -------')
        for n in root['nodes']:
            print(n)
            print(' ** NODES **')
            for nn in n['nodes']:
                print(nn)
            print()
        print('----------------------------')



        # self.usedAttributes.append(attributeName)
        return root

    def __buildTree(self, samples, target, targetAttribute, attributes):
        root = self.newID3Node()

        allSameClass = True
        lastClass = ""
        for classification in target:
            if lastClass != classification and lastClass != "":
                allSameClass = False
                break
            lastClass = classification

        if allSameClass:
            print('\nXXXXXXXXXXXXXXXX ALL SAME CLASS XXXXXXXXXXXXX\n')
            root["label"] = lastClass
            root["entropy"] = 0.0
            root["samples"] = len(target)
            return root

        if len(attributes) == 0:
            print('\nXXXXXXXXXXXXXXXX NO ATTRIBUTES LEFT XXXXXXXXXXXXX\n')
            classCounter = {}
            for classification in target:
                try:
                    classCounter[classification] += 1
                except:
                    classCounter[classification] = 1
            maxCount = list(classCounter.values())[0]
            mostCommon = list(classCounter.keys())[0]
            for key in classCounter:
                if classCounter[key] > maxCount:
                    maxCount = classCounter[key]
                    mostCommon = key

            root["label"] = mostCommon
            root["entropy"] = 0.0
            root["samples"] = sum(classCounter.values())
            root["classCount"] = classCounter
            print(root)
            return root
        else:
            classCounter = {}
            for classification in target:
                try:
                    classCounter[classification] += 1
                except:
                    classCounter[classification] = 1

            # print('***** CLASS COUNTER *******')
            # print(classCounter)
            # print('**************')

            stateEntropy = self.__entropy(classCounter)
            # print("stateEntropy: ", stateEntropy)

            root["entropy"] = stateEntropy
            root["classCount"] = classCounter
            root["samples"] = len(samples)



            # Root here contains e.g.
            # {
            #   'id': 0,
            #   'label': None,
            #   'attribute': None,
            #   'entropy': 0.9886994082884974,
            #   'samples': 16,
            #   'classes': [],
            #   'nodes': None,
            #   'classCount': {'+': 9, '-': 7}
            # }

            # select target_attribute from highest IG
            split_attribute_dict = self.findSplitAttr(stateEntropy, samples, target, attributes)
            print('******* split_attribute_dict ********')
            print(split_attribute_dict)
            print('***************')

            targetAttribute = split_attribute_dict['attribute']
            root['attribute'] = targetAttribute
            root['classes'] = attributes[targetAttribute]

            print('***** root in id3 *******')
            print(root)
            print('**************')

            # split_attribute_index = split_attribute_dict['i']
            split_attribute_index = self.attribute_index[targetAttribute]
            print('split_attribute_index', split_attribute_index)
            print('split_attribute_values', split_attribute_dict['values'])



            for value in split_attribute_dict['values']:
                value_node = self.newID3Node()

                sub_samples = []
                sub_targets = []

                for i in range(len(samples)):
                    s = samples[i][split_attribute_index]
                    if s == value:
                        sub_samples.append(samples[i])
                        sub_targets.append(target[i])

                print('******* subs ********')
                print(sub_samples)
                print('----')
                print(sub_targets)
                print('***************')

                # for s in samples:
                #     if s[split_attribute_index] == value:
                #         sub_samples.append(s)

                if len(sub_samples) == 0:
                    # Below this new branch add a leaf node with label  = most common class value in Samples
                    print('***** No sub samples ******')
                    # leaf = self.newID3Node()
                    classCounter = {}
                    for classification in target:
                        try:
                            classCounter[classification] += 1
                        except:
                            classCounter[classification] = 1



                    maxCount = list(classCounter.values())[0]
                    mostCommon = list(classCounter.keys())[0]
                    for key in classCounter:
                        if classCounter[key] > maxCount:
                            maxCount = classCounter[key]
                            mostCommon = key

                    print()
                    print('maxCount', maxCount)
                    print('mostCommon', mostCommon)

                    value_node["label"] = mostCommon
                    value_node["entropy"] = 0.0
                    value_node["samples"] = sum(classCounter.values())
                    value_node["classCount"] = classCounter

                    value_node["attribute"] = 'TEMP: leaf'

                    # print("*** ROOT NODES: ***")
                    # print(root['nodes'])

                    print(1)


                    root['nodes'].append(value_node)
                    # return root
                else:
                    print(2)
                    # Below this new branch add the subtree ID3 (Samples(v), A, Attributes/{A})
                    print('****** Sub sample exists *******')
                    # del attributes[targetAttribute]
                    # print("POPS ATTRIBUTE: ", targetAttribute)
                    # attributes.pop(targetAttribute, None)

                    sub_attributes = {}
                    for attr in attributes:
                        if attr != targetAttribute:
                            sub_attributes[attr] = attributes[attr]

                    # print('sub_attributes', sub_attributes)
                    print('targetAttribute', targetAttribute)
                    print('\n*** CALLS __buildTree ***')
                    n = self.__buildTree(sub_samples, sub_targets, targetAttribute, sub_attributes)
                    root['nodes'].append(n)
            return root


                # self.__buildTree(samples, target, targetAttribute, attributes)

        # else:
        # givet nodes attribut ta dess class
        # hitta nya counters givet nodes attribut och klassifikation
        # räkna ut ny entropi
        # om entropin == 0
        # gör denna nod till ett löv
        # annars hitta det bästa nya attributet
        # gör denna noden till det attributet
        # rekrusivt...

        print('\n************ THE ROOT ***********')
        print(root)
        print('**************\n')
        return root


    def predict(self, data, tree):
        predicted = list()

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted
