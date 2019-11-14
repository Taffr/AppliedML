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
    def newID3Node(self, label=None, attribute=None, entropy=None, samples=None, classes=[], nodes=None):
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

        print(nodeString)

        return

    # make the visualisation available
    def makeDotData(self):
        return self.dot

    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def findSplitAttr(self, stateEntropy, samples, target, attributes):
        attributeMapMapMap = {}
        # Create each map that should go in the MapMapMap
        for attr in attributes:
            attributeMapMapMap[attr] = {}
            for value in attributes[attr]:
                attributeMapMapMap[attr][str(value)] = {}
                for classification in self.classes:
                    attributeMapMapMap[attr][str(value)][classification] = 0

        for row in range(len(samples)):
            for col in range(len(samples[0])):  # What if missing values?
                for key in attributes.keys():
                    if samples[row][col] in attributes[key]:
                        attributeMapMapMap[key][samples[row][col]][target[row]] += 1
        print(attributeMapMapMap)
        attrEntropy = {}
        for attr in attributeMapMapMap:
            attrEntropy[attr] = 0
            for value in attributeMapMapMap[attr].keys():
                for classification in attributeMapMapMap[attr][value]:
                    attrEntropy[attr] = attributeMapMapMap[attr][value][classification] / sum(
                        attributeMapMapMap[attr][value].values()) * self.__entropy(attributeMapMapMap[attr][value])

        # Change this to make some more sense
        for attr in attrEntropy:
            print(stateEntropy - attrEntropy[attr])
        # return attributeEntropy

    def __entropy(self, classCount):
        entropy = 0
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
        self.classes = classes
        root = self.__buildTree(data, target, None, attributes)
        self.addNodeToGraph(root)
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
            root["label"] = lastClass
            root["entropy"] = 0.0
            root["samples"] = len(samples)
            root["classCount"] = {lastClass: len(target)}
            return root

        if len(attributes) < 1:
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
            return root
        else:
            classCounter = {}
            for classification in target:
                try:
                    classCounter[classification] += 1
                except:
                    classCounter[classification] = 1

            stateEntropy = self.__entropy(classCounter)

            root["entropy"] = stateEntropy
            root["classCount"] = classCounter
            root["samples"] = len(samples)

            # select target_attribute from highest IG
            attributeInfo = {}
            attributeInfo = self.findSplitAttr(stateEntropy, samples, target, attributes)

        # else:
        # givet nodes attribut ta dess class
        # hitta nya counters givet nodes attribut och klassifikation
        # räkna ut ny entropi
        # om entropin == 0
        # gör denna nod till ett löv
        # annars hitta det bästa nya attributet
        # gör denna noden till det attributet
        # rekrusivt...
        return root

    def predict(self, data, tree):
        predicted = list()

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted
