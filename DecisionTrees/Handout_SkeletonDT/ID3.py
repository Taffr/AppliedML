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
    def newID3Node(self, label=None, attribute=None, entropy=None, samples=None, classCount=None, nodes=None):
        node = {'id': self.nodeCounter, 'label': label, 'attribute': attribute, 'entropy': entropy, 'samples': samples,
                'classCounts': classCount, 'nodes': nodes}

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
    def findSplitAttr(self):

        # Change this to make some more sense
        return None

    def __calculateRoot(self):
        self.classifications = []
        for value in self.target:
            if value not in self.classifications:
                self.classifications.append(value)

        # Adds all the classifications to map detailing their freq.
        setEntropy = 0
        classificationMap = {}
        for classification in self.classifications:
            classificationMap[classification] = 0
        total = 0
        for value in self.target:
            classificationMap[value] += 1
            total += 1

        for classification in classificationMap:
            setEntropy += self.__entropy(classificationMap[classification] / total)

        return self.__findBestAttribute(setEntropy)

    def __entropy(self, probability):
        return - probability * math.log(probability, 2)

    def __attributeInformationGain(self, mapOfValues, setEntropy):
        informationGain = setEntropy
        attributeEntropy = 0
        dataTotal = len(self.data)
        attrClassInfo = {}

        for attrClass in mapOfValues:
            attrClassInfo[attrClass] = {}
            attrClassInfo[attrClass]["P(" + attrClass + ")"] = sum(mapOfValues[attrClass].values()) / dataTotal
            for classification in self.classifications:
                attrClassInfo[attrClass]["P(" + classification + ")"] = mapOfValues[attrClass][classification] / sum(mapOfValues[attrClass].values())
                pAttrClass = attrClassInfo[attrClass]["P(" + attrClass + ")"]
                pClassification = attrClassInfo[attrClass]["P(" + classification + ")"]
                attributeEntropy += pAttrClass * self.__entropy(pClassification)

        return informationGain - attributeEntropy

    def __findBestAttribute(self, setEntropy):
        # Check the length of the row => number of attributes (What do if first row is missing?)
        numberOfAttributes = len(self.data[0])
        # Map of Map of Map for each attribute
        attributeMapMapMap = {}
        # Create each map that should go in the MapMap
        for i in range(numberOfAttributes):
            attributeMapMapMap["attr" + str(i)] = {}

        # Create each map that should go in the MapMapMap
        for rowIndex in range(len(self.data)):
            for x in range(numberOfAttributes):
                # Create a new map for each classification of attribute
                attributeMapMapMap["attr" + str(x)][str(self.data[rowIndex][x])] = {}
                # Add classifications
                for classification in self.classifications:
                    attributeMapMapMap["attr" + str(x)][str(self.data[rowIndex][x])][classification] = 0

        for row in range(len(self.target)):
            for i in range(numberOfAttributes):
                for key in attributeMapMapMap["attr" + str(i)][str(self.data[row][i])].keys():
                    if key == self.target[row]:
                        attributeMapMapMap["attr" + str(i)][str(self.data[row][i])][key] += 1

        attributeScores = {}
        for attr in attributeMapMapMap:
            attributeScores[attr] = self.__attributeInformationGain(attributeMapMapMap[attr], setEntropy)

        bestAttributeKey = ""
        bestAttributeIndex = 0
        index = 0
        for key in attributeScores:
            try:
                if attributeScores[key] > attributeScores[bestAttributeKey]:
                    bestAttributeKey = key
                    bestAttributeIndex = index
            except:
                bestAttributeKey = key
            index += 1

        return {"attr" + str(bestAttributeIndex): attributeMapMapMap["attr" + str(bestAttributeIndex)], "e": setEntropy - attributeScores[bestAttributeKey]}


    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):
        self.data = data
        self.target = target
        for index in range(len(self.data)):
            print(self.data[index], self.target[index])
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

        # Repeat until we run out of all attributes, or the decision tree has all leaf nodes.
        attributeInfo = self.__calculateRoot()
        attributeName = list(attributeInfo.keys())[0]
        entropy = attributeInfo["e"]
        samples = 0
        classCount = len(list(attributeInfo[attributeName].keys()))
        for key in attributeInfo[attributeName]:
            samples += sum(attributeInfo[attributeName][key].values())
        root = self.newID3Node(None, attributeName, entropy, samples, classCount)
        self.addNodeToGraph(root)

        return root

    def predict(self, data, tree):
        predicted = list()

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted
