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

        print(nodeString)

        return node

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
        for row in range(len(samples)):
            for col in range(len(samples[0])):  # What if missing values?
                if self.attributeIndex[col] in attributes:
                    attributeMapMapMap[self.attributeIndex[col]][samples[row][col]] = {}  # Only adds present values in current samples

        for row in range(len(samples)):
            for col in range(len(samples[0])):  # What if missing values?
                if self.attributeIndex[col] in attributes:
                    try:
                        attributeMapMapMap[self.attributeIndex[col]][samples[row][col]][target[row]] += 1
                    except:
                        attributeMapMapMap[self.attributeIndex[col]][samples[row][col]][target[row]] = 1
        print(attributeMapMapMap)
        attrEntropy = {}
        total = len(samples)
        for attr in attributeMapMapMap:
            attrEntropy[attr] = 0
            for v in attributeMapMapMap[attr].keys():
                valueCount = sum(attributeMapMapMap[attr][v].values())
                attrEntropy[attr] += (valueCount / total) * self.__entropy(attributeMapMapMap[attr][v])

        # Change this to make some more sense
        bestAttribute = ""
        highestIG = 0

        for attr in attrEntropy:
            if (stateEntropy - attrEntropy[attr]) > highestIG:
                bestAttribute = attr
                highestIG = stateEntropy - attrEntropy[attr]
        return bestAttribute

    def __entropy(self, classCount):
        entropy = 0
        total = sum(classCount.values())
        for key in classCount.keys():
            probability = classCount[key] / total
            entropy += probability * math.log(probability, 2)
        return -entropy

    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):
        self.classes = classes
        self.attributeIndex = {}
        self.indexOfAttribute = {}
        index = 0
        for key in attributes:
            self.indexOfAttribute[key] = index
            self.attributeIndex[index] = key
            index += 1
        root = self.__buildTree(data, target, attributes)
        self.addNodeToGraph(root)

        print('------- ROOT\'S NODES -------')
        for n in root['nodes']:
            print(n)
            print(' ** NODES **')
            for nn in n['nodes']:
                print(nn)
            print()
        print('----------------------------')
        return root

    def __buildTree(self, samples, target, attributes):
        root = self.newID3Node()
        allSameClass = True
        lastClass = ""
        for index in range(len(target)):
            if lastClass != target[index] and lastClass != "":
                allSameClass = False
                break
            lastClass = target[index]

        if allSameClass:
            root["label"] = lastClass
            root["entropy"] = 0.0
            root["samples"] = len(target)
            self.addNodeToGraph(root, self.nodeCounter)
            return root

        if len(attributes) <= 1:
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
            self.addNodeToGraph(root, self.nodeCounter)
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
            root["samples"] = len(samples)
            # select target_attribute from highest IG
            bestAttribute = self.findSplitAttr(stateEntropy, samples, target, attributes)
            root["attribute"] = bestAttribute
            root["classes"] = attributes[bestAttribute]

            subSets = {}
            for value in attributes[bestAttribute]:
                subSets[value] = {"samples": [], "target": []}
                for rowIndex in range(len(samples)):
                    if value == samples[rowIndex][self.indexOfAttribute[bestAttribute]]:
                        subSets[value]["samples"].append(samples[rowIndex])
                        subSets[value]["target"].append(target[rowIndex])

            newAttributes = {}
            for attribute in attributes.keys():
                if attribute != bestAttribute:
                    newAttributes[attribute] = attributes[attribute]

            print(newAttributes)
            for v in subSets:
                if len(subSets[v]["samples"]) < 1:
                    counter = {}
                    for c in subSets[v]["target"]:
                        try:
                            counter[c] += 1
                        except:
                            counter[c] = 1
                    maxCount = 0
                    maxLabel = ""
                    for key in counter:
                        if counter[key].value() >= maxCount:
                            maxCount = counter[key].value()
                            maxLabel = key
                    self.addNodeToGraph(self.newID3Node(maxLabel), self.nodeCounter)
                else:
                    self.__buildTree(subSets[v]["samples"], subSets[v]["target"], newAttributes)
        self.addNodeToGraph(root, self.nodeCounter)
        return root
    def predict(self, data, tree):
        predicted = list()

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted