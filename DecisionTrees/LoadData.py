from sklearn import datasets, svm, metrics, tree
from os import system
import matplotlib.pyplot as plt
digits = datasets.load_digits()
split = int(0.7*len(digits.data))

# Splitting data
trainingFeatures = digits.data[:split]
testFeatures = digits.data[split:]
trainingLabels = digits.target[:split]
testLabels = digits.target[split:]
print(trainingFeatures)
# Model to be used as a classifier
#classifier = svm.D(gamma=0.001)
classifier = tree.DecisionTreeClassifier()
# Train the data
classifier.fit(trainingFeatures, trainingLabels)

# Using the model
predicted = classifier.predict(testFeatures)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(testLabels, predicted)))
print("Confusion matrix:\n%s"
      % metrics.confusion_matrix(testLabels, predicted))
# Gives accuracy avg =~ 0.79
# Gives recall avg =~ 0.79
# Gives f-score avg =~ 0.79
# Comparing the confusion matrix with that from the SVC shows that the Tree had much more miss-classifications
dotFile = open("dtree.dot", 'w');
tree.export_graphviz(classifier, out_file=dotFile)
dotFile.close()
system("dot -Tpng dtree.dot -o dTree.png") # Export as dTree.png (No good labels)


### Same thing but change the number of leaves
classifierMod = tree.DecisionTreeClassifier(min_samples_leaf=50);
# Train the data
classifierMod.fit(trainingFeatures, trainingLabels)

# Using the model
predicted = classifierMod.predict(testFeatures)

print("Classification report for classifier %s:\n%s\n"
      % (classifierMod, metrics.classification_report(testLabels, predicted)))
print("Confusion matrix:\n%s"
      % metrics.confusion_matrix(testLabels, predicted))
# Gives accuracy avg =~ 0.72
# Gives recall avg =~ 0.71
# Gives f-score avg =~ 0.71
# More samples/leaf -> Less leafs -> smaller tree -> more miss-classifications
dotFile = open("dtreeMod.dot", 'w');
tree.export_graphviz(classifierMod, out_file=dotFile)
dotFile.close()
system("dot -Tpng dtreeMod.dot -o dTreeMod.png") # Export as dTreeMod.png (No good labels)

