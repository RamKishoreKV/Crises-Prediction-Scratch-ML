# -- coding: utf-8 --
"""
Created on Wed Dec  6 15:53:50 2023

@author: samue
"""
import csv, random
import numpy as np
from numpy import array, append, zeros
from math import log, ceil, e

CRISIS_TYPE = {
    0: 'Currency Crisis',
    1: 'Inflation Crisis',
    2: 'Banking Crisis',
    3: 'Systemic Crisis'
}

MODEL_TYPE = {
    0: 'Decision Tree',
    1: 'Naive Bayes',
    2: 'Logistic Regression',
    3: 'Ensemble'    
}

MEASURES = {
    0: 'Accuracy',
    1: 'Precision',
    2: 'Recall'    
}

class LogisticRegressionModel:
    def __init__(self, learning_rate=0.01, epochs=10000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.W = None
        self.bias = None

    def logloss(self, Y, sigmoid):
        logloss = -np.mean(Y * np.log(sigmoid + 1e-15) + (1 - Y) * np.log(1 - sigmoid + 1e-15))
        return logloss

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, xtrain, ytrain):
        minimum = -1e-4
        maximum = 1e-4

        self.W = np.random.uniform(minimum, maximum, xtrain.shape[1])
        self.bias = 0

        TrainLogloss = []

        for i in range(self.epochs):
            TrainZ = np.dot(xtrain, self.W) + self.bias
            sigmoidT = self.sigmoid(TrainZ)
            loss = self.logloss(ytrain, sigmoidT)
            TrainLogloss.append(loss)

            GradientW = np.dot(xtrain.T, (sigmoidT - ytrain)) / len(ytrain)
            GradientBias = np.mean(sigmoidT - ytrain)
            self.W -= self.learning_rate * GradientW
            self.bias -= self.learning_rate * GradientBias

            if i > 1 and abs(TrainLogloss[i] - TrainLogloss[i - 1]) < self.tolerance:
                break

    def predict(self, xvalid):
        ValidZ = np.dot(xvalid, self.W) + self.bias
        sigmoid_valid = self.sigmoid(ValidZ)
        predictions = (sigmoid_valid >= 0.5).astype(int)
        return predictions


class node:
    def __init__(self, val):
        self.val = val
        self.lchild = None
        self.rchild = None

    def getval(self):
        return self.val

    def left(self):
        return self.lchild

    def right(self):
        return self.rchild

    def isleaf(self):
        return self.rchild == self.lchild == None

    def addrchild(self, child):
        self.rchild = child

    def addlchild(self, child):
        self.lchild = child

    def _str_(self):
        return "val: " + str(self.val)


def probs(ytrain):
    out = []
    for i in set(ytrain):
        count = 0.0
        for j in ytrain:
            if j == i:
                count += 1.0
        prob = count / len(ytrain)
        out += [(int(i), prob)]
    return out


# calculates entropy
def bestfeature(x, y, features):
    data = append(x, y.reshape(x.shape[0], 1), 1)
    out = (features[0], 1.0)
    for i in features:
        pos = []
        neg = []
        for row in data:
            if row[i] < 0:
                pos += [row]
            else:
                neg += [row]
        numpos = len(pos)
        if numpos != 0:
            pos1 = 0.0
            pos2 = 0.0
            for row in pos:
                if row[-1] == 0:
                    pos1 += 1
                else:
                    pos2 += 1
            pos1 /= numpos
            pos2 /= numpos
            if pos2 > 0:
                pos2 = -log(pos2, 2) * pos2
            if pos1 > 0:
                pos1 = -log(pos1, 2) * pos1
            posterm = numpos / len(x) * (pos1 + pos2)
        else:
            posterm = 0

        numneg = len(neg)
        if numneg != 0:
            neg1 = 0.0
            neg2 = 0.0
            for row in neg:
                if row[-1] == 0:
                    neg1 += 1
                else:
                    neg2 += 1
            neg1 /= numneg
            neg2 /= numneg
            if neg1 > 0:
                neg1 = -log(neg1, 2) * neg1
            if neg2 > 0:
                neg2 = -log(neg2, 2) * neg2
            negterm = numneg / len(x) * (neg1 + neg2)
        else:
            negterm = 0

        entropy = posterm + negterm
        if entropy < out[1]:
            out = (i, entropy)
    return out[0]


def DTL(xtrain, ytrain, features, default):
    if len(xtrain) == 0:
        # produce a node with a list of probabilities from previous layer
        return node(default)
    elif len(set(ytrain)) <= 1:
        # only one class; return a node with that class
        return node(int(ytrain[0]))
    elif len(features) == 0:
        # make a list of probabilities for classes
        return node(probs(ytrain))
    else:
        # get probs
        prob = probs(ytrain)

        # get feature with lowest entropy to split on
        best = bestfeature(xtrain, ytrain, features)
        out = node(best)

        # we make a copy to avoid lower recursion levels affecting higher ones
        featurescopy = features.copy()
        featurescopy.remove(best)

        # put the data in one array to align the rows easier
        data = append(xtrain, ytrain.reshape(xtrain.shape[0], 1), 1)

        # split the data on the best feature
        lsubdata = []
        rsubdata = []
        for row in data:
            if row[best] < 0:
                lsubdata += [row]
            else:
                rsubdata += [row]
        lsubdata, rsubdata = array(lsubdata), array(rsubdata)

        # we want to split into x and y but doing so via splices doesn't work if
        # we have no data
        if len(lsubdata) != 0:
            lsubx = lsubdata[:, :-1]
            lsuby = lsubdata[:, -1]
        else:
            lsubx = []
            lsuby = []
        if len(rsubdata) != 0:
            rsubx = rsubdata[:, :-1]
            rsuby = rsubdata[:, -1]
        else:
            rsubx = []
            rsuby = []

        # left subtree
        subtree = DTL(lsubx, lsuby, featurescopy, prob)
        out.addlchild(subtree)
        # right subtree
        subtree = DTL(rsubx, rsuby, featurescopy, prob)
        out.addrchild(subtree)

        return out


# wrapper for DTL 
def myDT(xtrain, ytrain, xvalid):
    # get probs
    prob = probs(ytrain)

    # train tree
    dt = DTL(xtrain, ytrain, [i for i in range(len(xtrain[0]))], prob)

    # navigate tree for each row in xvalid
    yhat = []
    randgen = random.Random(0)
    for row in xvalid:
        curr = dt
        while not curr.isleaf():
            feature = curr.getval()
            if row[feature] < 0:
                curr = curr.left()
            else:
                curr = curr.right()
        leaf = curr.getval()
        if type(leaf) == int:
            yhat += [leaf]
        else:
            # pick at random
            guess = randgen.random()
            if guess <= leaf[0][1]:
                yhat += [leaf[0][0]]
            elif guess <= leaf[0][1] + leaf[1][1]:
                yhat += [leaf[1][0]]
            elif len(leaf) > 2:
                yhat += [leaf[2][0]]
            else:
                yhat += [leaf[1][0]]
    return yhat


def naivebayes(xtrain, ytrain, xvalid):
    # sort training data by class
    cdata = []
    classes = [0, 1]
    tdata = append(xtrain, ytrain.reshape(xtrain.shape[0], 1), 1)
    for c in classes:
        classn = []
        for row in tdata:
            if row[-1] == c:
                classn += [row]
        cdata += [classn]

    # get probability of y
    yprobs = []
    for c in cdata:
        yprobs += [len(c) / len(xtrain)]

    # get probabilities of x>mean|y
    probs = []
    for c in range(len(classes)):
        cprob = []
        for i in range(len(cdata[0][0]) - 1):
            count = 0
            for row in cdata[c]:
                if row[i] > 0:
                    count += 1
            if len(cdata[c]) != 0:
                cprob += [count / len(cdata[c])]
            else:
                cprob += [0]
        probs += [cprob]

    # classify the validation data
    vprobs = []
    for row in xvalidz:
        rho = 0
        vprob = [0] * len(classes)
        for c in range(len(classes)):
            for i, j in zip(row, probs[c]):
                if i > 0:
                    vprob[c] += log(max(2e-10, j))
                else:
                    vprob[c] += log(max(2e-10, 1 - j))
            vprob[c] += log(max(2e-10, yprobs[c]))
            rho += e ** vprob[c]
        vprobs += [[e ** (i - log(rho)) for i in vprob]]

    # select randomly using probabilities
    yhat = []
    for i in vprobs:
        r = randgen.random()
        for j in range(1, len(i) + 1):
            if r < sum(i[:j]) or j == len(i):
                yhat += [float(j - 1)]
                break
    return yhat


def ensemble(prediction_list, dtype=int):
    prediction_list = np.array(prediction_list, dtype=dtype)
    # print(prediction_list)
    minority_vote = []
    for i in range(len(prediction_list[0])):
        if (prediction_list[0][i] or prediction_list[1][i] or prediction_list[2][i]):
            minority_vote.append(1)
        else:
            minority_vote.append(0)
    # majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=prediction_list)
    # print(minority_vote)
    return minority_vote


def measures(yhat, yvalid):
    true_positives = np.sum((np.array(yhat) == 1) & (yvalid == 1))
    false_positives = np.sum((np.array(yhat) == 1) & (yvalid == 0))
    false_negatives = np.sum((np.array(yhat) == 0) & (yvalid == 1))

    predictions = np.sum(np.array(yhat) == yvalid)
    accuracy = predictions / len(yvalid)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1

    return accuracy, precision, recall

# actual body

file = open('Downloads/global_crisis_data.csv')
reader = csv.reader(file, delimiter=',')
colnames = []
data = []
bad = ['n/a', '', ' ', 'n.a.', 'NA', '  ', '#VALUE!', '#REF!']
firstrow = True
secondrow = False
for i in reader:
    if firstrow:
        colnames = [i[2]] + i[7:12] + i[16:17] + i[18:20] + i[21:] + [i[4]] + [i[6]]
        firstrow = False
        secondrow = True
    elif secondrow:
        secondrow = False
    else:
        row = [i[2]] + i[7:12] + i[16:17] + i[18:20] + i[21:] + [i[4]] + [i[6]]
        cleanrow = [0 if k in bad else k.replace(',', '') for k in row]
        if cleanrow[7] not in [0, 1]:
            cleanrow[7] = 1
        for i in range(4):
            if int(cleanrow[-(i + 1)]) > 1:
                cleanrow[-(i + 1)] = 1
        data += [cleanrow]
data = array(data)

countries = set(data[:, 0])

avgmeasures = zeros((4, 4, 3))

# split data by country
for country in countries:
    print(country)
    cdata = [i[1:] for i in data if i[0] == country]
    cdata = array(cdata, dtype=float)

    # shuffle and separate into training and validation
    randgen = random.Random(0)
    randgen.shuffle(cdata)
    vstart = ceil(2 * len(cdata) / 3)
    tdata = cdata[:vstart]
    vdata = cdata[vstart:]

    # split data into x and y
    xtrain = tdata[:, :-4]
    ytrain = tdata[:, -4:]
    xvalid = vdata[:, :-4]
    yvalid = vdata[:, -4:]

    # zero mean
    means = array([j.mean() for j in xtrain.transpose()])
    xtrainz = (xtrain.transpose() - means.reshape(len(means), 1)).transpose()
    xvalidz = (xvalid.transpose() - means.reshape(len(means), 1)).transpose()

    # function call for each crisis
    for k in range(4):
        ytra = ytrain.transpose()[k]
        yval = yvalid.transpose()[k]
        yhat_dt = myDT(xtrainz, ytra, xvalidz)
        yhat_nb = naivebayes(xtrainz, ytra, xvalidz)
        log_reg_model = LogisticRegressionModel()
        log_reg_model.fit(xtrainz, ytra)
        yhat_lr = log_reg_model.predict(xvalidz)
        yhat_ensemble = ensemble([yhat_dt, yhat_nb, yhat_lr])

        # calculate measures
        # print(CRISIS_TYPE[k])
        submeasures = []
        submeasures += [measures(yhat_dt, yval)]
        submeasures += [measures(yhat_nb, yval)]
        submeasures += [measures(yhat_lr, yval)]
        submeasures += [measures(yhat_ensemble, yval)]
        avgmeasures[k] += submeasures

avgmeasures = avgmeasures / 70
print(avgmeasures)

    # CRISIS_TYPE = {
    #     0: 'Currency Crisis',
    #     1: 'Inflation Crisis',
    #     2: 'Banking Crisis',
    #     3: 'Systemic Crisis'
    # }
    # print(accuracies_by_crisis)

# print("Accuracy")
# print(f'Decision Trees: {np.average(accuracy_dts)}')
