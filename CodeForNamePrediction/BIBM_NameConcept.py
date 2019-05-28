
"""""""""""
Created on Sun June 3 16:25:30 2018

Last Modified on Fri Nov 30 20:30:30 2018 

@author: Fengbo Zheng
"""

# Part 1 Word Embedding
from gensim.models import Word2Vec

wordSequence = []
file4 = open('File_Used_To_Train_WordEmbedding.txt','rb+')
for linesd in file4:
    linesdd = linesd.replace(","," , ")
    listq = []
    wordSequenceString = linesdd.split()
    for worditem in wordSequenceString:
        listq.append(worditem.lower())
    wordSequence.append(listq)
file4.close()

#W2V and some experiment to check the performance of generated word embedding
#Word2Vec
model = Word2Vec(wordSequence, size = 125, min_count=1, window = 3)   #window = 3,5,7
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['injury'])
# save model
model.save('modelW2V(17).bin')
# load model
model = Word2Vec.load('modelW2V(all18Window7).bin')
print(model)
# check similar vector
W1 = "finger"
model.wv.most_similar(positive = W1)
# check linear subrelation 
#result = model.wv.most_similar(positive = ["women","king"], negative = ["man"])
#print("{}: {:.4f}".format(*result[0]))
# similarity between two vector
#similarity = model.wv.similarity('woman', 'man')


# Part 2 ID map to Word, Word map to ID
import numpy as np

dicWordAndW2Vector = {}
words = list(model.wv.vocab)
for singleWords in words:
    representation = model[singleWords]    #DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    dicWordAndW2Vector[singleWords] = representation

i = 1
IDs = set()
while i <= len(words):
    IDs.add(i)
    i = i+1

#Word map to ID
outputFile = open("Word_And_ID.txt","w")
for SingleWords in words:
    a = IDs.pop()
    outputFile.write(SingleWords+" "+str(a)+"\n")
outputFile.close()

file8 = open("Word_And_ID.txt","rb+")
wordToID =  {}
IDToWord = {}
for line in file8:
    linee = line.split()
    wordToID[linee[0]] = int(linee[1])
    IDToWord[int(linee[1])] = linee[0]
file8.close()


# Part 3 Generate Word Embedding Weight Matrix
Word_Embed_Matrix = np.array([],dtype = "float32")  
ZeroVector = np.zeros((125,), dtype = "float32")
Word_Embed_Matrix = np.append(Word_Embed_Matrix, ZeroVector)
i = 1
while i <=len(words):
    Word_Embed_Matrix = np.append(Word_Embed_Matrix, dicWordAndW2Vector.get(IDToWord.get(i)))
    i = i+1
Word_Embed_Matrix = Word_Embed_Matrix.reshape(-1,125)


np.save("Word_Embed_Matrix", Word_Embed_Matrix)

Word_Embed_Matrix = np.load("Word_Embed_Matrix.npy")


# Part 4 Generate Labled Train/Test Data
import random

def error():
    return 0
LabelZeroNum = 5
OneWordsConcept = []
file7 = open("Test/Train_Concept_Name","rb+")
output7 = open("Labeled_Test/Train_Data","w")
m = 0
for lines in file7:
    print m
    m = m+1
    liness = lines.replace(","," , ")
    line = liness.split()
    originalOrder = []
    labeledOneOrder = []
    for eachItem in line:
        if wordToID.get(eachItem.lower(),"default") != "default":
            originalOrder.append(str(wordToID.get(eachItem.lower())))
            labeledOneOrder.append(str(wordToID.get(eachItem.lower())))
        else:
            error()#return eachItem.lower()#"error!error!error!error!error!error!error!error!"            
    zeroNumber = 45- len(originalOrder)  #eg 5
    i = 0
    while i <zeroNumber:
        output7.write("0,")
        i = i+1
    output7.write(",".join(originalOrder)+"\t"+"1"+"\n")
    if len(set(originalOrder)) >1:
        j = 0
        while j < LabelZeroNum:
            random.shuffle(originalOrder)
            if originalOrder != labeledOneOrder:
                k = 0
                while k <zeroNumber:
                    output7.write("0,")
                    k = k+1
                output7.write(",".join(originalOrder)+"\t"+"0"+"\n")
                j = j+1
    else:
        #print originalOrder
        OneWordsConcept.append(originalOrder)
output7.close()


# Part 5 Generate Input Matrix and Label Matrix for Train/Test Data
import numpy as np

TrainWordSequence = np.empty(shape = (1753513,45))  #numpy arrary ([[],[],[]])
TrainWordSequence = TrainWordSequence.astype(int)
TrainLabels = np.empty(shape = (1753513,))    #only one row  e.g. labels = array([1,1,1,1,1,0,0,0,0,0])
TrainLabels = TrainLabels.astype(int)

file4 = open('Labeled_Train_Data.txt','rb+')
i = 0
for linesd in file4:
    lined = linesd.split("\n")[0]
    wordSequenceString = lined.split("\t")[0]
    lableItem = lined.split("\t")[1]
    TrainLabels[i] = int(lableItem)
    wordSequenceItem = []
    wordSequenceList = wordSequenceString.split(",")
    for item in wordSequenceList:
        wordSequenceItem.append(int(item))
    TrainWordSequence[i] = wordSequenceItem
    i = i+1
file4.close()


TestWordSequence = np.empty(shape = (1784744,45))  #numpy arrary ([[],[],[]])
TestWordSequence = TestWordSequence.astype(int)
TestLabels = np.empty(shape = (1784744,))    #only one row  e.g. labels = array([1,1,1,1,1,0,0,0,0,0])
TestLabels = TestLabels.astype(int)

file4 = open('Labeled_Test_Data.txt','rb+')
i = 0
for linesd in file4:
    lined = linesd.split("\n")[0]
    wordSequenceString = lined.split("\t")[0]
    lableItem = lined.split("\t")[1]
    TestLabels[i] = int(lableItem)
    wordSequenceItem = []
    wordSequenceList = wordSequenceString.split(",")
    for item in wordSequenceList:
        wordSequenceItem.append(int(item))
    TestWordSequence[i] = wordSequenceItem
    i = i+1
file4.close()


# Part 6 Neural Network
Word_Embed_Matrix = np.load("Word_Embed_Matrix.npy")

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


class_weight = {0 : 1, 1 : 5}

#Model Replacement

##1
model = Sequential()
model.add(Embedding(76889,125, weights = [Word_Embed_Matrix], input_length = 45, trainable = False))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
model.fit(TrainWordSequence, TrainLabels, epochs=3, verbose=1, class_weight = class_weight)
loss, accuracy = model.evaluate(TestWordSequence, TestLabels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
model.save("Model_Name.h5")


##2
model = Sequential()
model.add(Embedding(76889,125, weights = [Word_Embed_Matrix], input_length = 45, trainable = False))
model.add(LSTM(100))
model.add(Dense(1,activation = "sigmoid"))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
model.fit(TrainWordSequence, TrainLabels, epochs=3, verbose=1, batch_size = 64, class_weight = class_weight)
scores = model.evaluate(TestWordSequence, TestLabels, verbose=2)
print ("Accuracy: %.2f%%" % (scores[1]*100))
model.save("Model_Name.h5")

##3
model = Sequential()
model.add(Embedding(76889,125, weights = [Word_Embed_Matrix], input_length = 45, trainable = False))
model.add(Conv1D(filters = 32, kernel_size = 3, padding = "same", activation = "relu"))
model.add(MaxPooling1D(pool_size = 2))
model.add(LSTM(100))
model.add(Dense(1, activation = "sigmoid"))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
model.fit(TrainWordSequence, TrainLabels, epochs=3, verbose=1, batch_size = 64, class_weight = class_weight)
scores = model.evaluate(TestWordSequence, TestLabels, verbose=2)
print ("Accuracy: %.2f%%" % (scores[1]*100))
model.save("Model_Name.h5")


#save model
model.save("Model_Name.h5")


#reuse trained model
from keras.models import load_model
model = load_model("Model_Name.h5")


plot_model(model, to_file='model_plotLSTM(Noshape).png', show_layer_names=True)

# Part 7 Predict (complete) concept names given bag of words  Top1
import itertools as it
import numpy as np

def sec_elem(s):
    return s[0] 

file1 = open("Concept_Name_Bag_To_Predict.txt","rb+")
m = 0
n = 0
for lines in file1:
    output2 = open("Predication_Result.txt","a")
    m = m+1
    linesss = lines.replace(","," , ")
    readyForPredication = []
    line = linesss.split()
    lineLength = len(line)
    zeroNum = 45-lineLength
    candidate = []
    for i in range(0,len(line)):
        candidate.append(str(wordToID.get(line[i].lower())))
    if len(candidate) <= 9: 
        allcombination = list(it.permutations(candidate))
        for eachComb in allcombination:
            j = 0
            stringForPredict = ""
            while j <zeroNum:
                stringForPredict = stringForPredict+"0,"
                j = j+1
            stringForPredict = stringForPredict +",".join(eachComb)
            readyForPredication.append(stringForPredict)
        #generate predication input
        matrixLength = len(readyForPredication)
        predictSequence = np.empty(shape = ((matrixLength),45))  #numpy arrary ([[],[],[]])
        predictSequence = predictSequence.astype(int)
        k = 0
        for row in readyForPredication:
            newRow = []
            rowList = row.split(",")
            for eachItem in rowList:
                newRow.append(int(eachItem))
            predictSequence[k] = newRow
            k = k+1
        predictLabel = model.predict_proba(predictSequence)
        tupleList = []  # a list of tuples
        for l in range(len(predictLabel)):
            y = round(predictLabel[l],2)
            print y
            newTuple = (y, predictSequence[l])
            tupleList.append(newTuple)
        sortedTupleList = sorted(tupleList, key=sec_elem)
        LastItm = sortedTupleList[len(sortedTupleList)-1]
        maxConfidence = LastItm[0]
        readyForInterpretation = []
        for items in sortedTupleList:
            if items[0] == maxConfidence:
                readyForInterpretation.append(items[1])
        InterpretedList = []
        for conceptNamess in readyForInterpretation:
            interpretedSequence = ""
            for each in conceptNamess:
                if each != 0:
                    interpretedSequence = interpretedSequence + IDToWord.get(int(each))+" "
            InterpretedList.append(interpretedSequence)
        if (" ".join(line)).lower() in InterpretedList:
            n = n+1
        output2.write(" ".join(line)+"\n")
        for i in range(0,(len(InterpretedList)-1)):
            output2.write(InterpretedList[i]+"\n")
        output2.write(InterpretedList[len(InterpretedList)-1]+"\n"+"\n"+"\n")
        output2.close()


#predict (complete)  Top1 for Dr.Cui's earlier work
import itertools as it
import numpy as np

def sec_elem(s):
    return s[0] 

file1 = open("CUI_Earlier_SetOfWord.txt","rb+")

for lines in file1:
    output2 = open("PredicationResultForCuiEarlierWord.txt","a")
    liness = lines.split("\n")[0]
    readyForPredication = []
    line = liness.split("\t")
    lineLength = len(line)
    zeroNum = 45-lineLength
    candidate = []
    for i in range(0,len(line)):
        print [line[i]]
        candidate.append(str(wordToID.get(line[i].lower())))
    allcombination = list(it.permutations(candidate))
    for eachComb in allcombination:
        j = 0
        stringForPredict = ""
        while j <zeroNum:
            stringForPredict = stringForPredict+"0,"
            j = j+1
        stringForPredict = stringForPredict +",".join(eachComb)
        readyForPredication.append(stringForPredict)
    #generate predication input
    matrixLength = len(readyForPredication)
    predictSequence = np.empty(shape = ((matrixLength),45))  #numpy arrary ([[],[],[]])
    predictSequence = predictSequence.astype(int)
    k = 0
    for row in readyForPredication:
        newRow = []
        rowList = row.split(",")
        for eachItem in rowList:
            newRow.append(int(eachItem))
        predictSequence[k] = newRow
        k = k+1
    predictLabel = model.predict_proba(predictSequence)
    tupleList = []  # a list of tuples
    for l in range(len(predictLabel)):
        y = round(predictLabel[l],2)
        print y
        newTuple = (y, predictSequence[l])
        tupleList.append(newTuple)
    sortedTupleList = sorted(tupleList, key=sec_elem)
    LastItm = sortedTupleList[len(sortedTupleList)-1]
    maxConfidence = LastItm[0]
    readyForInterpretation = []
    for items in sortedTupleList:
        if items[0] == maxConfidence:
            readyForInterpretation.append(items[1])
    InterpretedList = []
    for conceptNamess in readyForInterpretation:
        interpretedSequence = ""
        for each in conceptNamess:
            if each != 0:
                interpretedSequence = interpretedSequence + IDToWord.get(int(each))+" "
        InterpretedList.append(interpretedSequence)
    for i in range(0,(len(InterpretedList)-1)):
        output2.write(InterpretedList[i]+"\n")
    output2.write(InterpretedList[len(InterpretedList)-1]+"\n"+"\n"+"\n")
    output2.close()



# Part 8 Two-Step Filter

#Step1 Use Concetp Name which is similar in content to provide suggestions on sequencing order
bagsInTrainData = set()
inputFile = open("Train_Concept_Name","rb+")
for lines in inputFile:
    liness = lines.replace(","," , ")
    newLineList =" ".join([x.lower() for x in liness.split()])
    bagsInTrainData.add(newLineList)
inputFile.close()

file2 = open("Predication_Result","rb+")
#file2 = open("ProcessedTest.txt","rb+")
bag = []
readyforProcessed= []
for lines in file2:
    if bag.count("\n") != 2:
        bag.append(lines)
    else:
        bag.remove("\n")
        bag.remove("\n")
        readyforProcessed.append(bag)
        bag = []
        bag.append(lines)

def findSimilarBags(inputSequence):
    maximumSim = 0
    similarBag = ""
    inputList = [x.lower() for x in inputSequence.split()]
    for concepts in bagsInTrainData:
        candidateList = concepts.split()
        similarityI = float(len(set(inputList) & set(candidateList)))/float(len(set(inputList+candidateList)))
        if similarityI > maximumSim:
            maximumSim = similarityI
            similarBag = concepts
    return similarBag    


def levenshtein(a,b):
   # "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
        
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
            
    return current[n]


file3 = open("Filtered1_Predication_Result","w")      
for eachCandidateSet in readyforProcessed:
    if len(set(eachCandidateSet))>2:
        file3.write(eachCandidateSet[0])
        referencedConceptName = findSimilarBags(eachCandidateSet[0])
        referencedList = referencedConceptName.split()
        SimilarityList = []
        for i in range(1,len(eachCandidateSet)):
            predicatedSequence = eachCandidateSet[i]
            predicatedList = predicatedSequence.split()
            SimilarityII = levenshtein(referencedList,predicatedList)
            SimilarityList.append((SimilarityII, predicatedSequence))
        SimilarityList.sort(key=lambda tup: tup[0])
        Best = SimilarityList[0][0]
        for eachOne in SimilarityList:
            if eachOne[0] == Best:
                file3.write(eachOne[1])
        file3.write("\n")
        file3.write("\n")
    
    else:
        for eachPredication in eachCandidateSet:
            file3.write(eachPredication)
        file3.write("\n")
        file3.write("\n") 
file3.close()

#Step 2 A never be placed before B in training data
file2 = open("Filtered1_Predication_Result","rb+")
bag = []
readyforStatis = []
noWordIn2017 = []
for lines in file2:
    if bag.count("\n") != 2:
        bag.append(lines)
    else:
        bag.remove("\n")
        bag.remove("\n")
        readyforStatis.append(bag)
        bag = []
        bag.append(lines)

file3 = open("Filtered2_Predication_Result.txt","w")
for eachResult in readyforStatis:
    if len(set(eachResult))>2:
        file3.write(eachResult[0])
        for i in range(1,len(eachResult)):
            predicatedSequence = eachResult[i]
            sequence = predicatedSequence.split()
            k = 0
            m = 0
            for j in range(0,(len(sequence)-1)):
                if (AdjacencyDic.get(sequence[j],"default")!="default") & (AdjacencyDic.get(sequence[j+1],"default")!="default"):
                    if sequence[j+1] not in AdjacencyDic.get(sequence[j]):
                        k = k + 1
                        break                 
                else:
                    if AdjacencyDic.get(sequence[j],"default")=="default":
                        noWordIn2017.append(sequence[j])
                    else:
                        noWordIn2017.append(sequence[j+1])
                    #stemmingWord = ps.stem(sequence[j])
                    #if AdjacencyDic.get(stemmingWord,"default")!="default":
                    #    if sequence[j+1] not in AdjacencyDic.get(stemmingWord):
                    #            k = k+1
                    #else:
            if k ==0:
                file3.write(eachResult[i])
        file3.write("\n")
        file3.write("\n")
    else:
        for eachPredication in eachResult:
            file3.write(eachPredication)
        file3.write("\n")
        file3.write("\n")    
file3.close()    






















                