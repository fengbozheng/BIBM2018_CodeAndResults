This repository is for BIBM 2018 submission -- Exploring Deep Learning-based Approaches for Predicting Concept Names in SNOMED CT. 
There are two folders:
	CodeForNamePrediction: contains code for neural network models, name prediction and filtering.
	InputAndPredictionResult: contains three files. 
		PreviousWork_SetOfWord.txt	bags of words generated from pervious work which is necessary to construct the concept names
		NamePrediction_NewlyAddedIn2018.txt	name prediction result for newly added concept whose original names are considered as bags of words). Each block (separated by two breaks) contains ground truth (the first row) and predicted names (rest rows in the block).
		NamePrediction_PreviousWork.txt	name prediction result for bags of words generated from pervious work. 

BIBM_NameConcept.py Code Structure:
	Part1 ~ Part5: word embedding and preprocessing
	Part6: three basic neural networks for binary classification
	Part7: name prediction based on bag of words (classify all permutations)
	Part8: two-step filter as tie-breaker
	
