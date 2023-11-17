'''
Notes for user:
Please install the following packages (3.8 - 3.11)...
pip install numpy
pip install tflearn
pip install tensorflow==2.12.0

If any error such as "module 'PIL.Image' has no attribute 'ANTIALIAS'" occurs, your version of pillow doesn't support the application please do the following in your command prompt
1. pip uninstall Pillow
2. pip install Pillow==9.5.0 --user
'''

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle


with open("intents.json") as file:
	data = json.load(file)

try:
	with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)
except:
	words = []
	labels = []
	docs_x = []
	docs_y = []

	print(data)

	for intent in data["intents"]:
	    for pattern in intent["patterns"]:
	        wrds = nltk.word_tokenize(pattern)
	        words.extend(wrds)
	        docs_x.append(wrds)
	        docs_y.append(intent["tag"])

	    if intent["tag"] not in labels:
	        labels.append(intent["tag"])

	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))

	labels = sorted(labels)

	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	for x, doc in enumerate(docs_x):
	    bag = []

	    wrds = [stemmer.stem(w.lower()) for w in doc]

	    for w in words:
	        if w in wrds:
	            bag.append(1)
	        else:
	            bag.append(0)

	    output_row = out_empty[:]
	    output_row[labels.index(docs_y[x])] = 1

	    training.append(bag)
	    output.append(output_row)


	training = numpy.array(training)
	output = numpy.array(output)

	with open("data.pickle", "wb") as f:
		pickle.dump((words, labels, training, output), f)

# model
tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape =[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# training model
# Improvement for accuracy: add more patterns, maybe get input from UH students themselves?
model = tflearn.DNN(net)

try:
	model.load("model.tflearn")
except:
	# Looking at the data
	model.fit(training, output, n_epoch=500, batch_size=8, show_metric=True)
	model.save("model.tflearn")

# Predictions
def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	# if word from inetents pattern appears, replace with 1
	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)

def chat():
	print("Welcome to the UH ITS desk. I'm your virtual assisstant here to help you with any IT related problems. What can I do for you today?")
	while True:
		inp = input("You: ")
		if inp.lower() == "quit":
			break

		# Model Prediction might need some work -Brandon
		results = model.predict([bag_of_words(inp, words)])
		
		results_index = numpy.argmax(results)
		tag = labels[results_index]

		for tg in data["intents"]:
			if tg['tag'] ==  tag:
				responses = tg['responses']

		print(random.choices(responses))
		

chat()