from keras.models import model_from_json
from gensim.parsing.preprocessing import *
from keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
import csv

CUSTOM_FILTERS = [  lambda x: x.lower(), #To lowercase
                    lambda text: re.sub(r'https?:\/\/.*\s', '', text, flags=re.MULTILINE), #To Strip away URLs
                    strip_tags, #Remove tags from s using RE_TAGS.
                    strip_non_alphanum,#Remove non-alphabetic characters from s using RE_NONALPHA.
                    strip_punctuation, #Replace punctuation characters with spaces in s using RE_PUNCT.
                    strip_numeric, #Remove digits from s using RE_NUMERIC.
                    strip_multiple_whitespaces,#Remove repeating whitespace characters (spaces, tabs, line breaks) from s and turns tabs & line breaks into spaces using RE_WHITESPACE.
                    remove_stopwords, # Set of 339 stopwords from Stone, Denis, Kwantes (2010).
                    lambda x: strip_short(x, minsize=3), #Remove words with length lesser than minsize from s.
                ]

with open('word_to_index_top_30000.json', 'r') as f:
    d = json.load(f)

x = []
labels = []

with open('test_data.csv', 'r', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile)
    for r in reader:
        words = preprocess_string(r[0], CUSTOM_FILTERS)
        nums = [0] * len(words)
        for i, word in enumerate(words):
            if word in d:
                nums[i] = d[word]
        x.append(nums)
        labels.append(r[-1])
                
x = pad_sequences(x[1:], maxlen=59, truncating='post', padding='pre')               
y = np.zeros(shape=(len(labels[1:]),3))
for i, l in enumerate(labels[1:]):
    label = int(l)
    if label == -1:
        y[i, 0] = 1.
    elif label == 0:
        y[i, 1] = 1.
    else:
        y[i, 2] = 1.     
        
 



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
print(loaded_model.evaluate(x,y))