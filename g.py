import csv
import json
from prepare_data import prepare
from gensim.parsing.preprocessing import *

MAX_SIZE = 500
NUM_CATEGORIES = 3


with open('word_to_index_top_30000.json', 'r') as f:
    d = json.load(f)

data = []
labels = []

with open('test_data.csv', 'r', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile)
    for r in reader:
        words = preprocess_string(r[0], CUSTOM_FILTERS)
        nums = [0] * len(words)
        for i, word in enumerate(words):
            if word in d:
                nums[i] = d[word]
        data.append(nums)
        labels.append(r[-1])

x_test, y_test = prepare(data[1:], labels[1:], MAX_SIZE, NUM_CATEGORIES)

for i in range(len(x_test)):
    print(x_test[i], y_test[i])