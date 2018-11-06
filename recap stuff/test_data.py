import csv
import json
from prepare_data import prepare
from gensim.parsing.preprocessing import *

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

def get_test_data(max_size, n_cats):
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
            
    x_test, y_test = prepare(X=data[1:], y=labels[1:], max_size=max_size, n_cats=n_cats, shuffle_data=True)

    return x_test, y_test