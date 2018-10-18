import json
import matplotlib.pyplot as plt
import numpy as np

with open('results_appended.json', 'r') as f:
    results = json.load(f)
    
x = []
labels = []
colours = []

for r in results:
    for k in r.keys():
        test_acc = r[k]['test_acc']
        x.append(test_acc)
        labels.append(k)
        colours.append((np.random.random(), np.random.random(), np.random.random()))

y = np.arange(len(x))        

plt.figure(figsize=(15,5))
plt.bar(y, x, align='center', alpha=0.5, color=colours)
plt.xticks(y, labels, rotation='vertical')
plt.ylabel('Test Accuracy')
plt.title('Model Performance')
 
plt.show()


x = []
labels = []

for r in results:
    for k in r.keys():
        test_acc = r[k]['test_loss']
        x.append(test_acc)
        labels.append(k)

y = np.arange(len(x))        

plt.figure(figsize=(15,5))
plt.bar(y, x, align='center', alpha=0.5, color=colours)
plt.xticks(y, labels, rotation='vertical')
plt.ylabel('Test Loss')
plt.title('Model Performance')
 
plt.show()

