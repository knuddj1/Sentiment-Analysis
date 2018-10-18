import csv
from prepare_data import prepare


def get_train_data(max_size, n_cats):
    data = []
    labels = []

    print('collecting training data..')
    with open('data.csv', 'r') as f:
        for row in csv.reader(f):
            nums = [0] * len(row)
            for i, d in enumerate(row):
                nums[i] = int(d)
            data.append(nums)
    f.close()
    print('collected training data successfully.')

    print('collecting training labels..')
    with open('labels.csv', 'r') as f:
        for row in csv.reader(f):
            labels.append(int(row[0]))
    f.close()
    print('collected training labels successfully.')

    print("preparing training data and labels..")
    x_train, y_train = prepare(X=data, y=labels, max_size=max_size, n_cats=n_cats, shuffle_data=True) 
    print("prepared training data and labels successfully.")

    return x_train, y_train