#coding=utf8

import sys

def generate_train_dev_test_datasets(path, out_dir):
    """ Generate three different datasets for training
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    print("Reading dataset from :" + path)
    data = pd.read_csv(path, ',') # read the training dataset
    print("Spliting dataset...")
    train, test = train_test_split(data, test_size = 0.4, random_state=7)
    dev, test = train_test_split(test, test_size = 0.5, random_state=13)

    print('Saving datasets...')
    train.to_csv(out_dir + "/train.csv", index = False)
    dev.to_csv(out_dir + "/dev.csv", index = False)
    test.to_csv(out_dir + "/test.csv", index = False)

if __name__ == '__main__':
    path = sys.argv[1]
    out_dir = sys.argv[2]
    generate_train_dev_test_datasets(path, out_dir)
