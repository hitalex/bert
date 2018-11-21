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


def fakenews_evaluation(true_csv_path, prediction_prob_path):
    """ evaluate the prediction results
    Input:
        true_csv_path: the labeled file
        prediction_prob_path: the predicted results
    Output:
        weighted accuracy
    """
    import pandas as pd
    import numpy as np

    label_names = ['unrelated', 'agreed', 'disagreed'] 
    label_name_map = {'unrelated':0, 'agreed':1, 'disagreed':2}
    class_weights = np.array([1.0/16, 1.0/15, 0.2]) # unrelated, agreed, disagreed
    true_df = pd.read_csv(true_csv_path)
    y_true = true_df['label'].values
    pred_df = pd.read_csv(prediction_prob_path, header=None, delimiter='\t')
    y_pred = np.argmax(pred_df.values, axis=1)

    y_true = list(map(lambda v : label_name_map[v], y_true))
    weighted_acc = 0.0
    tmp = sum(class_weights[y_true])
    N = len(y_true)
    for i in range(N):
        #tmp += class_weights[y_true[i]]
        if y_true[i] == y_pred[i]:
            weighted_acc += class_weights[y_true[i]]

    weighted_acc = (weighted_acc / tmp)

    print('Weighted acc: ', weighted_acc)

    return weighted_acc


if __name__ == '__main__':
    path = sys.argv[1]
    out_dir = sys.argv[2]
    generate_train_dev_test_datasets(path, out_dir)
