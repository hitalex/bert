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

def add_additional_features(input_path, output_path):
    """ add additional features for bert
    List of features:
    num_question_mark: number of question marks,?
    num_exclamation_mark: number of exclamation mark, !
    num_words: number of segmented words
    """
    import pandas as pd
    import jieba
    import numpy as np

    df = pd.read_csv(input_path)
    num_question_mark = np.zeros((2, len(df)), dtype = int)
    num_exclamation_mark = np.zeros((2, len(df)), dtype = int)
    num_words = np.zeros((2, len(df)), dtype = int)

    for i in range(len(df)):
        try:
            title1 = str(df['title1_zh'].iloc[i])
            title2 = str(df['title2_zh'].iloc[i])
            num_question_mark[0, i] = title1.count('?') + title1.count('？')
            num_question_mark[1, i] = title2.count('?') + title2.count('？')
            num_exclamation_mark[0, i] =  title1.count('!') + title1.count('！')
            num_exclamation_mark[1, i] =  title2.count('!') + title2.count('！')
        except:
            import ipdb; ipdb.set_trace()

        c1 = len(list(jieba.cut(title1)))
        c2 = len(list(jieba.cut(title2)))
        num_words[0, i] = c1
        num_words[1, i] = c2

        if i % 10000 == 0:
            print('Current count: %d, total: %d' % (i, len(df)))

    df['num_question_mark1'] = num_question_mark[0]
    df['num_question_mark2'] = num_question_mark[1]
    df['num_exclamation_mark1'] = num_exclamation_mark[0]
    df['num_exclamation_mark2'] = num_exclamation_mark[1]
    df['num_words1'] = num_words[0]
    df['num_words2'] = num_words[1]

    print('Saving to: ', output_path)
    df.to_csv(output_path, columns = ['id', 'tid1', 'tid2', 'title1_zh', 'title2_zh', 'title1_en', 'title2_en', 
        'num_question_mark1', 'num_question_mark2', 'num_exclamation_mark1', 'num_exclamation_mark2', 
        'num_words1', 'num_words2', 'label'], index = False)

def generate_submit_file(test_id_list, test_result, submit_path):
    """ generate submit files
    Input:
        test_id_list: id list
        test_result: prediction results
        submit_path: output file
    Output:
        void
    """
    import pandas as pd
    import numpy as np
    #import ipdb; ipdb.set_trace()

    class_labels = np.array(["unrelated", "agreed", "disagreed"])
    result = pd.read_csv(test_result, header = None, delimiter = '\t').values
    prediction = np.argmax(result, axis = 1)
    prediction = class_labels[prediction]

    submit_dict = {'Id':test_id_list, 'Category':prediction}
    submit_df = pd.DataFrame(submit_dict)
    submit_df.to_csv(submit_path, index = False)

def populate_train_dataset(train_path, output_path):
    """ 将训练集中的unrelated以及agreed样本复制一份以扩充训练集
    """
    import pandas as pd
    train = pd.read_csv(train_path)
    df1 = train[train['label'] == 'unrelated']
    df2 = train[train['label'] == 'agreed']
    df = df1.append(df2) # merge two data frames
    # 交换特定的列
    tmp = df['tid1']; df['tid1'] = df['tid2']; df['tid2'] = tmp
    # 情感值列交换
    alist = ['pos_', 'neg_', 'neu_', 'compound_']
    for a in alist:
        if (a + '1') in train.columns:
            print('Switching %s and %s' % (a + '1', a + '2'))
            tmp = df[a + '1']
            df[a + '1'] = df[a + '2']
            df[a + '2'] = tmp

    # 交换title以及topic相关特征
    for c in list(train.columns):
        if 'title1' in c:
            t = c.replace('title1', 'title2')
            print('Switching %s and %s' % (c, t))
            tmp = df[c]
            df[c] = df[t]
            df[t] = tmp

    new_train = train.append(df)
    #import ipdb; ipdb.set_trace()
    new_train.to_csv(output_path, index = False)

if __name__ == '__main__':
    path = sys.argv[1]
    out_dir = sys.argv[2]
    generate_train_dev_test_datasets(path, out_dir)
