#coding=utf8

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
    train.to_csv(out_dir + "/train.csv", ',')
    dev.to_csv(out_dir + "/dev.csv", ',')
    test.to_csv(out_dir + "/test.csv", ',')

if __name__ == '__main__':
    
    generate_train_dev_test_datasets('../datasets/wsdm2019-fakenews/train.csv', '../datasets/fakenews/')
