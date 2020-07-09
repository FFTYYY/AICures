from .featurization import morgan_features


def read_data(choice: int = 0):
    ''' 
    Read single fold data and compute features.

    :param choice: choose which fold to read data from.
    
    '''
    train_set = []
    train_label = []
    valid_set = []
    valid_label = []
    test_set = []
    test_label = []
    
    # 读取训练集
    train_file = 'data/train_cv/fold_' + str(choice) + '/train.csv'
    with open(train_file, 'r') as f:
        f.readline()    # ignore the first line
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            train_set.append(morgan_features(line[0]))
            train_label.append(int(line[1]))
    
    # 读取验证集
    valid_file = 'data/train_cv/fold_' + str(choice) + '/dev.csv'
    with open(valid_file, 'r') as f:
        f.readline()    # ignore the first line
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            valid_set.append(morgan_features(line[0]))
            valid_label.append(int(line[1]))
    
    # 读取测试集
    test_file = 'data/train_cv/fold_' + str(choice) + '/test.csv'
    with open(test_file, 'r') as f:
        f.readline()    # ignore the first line
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            test_set.append(morgan_features(line[0]))
            test_label.append(int(line[1]))
        
    return train_set, train_label, valid_set, valid_label, test_set, test_label


def read_all_data():
    ''' read data in all folds
    '''
    train_set = []
    train_label = []
    valid_set = []
    valid_label = []
    test_set = []
    test_label = []
    
    # read all folds
    for i in range(10):
        p_train_set, p_train_label, p_valid_set, p_valid_label, p_test_set, p_test_label = read_data(choice=i)
        train_set.append(p_train_set)
        train_label.append(p_train_label)
        valid_set.append(p_valid_set)
        valid_label.append(p_valid_label)
        test_set.append(p_test_set)
        test_label.append(p_test_label)
    
    return train_set, train_label, valid_set, valid_label, test_set, test_label

