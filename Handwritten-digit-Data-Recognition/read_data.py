import csv
import numpy as np
data_dir = './data'


def read_train():
    text = []
    with open(data_dir + '/train.csv', 'rt') as file:
        lines = csv.reader(file)  # (42001*785)
        count = 0
        for line in lines:
            text.append(line)
            count += 1
            if count == 100:
                break

    lable_feature = text[1:]
    lable_feature = np.array(lable_feature)
    label = lable_feature[:, 0].astype(int)
    print("train_label shape:{}".format(label.shape))
    feature = lable_feature[:, 1:].astype(int)
    print("train_feature shape:{}".format(feature.shape))
    return label, feature


def read_test():
    text = []
    with open(data_dir + '/test.csv', 'rt') as file:
        lines = csv.reader(file)  # (28001*784)
        count = 0
        for line in lines:
            text.append(line)
            count += 1
            if count == 100:
                break
    lable_feature = text[1:]
    feature = np.array(lable_feature).astype(int)
    print("feature shape:{}".format(feature.shape))

    return feature


def read_handwritten_data():
    train_y, train_x = read_train()
    test_x = read_test()
    return train_x[:10], train_y[:10], test_x[:100]


# read_handwritten_data()
