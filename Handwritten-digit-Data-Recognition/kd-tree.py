import numpy as np
import csv
from read_data import read_handwritten_data

data_dir = './data'


class KD_node:

    def __init__(self, data=None, split=None, left=None, right=None, label=None):
        self.data = data
        self.split = split
        self.left = left
        self.right = right
        self.label = label


def createKDTREE(root, data_array, label_array):
    """输入当前初始化的根节点以及当前待分割的数据
        输出将这些数据分割成两部分的根节点
    """

    num = data_array.shape[0]  # 样本数目

    if num == 0:  # 如果没有样本了, 则返回空结点
        return None
    if num == 1:  # 如果只剩一个样本, 则就以此样本为根节点
        root.data = [np.array(data_array[0])]
        root.label = label_array[0]
        return root

    # 否则计算样本数据的维度
    dim = data_array.shape[1]

    variance = data_array.var(axis=0)  # take the varipy of each col
    split_dim = np.argmax(variance)

    data = data_array[:, split_dim]
    midian = sorted(data)[int(num / 2)]  # 得到split_dim维度上的中位数

    # 根节点就是在split_dim维度上等于中位数的样本
    for index, item in enumerate(data_array):
        if item[split_dim] == midian:
            root_data = [item]
            label = label_array[index]
    root_index = index

    # 去除根节点以后的数据, 并将其根据中位数分为左右两部分
    left = []
    left_label = []
    for index, item in enumerate(data_array):
        if item[split_dim] <= midian and index != root_index:
            left.append(item)
            left_label.append(label_array[index])
    left_label = np.array(left_label)

    right = []
    right_label = []
    for index, item in enumerate(data_array):
        if item[split_dim] > midian:
            right.append(item)
            right_label.append(label_array[index])
    right_label = np.array(right_label)

    root.data = root_data
    root.split = split_dim
    root.label = label

    left_node = KD_node()
    right_node = KD_node()
    root.left = createKDTREE(left_node, np.array(
        left), left_label)  # 返回的根节点作为此root的左子树

    root.right = createKDTREE(right_node, np.array(
        right), right_label)  # 返回的根节点作为此root的右子树

    # 返回根节点
    return root


def node2node_dist(node, target):
    '''计算两个样本数据之间的欧式距离
        输入:
        node:[array], 因此在程序中使用了node.data[0]
        target:array
    '''
    dim = 784
    dist = 0
    for i in range(dim):
        dist += (node.data[0][i] - target[i]) * (node.data[0][i] - target[i])
    dist = np.sqrt(dist)
    return dist


def node2plain_dist(node, target):
    '''计算target样本数据与node所确定的超平面之间的距离
        输入:
        node:[array], 因此在程序中使用了node.data[0]
        target:array
    '''
    split_dim = node.split
    dist = np.abs(node.data[0][split_dim] - target[split_dim])
    return dist


def Search(target, root):
    #print("step into search, root is:", root.data)
    node = root
    MIN_DIST = 100000
    MIN_NODE = node
    while(node is not None and node.split is not None):  # 在非叶子节点中搜索
        split_dim = node.split
        dist = node2node_dist(node, target)
        MIN_DIST = dist if dist < MIN_DIST else MIN_DIST
        MIN_NODE = node if dist < MIN_DIST else MIN_NODE

        if target[split_dim] <= node.data[0][split_dim]:
            node = node.left
        else:
            node = node.right

    # 到达叶子节点了
    if node is not None:
        dist = node2node_dist(node, target)
        MIN_DIST = dist if dist < MIN_DIST else MIN_DIST
        MIN_NODE = node if dist < MIN_DIST else MIN_NODE
    return MIN_DIST, MIN_NODE


def KNN(root, target):
    # 二叉搜索, 直到叶子结点
    node = root
    node_list = [node]  # 用来保存遍历过的结点
    MIN_NODE = node
    MIN_DIST = 100000

    while(node is not None and node.split is not None):
        split_dim = node.split
        dist = node2node_dist(node, target)
        MIN_NODE = node if dist < MIN_DIST else MIN_NODE
        MIN_DIST = dist if dist < MIN_DIST else MIN_DIST
        node_list.append(node)

        if(target[split_dim] <= node.data[0][split_dim]):
            node = node.left
        else:
            node = node.right

    if node is not None:
        dist = node2node_dist(node, target)
        MIN_DIST = dist if dist < MIN_DIST else MIN_DIST
        MIN_NODE = node if dist < MIN_DIST else MIN_NODE

    while(len(node_list) >= 1):
        current = node_list.pop()
        current_split = current.split
        dist = node2plain_dist(current, target)
        if dist < MIN_DIST:

            if target[current_split] <= current.data[0][current_split]:
                # 说明在此节点的左区域
                if current.right is not None:
                    min_dist, min_node = Search(target, current.right)
                    MIN_DIST = min_dist if min_dist < MIN_DIST else MIN_DIST
                    MIN_NODE = node if min_dist < MIN_DIST else MIN_NODE
            else:  # 说明在此节点的右区域
                if current.left is not None:
                    min_dist, min_node = Search(target, current.left)
                    MIN_DIST = min_dist if min_dist < MIN_DIST else MIN_DIST
                    MIN_NODE = node if min_dist < MIN_DIST else MIN_NODE
    return MIN_DIST, MIN_NODE


def display_tree(root):
    print("begin to display")
    with open("graphviz.dot", "a") as f:
        if root is not None:
            if root.left is not None:
                f.write(str(root.data[0])[1:-1]+'--'+str(root.left.data[0])[1:-1]+";")
                f.write('\n')   
                display_tree(root.left)
            if root.right is not None:
                f.write(str(root.data[0])[1:-1]+'--'+str(root.right.data[0])[1:-1]+";")
                display_tree(root.right)
                f.write('\n')   
            else:
                return


def test():
    root = KD_node()
    data_array = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    root = createKDTREE(root, data_array)
    display_tree(root)
    target = np.array([8, 3])
    MIN_DIST, MIN_NODE = KNN(root, target)
    print("min dist:", MIN_DIST)
    print("MIN_NODE dist:", MIN_NODE.data)


def main():
    train_x, train_y, test_x = read_handwritten_data()
    root = KD_node()
    data_array = train_x
    root = createKDTREE(root, data_array, train_y)
    display_tree(root)

    with open(data_dir + '/result.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageId', 'Lable'])

        for index, test in enumerate(test_x):
            target = test

            MIN_DIST, MIN_NODE = KNN(root, target)
            writer.writerow([int(index + 1), int(MIN_NODE.label)])

main()
