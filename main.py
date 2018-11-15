import numpy as np
from math import log


def inform_gain(attrs, classes, value):
    result = entropy(attrs, classes)
    result -= cond_entropy(attrs, classes, value)
    return result


def entropy(attrs, classes):
    total = len(attrs)
    pos = len(attrs[np.where(classes == 1)])
    neg = len(attrs[np.where(classes == 0)])
    result = 0.0
    if pos > 0:
        pos = float(pos) / total
        result += pos*log(1.0 / pos, 2)
    if neg > 0:
        neg = float(neg) / total
        result += neg * log(1.0 / neg, 2)
    return result


def cond_entropy(attrs, classes, value):
    total = len(attrs)
    left_split = attrs[np.where(attrs < value)]
    right_split = attrs[np.where(attrs >= value)]
    num_left = len(left_split)
    num_right = len(right_split)
    left_entropy = entropy(left_split, classes[np.where(attrs < value)])
    right_entropy = entropy(right_split, classes[np.where(attrs >= value)])
    result = (left_entropy * float(num_left) + right_entropy * float(num_right)) / total
    return result


def choose_split_value(attrs, classes):
    indices = np.argsort(attrs)
    classes = classes[indices]
    attrs = attrs[indices]
    max_gain = 0.0
    max_gain_value = None
    for i in range(len(attrs) - 1):
        if classes[i] != classes[i+1]:
            mean = (attrs[i] + attrs[i+1]) / 2.0
            gain = inform_gain(attrs, classes, mean)
            if gain > max_gain:
                max_gain = gain
                max_gain_value = mean
    return max_gain_value, max_gain


def majority_class(classes):
    num_pos = classes[np.where(classes == 1)]
    num_neg = len(classes) - num_pos
    return 1 if num_pos > num_neg else 0


def TDIDT(data, node):
    classes = data["class_label"].astype(int)
    attrs = data.dtype.names
    attrs = [x for x in attrs if x != "class_label"]
    are_equal = np.all([classes[i] == classes[0] for i in range(len(classes))])
    if are_equal:
        node['final_class'] = data[0]
        return
    if len(attrs) == 0:
        node['final_class'] = majority_class(data["class_label"].astype(int))
        return
    max_gain = 0.0
    max_gain_attr = None
    max_gain_split_value = 0.0
    for attr in attrs:
        value, gain = choose_split_value(data[attr], classes)
        if gain > max_gain:
            max_gain = gain
            max_gain_attr = attr
            max_gain_split_value = value
    node['split_value'] = max_gain_split_value
    node['split_attr'] = max_gain_attr
    node['left_child'] = {}
    node['right_child'] = {}
    left_data = data[np.where(data[max_gain_attr] < max_gain_split_value)]
    left_data = left_data[[b for b in list(attrs) if b != max_gain_attr]]
    right_data = data[np.where(data[max_gain_attr] >= max_gain_split_value)]
    right_data = right_data[[b for b in list(attrs) if b != max_gain_attr]]
    TDIDT(left_data, node['left_child'])
    TDIDT(right_data, node['right_child'])


# if __name__ == "__main__":
path_to_csv = "gene_expression_training.csv"
data = np.genfromtxt(path_to_csv, dtype=float, delimiter=',', names=True)
tree = {}
TDIDT(data, tree)
# all_attrs = data.dtype.names
# all_attrs = [x for x in all_attrs if x != "class_label"]
