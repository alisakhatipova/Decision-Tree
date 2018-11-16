import pydot
import numpy as np

# node id for visualization
global_node_count = 0


def inform_gain(attrs, classes, value):
    """
    Calculate information gain
    """
    result = entropy(classes)
    result -= cond_entropy(attrs, classes, value)
    return result


def entropy(classes):
    """
    Calculate entropy
    """
    total = len(classes)
    pos = len(classes[np.where(classes == 1)])
    neg = len(classes[np.where(classes == 0)])
    result = 0.0
    if pos > 0:
        pos = float(pos) / total
        result += pos * np.log2(1.0 / pos)
    if neg > 0:
        neg = float(neg) / total
        result += neg * np.log2(1.0 / neg)
    return result


def cond_entropy(attrs, classes, value):
    """
    Calculate conditional entropy
    """
    total = len(classes)
    left_split = attrs[np.where(attrs < value)]
    right_split = attrs[np.where(attrs >= value)]
    num_left = len(left_split)
    num_right = len(right_split)
    left_entropy = entropy(classes[np.where(attrs < value)])
    right_entropy = entropy(classes[np.where(attrs >= value)])
    result = (left_entropy * float(num_left) + right_entropy * float(num_right)) / total
    return result


def choose_split_value(attrs, classes):
    """
    Choose split value for the continious attribute
    that maximizes information gain
    """
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
    """
    Calculate the majority class for the attribute
    Needed in case when no more splits are possible
    """
    num_pos = len(classes[np.where(classes == 1)])
    num_neg = len(classes) - num_pos
    return 1 if num_pos > num_neg else 0


def TDIDT(data, node):
    classes = data["class_label"].astype(int)
    # stuff for visualization
    global global_node_count
    node['id'] = global_node_count
    global_node_count += 1
    num_pos = len(classes[np.where(classes == 1)])
    num_neg = len(classes) - num_pos
    node['neg'] = num_neg
    node['pos'] = num_pos
    # the main algorithm
    attrs = data.dtype.names
    attrs = [x for x in attrs if x != "class_label"]
    are_equal = np.all([classes[i] == classes[0] for i in range(len(classes))])
    if are_equal:
        node['final_class'] = classes[0]
        return
    if len(attrs) == 0:
        node['final_class'] = majority_class(classes)
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
    left_data = left_data[[b for b in list(left_data.dtype.names) if b != max_gain_attr]]
    right_data = data[np.where(data[max_gain_attr] >= max_gain_split_value)]
    right_data = right_data[[b for b in list(right_data.dtype.names) if b != max_gain_attr]]
    TDIDT(left_data, node['left_child'])
    TDIDT(right_data, node['right_child'])


def add_node(graph, node, parent, label):
    """
    Add node to the graph
    """
    neg = node['neg']
    pos = node['pos']
    total = str(neg + pos)
    neg = str(neg)
    pos = str(pos)
    samples_info = total + ' samples\n' + neg + ' of class 0, ' + pos + ' of class 1'
    if 'final_class' in node:
        legend = str(node['id']) + '. final class is ' + str(node['final_class'])
        new_node = pydot.Node(legend)
    else:
        legend = str(node['id']) + '. ' + node['split_attr'] + \
                 ' < ' + str(node['split_value']) + '\n' + samples_info
        new_node = pydot.Node(legend)
    graph.add_node(new_node)
    if parent:
        graph.add_edge(pydot.Edge(parent, new_node, label=str(label),labelfontcolor="#009933", fontsize="10.0", color="blue"))
    if 'left_child' in node:
        add_node(graph, node['left_child'], new_node, True)
    if 'right_child' in node:
        add_node(graph, node['right_child'], new_node, False)


def save_dot(tree):
    """
    Visualize the resulting decision tree
    """
    graph = pydot.Dot(graph_type='graph')
    add_node(graph, tree, None, None)
    graph.write_png('out_graph.png')


def predict(node, data):
    if 'final_class' in node:
        return node['final_class']
    attr_val = data[node['split_attr']]
    split_value = node['split_value']
    if attr_val < split_value:
        return predict(node['left_child'], data)
    else:
        return predict(node['right_child'], data)


if __name__ == "__main__":
    path_to_csv = "gene_expression_training.csv"
    data = np.genfromtxt(path_to_csv, dtype=float, delimiter=',', names=True)
    tree = {}
    TDIDT(data, tree)
    save_dot(tree)
    path_to_test_csv = "gene_expression_test.csv"
    data = np.genfromtxt(path_to_test_csv, dtype=float, delimiter=',', names=True)
    total = len(data)
    correctly_classified = 0
    for val in data:
        predicted = predict(tree, val)
        real = val["class_label"].astype(int)
        if predicted == real:
            correctly_classified +=1
    print(str(float(correctly_classified) / total) + ' correctly classified')