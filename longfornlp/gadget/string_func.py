import numpy as np
import re
import string
import codecs


def cut_and_padding(seq, max_len, pad=0):
    """
    cut or pad the sequence to fixed size
    Args:
        seq:     sequence
        max_len:    the fixed size specified
        pad:    symbol to pad
    Returns:
        fixed size sequence
    """
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        return seq + [pad] * (max_len - len(seq))


def to_categorical(y, nb_classes=None):
    """Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    """
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.0
    return Y


def base_filter():
    import string

    f = string.punctuation
    f = f.replace("'", "")
    f += "\t\n"
    return f


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def is_number(s):
    '''
    判断字符串是否为数字
    '''
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def number_filter(word_list):
    """去除数字"""
    words_list = [word for word in word_list if not is_number(word)]
    return words_list


def shuffle_in_unison(x, y):
    """打乱数据顺序"""
    assert len(x) == len(y)
    shuffled_x = [0] * len(x)
    shuffled_y = [0] * len(y)
    permutation = np.random.permutation(len(x))
    for old_index, new_index in enumerate(permutation):
        shuffled_x[new_index] = x[old_index]
        shuffled_y[new_index] = y[old_index]
    return shuffled_x, shuffled_y
