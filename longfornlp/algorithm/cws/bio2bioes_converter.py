# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import sys


def BIO2BIOES(input_labels_list):
    output_labels_list = []
    for labels in input_labels_list:
        new_labels = []
        sent_len = len(labels)
        for idx in range(sent_len):
            if "-" not in labels[idx]:
                new_labels.append(labels[idx])
            else:
                label_type = labels[idx].split('-')[-1]
                if "B-" in labels[idx]:
                    if (idx == sent_len - 1) or ("I-" not in labels[idx + 1]):
                        new_labels.append("S-"+label_type)
                    else:
                        new_labels.append("B-"+label_type)
                elif "I-" in labels[idx]:
                    if (idx == sent_len - 1) or ("I-" not in labels[idx + 1]):
                        new_labels.append("E-"+label_type)
                    else:
                        new_labels.append("I-"+label_type)
        assert len(labels) == len(new_labels)
        output_labels_list.append(new_labels)
    return output_labels_list


def BIOES2BIO(input_labels_list):
    output_labels_list = []
    for labels in input_labels_list:
        new_labels = []
        sent_len = len(labels)
        for idx in range(sent_len):
            if "-" not in labels[idx]:
                new_labels.append(labels[idx])
            else:
                label_type = labels[idx].split('-')[-1]
                if "E-" in labels[idx]:
                    new_labels.append("I-" + label_type)
                elif "S-" in labels[idx]:
                    new_labels.append("B-" + label_type)
                else:
                    new_labels.append(labels[idx])
        assert len(labels) == len(new_labels)
        output_labels_list.append(new_labels)
    return output_labels_list


def IOB2BIO(input_labels_list):
    output_labels_list = []
    for labels in input_labels_list:
        new_labels = []
        sent_len = len(labels)
        for idx in range(sent_len):
            if "I-" in labels[idx]:
                label_type = labels[idx].split('-')[-1]
                if (idx == 0) or (labels[idx - 1] == "O") or (label_type != labels[idx - 1].split('-')[-1]):
                    new_labels.append("B-" + label_type)
                else:
                    new_labels.append(labels[idx])
            else:
                new_labels.append(labels[idx])
        assert len(labels) == len(new_labels)
        output_labels_list.append(new_labels)
    return output_labels_list



def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


if __name__ == '__main__':
    '''Convert NER tagging schemes among IOB/BIO/BIOES.
        For example: if you want to convert the IOB tagging scheme to BIO, then you run as following:
            python taggingSchemes_Converter.py IOB2BIO input_iob_file output_bio_file
        Input data format is tsv format.
    '''
    input_file_name, output_file_name = sys.argv[2], sys.argv[3]
    words_list, labels_list, new_labels_list = [], [], []
    with open(input_file_name, 'r') as input_file:
        for line in input_file:
            item = line.rstrip().split('\t')
            assert len(item) == 2
            words, labels = item[0].split(' '), item[1].split(' ')
            if len(words) != len(labels):
                print("Error line: " + line.rstrip())
                continue
            words_list.append(words)
            labels_list.append(labels)

    if sys.argv[1].upper() == "IOB2BIO":
        print("Convert IOB -> BIO...")
        new_labels_list = IOB2BIO(labels_list)
    elif sys.argv[1].upper() == "BIO2BIOES":
        print("Convert BIO -> BIOES...")
        new_labels_list = BIO2BIOES(labels_list)
    elif sys.argv[1].upper() == "BIOES2BIO":
        print("Convert BIOES -> BIO...")
        new_labels_list = BIOES2BIO(labels_list)
    elif sys.argv[1].upper() == "IOB2BIOES":
        print("Convert IOB -> BIOES...")
        tmp_labels_list = IOB2BIO(labels_list)
        new_labels_list = BIO2BIOES(tmp_labels_list)
    else:
        print("Argument error: sys.argv[1] should belongs to \"IOB2BIO/BIO2BIOES/BIOES2BIO/IOB2BIOES\"")

    with open(output_file_name, 'w') as output_file:
        for index in range(len(words_list)):
            words, labels = words_list[index], new_labels_list[index]
            line = " ".join(words) + '\t' + " ".join(labels) + '\n'
            output_file.write(line)

