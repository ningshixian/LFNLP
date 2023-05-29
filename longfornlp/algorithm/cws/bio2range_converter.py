def conll2list(text, data_type=""):
    """
    Helper function converting data in conll format to word lists
    and token label lists.

    Args:
        text (str): Text string in conll format, e.g.
            "Amy B-PER
             ADAMS I-PER
             works O
             at O
             the O
             University B-ORG
             of I-ORG
             Minnesota I-ORG
             . O"
        data_type (str, optional): String that briefly describes the data,
            e.g. "train"
    Returns:
        tuple:
            (list of word lists, list of token label lists)
    """
    text_list = text.split("\n\n")
    if text_list[-1] in (" ", ""):
        text_list = text_list[:-1]

    max_seq_len = 0
    sentence_list = []
    labels_list = []
    for s in text_list:
        # split each sentence string into "word label" pairs
        s_split = s.split("\n")
        # split "word label" pairs
        s_split_split = [t.split() for t in s_split]
        sentence_list.append([t[0] for t in s_split_split if len(t) > 1])
        labels_list.append([t[1] for t in s_split_split if len(t) > 1])

        if len(s_split_split) > max_seq_len:
            max_seq_len = len(s_split_split)
    print("Maximum sequence length in the {0} data is: {1}".format(data_type, max_seq_len))
    return sentence_list, labels_list


def iob_ranges(words, tags):
    """
    IOB -> Ranges
    """
    assert len(words) == len(tags)
    ranges = []

    def check_if_closing_range():
        if i == len(tags) - 1 or tags[i + 1].split("_")[0] == "O":
            ranges.append({"entity": "".join(words[begin : i + 1]), "type": temp_type, "start": begin, "end": i})

    for i, tag in enumerate(tags):
        if tag.split("_")[0] == "O":
            pass
        elif tag.split("_")[0] == "B":
            begin = i
            temp_type = tag.split("_")[1]
            check_if_closing_range()
        elif tag.split("_")[0] == "I":
            check_if_closing_range()
    return ranges


def ents_iob(ents_dict, sen):
    """
    Ranges -> IOB
    """
    import jieba.posseg as pseg

    ranges_label = ["O"] * len(sen)
    ranges_flag = ["O"] * len(sen)

    for flag, ent in ents_dict.items():
        begin = sen.find(ent)
        while not begin == -1:
            end = begin + len(ent)
            ranges_label[begin:end] = ["B"] + ["I"] * (len(ent) - 1)
            ranges_flag[begin:end] = [flag] * len(ent)
            begin = sen.find(ent, end)

    words_list = []
    postags_list = []
    for x in pseg.cut(sen):
        words_list.append(x.word)
        postags_list.append(x.flag)
    print(words_list)

    new_ranges = []
    begin = 0
    for i in range(len(words_list)):
        w = words_list[i]
        end = begin + len(w)
        if "B" in ranges_label[begin:end]:
            new_ranges.append("B_" + ranges_flag[begin])
        elif "I" in ranges_label[begin:end]:
            new_ranges.append("I_" + ranges_flag[begin])
        else:
            new_ranges.append("O")
        begin = end
    return new_ranges
