from jionlp import logging


class TrieTree(object):
    """
    Trie 树的基本方法，用途包括：
    - 词典 NER 的前向最大匹配计算
    - 繁简体词汇转换的前向最大匹配计算

    """
    def __init__(self):
        self.dict_trie = dict()
        self.depth = 0

    def add_node(self, word, typing):
        """向 Trie 树添加节点。

        Args:
            word(str): 词典中的词汇
            typing(str): 词汇类型

        Returns: None

        """
        word = word.strip()
        if word not in ['', '\t', ' ', '\r']:
            tree = self.dict_trie
            depth = len(word)
            word = word.lower()  # 将所有的字母全部转换成小写
            for char in word:
                if char in tree:
                    tree = tree[char]
                else:
                    tree[char] = dict()
                    tree = tree[char]
            if depth > self.depth:
                self.depth = depth
            if 'type' in tree and tree['type'] != typing:
                logging.warning(
                    '`{}` belongs to both `{}` and `{}`.'.format(
                        word, tree['type'], typing))
            else:
                tree['type'] = typing

    def build_trie_tree(self, dict_list, typing):
        """ 创建 trie 树 """
        for word in dict_list:
            self.add_node(word, typing)

    def search(self, word):
        """ 搜索给定 word 字符串中与词典匹配的 entity，
        返回值 None 代表字符串中没有要找的实体，
        如果返回字符串，则该字符串就是所要找的词汇的类型
        """
        tree = self.dict_trie
        res = None
        step = 0  # step 计数索引位置
        for char in word:
            if char in tree:
                tree = tree[char]
                step += 1
                if 'type' in tree:
                    res = (step, tree['type'])
            else:
                break
        if res:
            return res
        return 1, None



def dictree_by_esm():
    import esm
    # pip install esmre

    print('获取字典树trie')
    word_list = ['apple', 'alien', 'app']
    dic = esm.Index()
    for i in range(len(word_list)):
        word = word_list[i].lower()
        dic.enter(word)
    dic.fix()

    sentence = 'i like apple and app'
    result = dic.query(sentence.lower())
    result = list(set(result))
    print(result)


def dictree_by_ahocorasick():
    import ahocorasick
    # > pip install pyahocorasick

    A = ahocorasick.Automaton()
    words = "he hers his she hi him man he"
    for i,w in enumerate(words.split()):
        A.add_word(w, (i, w))

    # convert the trie to an Aho-Corasick automaton to enable Aho-Corasick search
    A.make_automaton()

    """
    import cPickle
    >>> pickled = cPickle.dumps(A)
    >>> B = cPickle.loads(pickled)
    """

    s = "he rshershidamanza "
    print([x[1][1] for x in A.iter(s)])

    for end_index, (insert_order, original_value) in A.iter(s, 2, 8):
        start_index = end_index - len(original_value) + 1
        print((start_index, end_index, (insert_order, original_value)))
        assert s[start_index:start_index + len(original_value)] == original_value

    print("====")

    def callback(index, item):
        print(index, item)

    A.find_all(s, callback, 2, 11)


def dictree_by_flashtext():
    from flashtext.keyword import KeywordProcessor

    keyword_processor = KeywordProcessor(case_sensitive=False)
    words = "he hers his she hi him man he"
    keyword_processor.add_keywords_from_list(words.split())

    s = "he rshershidamanza "
    keywords_found = keyword_processor.extract_keywords(s, span_info=True)
    print(keywords_found)
