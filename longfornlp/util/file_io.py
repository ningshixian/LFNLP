import pickle as pkl
import json
import pandas as pd
import os

"""数据持久化存储和读取"""


__all__ = ['read_file_by_iter', 'read_file_by_line',
           'write_file_by_line']


def read_file_by_iter(file_path, line_num=None, 
                      skip_empty_line=True, strip=True,
                      auto_loads_json=True):
    """读取一个文件的前 N 行，按迭代器形式返回返回，
    文件中按行组织，要求 utf-8 格式编码的自然语言文本。
    若每行元素为 json 格式可自动加载。

    Args:
        file_path(str): 文件路径
        line_num(int): 读取文件中的行数，若不指定则全部按行读出
        skip_empty_line(boolean): 是否跳过空行
        strip(bool): 将每一行的内容字符串做 strip() 操作
        auto_loads_json(bool): 是否自动将每行使用 json 加载，默认为真

    Returns:
        list: line_num 行的内容列表

    Examples:
        >>> file_path = '/path/to/stopwords.txt'
        >>> print(jio.read_file_by_iter(file_path, line_num=3))

        # ['在', '然后', '还有']

    """
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while True:
            if line == '':  # 整行全空，说明到文件底
                break
            if line_num is not None:
                if count >= line_num:
                    break

            if line.strip() == '':
                if skip_empty_line:
                    count += 1
                    line = f.readline()
                else:
                    try:
                        if auto_loads_json:
                            cur_obj = json.loads(line.strip())
                            yield cur_obj
                        else:
                            if strip:
                                yield line.strip()
                            else:
                                yield line
                    except:
                        if strip:
                            yield line.strip()
                        else:
                            yield line
                    count += 1
                    line = f.readline()
                    continue
            else:
                try:
                    if auto_loads_json:
                        cur_obj = json.loads(line.strip())
                        yield cur_obj
                    else:
                        if strip:
                            yield line.strip()
                        else:
                            yield line
                except:
                    if strip:
                        yield line.strip()
                    else:
                        yield line

                count += 1
                line = f.readline()
                continue


def read_file_by_line(file_path, line_num=None, 
                      skip_empty_line=True, strip=True,
                      auto_loads_json=True):
    """ 读取一个文件的前 N 行，按列表返回，
    文件中按行组织，要求 utf-8 格式编码的自然语言文本。
    若每行元素为 json 格式可自动加载。

    Args:
        file_path(str): 文件路径
        line_num(int): 读取文件中的行数，若不指定则全部按行读出
        skip_empty_line(boolean): 是否跳过空行
        strip: 将每一行的内容字符串做 strip() 操作
        auto_loads_json(bool): 是否自动将每行使用 json 加载，默认是

    Returns:
        list: line_num 行的内容列表

    Examples:
        >>> file_path = '/path/to/stopwords.txt'
        >>> print(jio.read_file_by_line(file_path, line_num=3))

        # ['在', '然后', '还有']

    """
    content_list = list()
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while True:
            if line == '':  # 整行全空，说明到文件底
                break
            if line_num is not None:
                if count >= line_num:
                    break

            if line.strip() == '':
                if skip_empty_line:
                    count += 1
                    line = f.readline()
                else:
                    try:
                        if auto_loads_json:
                            cur_obj = json.loads(line.strip())
                            content_list.append(cur_obj)
                        else:
                            if strip:
                                content_list.append(line.strip())
                            else:
                                content_list.append(line)
                    except:
                        if strip:
                            content_list.append(line.strip())
                        else:
                            content_list.append(line)

                    count += 1
                    line = f.readline()
                    continue
            else:
                try:
                    if auto_loads_json:
                        cur_obj = json.loads(line.strip())
                        content_list.append(cur_obj)
                    else:
                        if strip:
                            content_list.append(line.strip())
                        else:
                            content_list.append(line)
                except:
                    if strip:
                        content_list.append(line.strip())
                    else:
                        content_list.append(line)

                count += 1
                line = f.readline()
                continue
                
    return content_list


def write_file_by_line(data_list, file_path, start_line_idx=None,
                       end_line_idx=None, replace_slash_n=True):
    """ 将一个数据 list 按行写入文件中，
    文件中按行组织，以 utf-8 格式编码的自然语言文本。

    Args:
        data_list(list): 数据 list，每一个元素可以是 str, list, dict
        file_path(str): 写入的文件名，可以是绝对路径
        start_line_idx(int): 将指定行的数据写入文件，起始位置，None 指全部写入
        end_line_idx(int): 将指定行的数据写入文件，结束位置，None 指全部写入
        replace_slash_n(bool): 将每个字符串元素中的 \n 进行替换，避免干扰

    Returns:
        None

    Examples:
        >>> data_list = [{'text': '上海'}, {'text': '广州'}]
        >>> jio.write_file_by_line(data_list, 'sample.json')

    """
    if start_line_idx is None:
        start_line_idx = 0
    if end_line_idx is None:
        end_line_idx = len(data_list)

    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list[start_line_idx: end_line_idx]:
            if type(item) in [list, dict]:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            elif type(item) is set:
                f.write(json.dumps(list(item), ensure_ascii=False) + '\n')
            elif type(item) is str:
                f.write(item.replace('\n', '') + '\n')
            elif type(item) in [int, float]:
                f.write(str(item) + '\n')
            else:
                wrong_line = 'the type of `{}` in data_list is `{}`'.format(
                    item, type(item))
                raise TypeError(wrong_line)


def load_from_pkl(pkl_path):
    with open(pkl_path, "rb") as fin:
        obj = pkl.load(fin)
    return obj


def dump_to_pkl(obj, pkl_path):
    with open(pkl_path, "wb") as fout:
        pkl.dump(obj, fout, protocol=pkl.HIGHEST_PROTOCOL)


def load_from_json(json_path):
    data = None
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.loads(f.read())
        except Exception as e:
            print(e)
            raise ValueError("%s is not a legal JSON file, please check your JSON format!" % json_path)
    return data


def dump_to_json(obj, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj))


# 读取EXCEL文件
def readExcel(excel_file, sheet_name=0, tolist=True):
    obj = None
    if os.path.exists(excel_file):
        io = pd.io.excel.ExcelFile(os.path.join(excel_file))
        df = pd.read_excel(io, sheet_name)
        df.fillna("", inplace=True)
        io.close()
        obj = df.values.tolist() if tolist else df
    return obj


# #  将数据写入EXCEL文件
# def writeExcel(file_path, datas):
#     import xlwt
#     f = xlwt.Workbook()
#     sheet1 = f.add_sheet(u"sheet1", cell_overwrite_ok=True)  # 创建sheet
#     for i in range(len(datas)):
#         data = datas[i]
#         for j in range(len(data)):
#             sheet1.write(i, j, str(data[j]))  # 将数据写入第 i 行，第 j 列
#     f.save(file_path)  # 保存文件


#  将数据写入EXCEL文件
def writeExcel(file_path, datas, header=[]):
    """
    file_path: 写入excel文件路径
    datas: 要写入的数据
    header 指定列名
    """
    writer = pd.ExcelWriter(file_path)
    if isinstance(datas, dict):
        # datas = {"col1": [1, 1], "col2": [2, 2]}
        df1 = pd.DataFrame(datas)
        df1.to_excel(writer, "Sheet1", index=False)
    elif isinstance(datas, list):
        # 二维数组型数据
        df = pd.DataFrame.from_records(list(datas))
        df.to_excel(
            writer, "Sheet1", index=False, header=header
        )
    writer.save()


def read_csv(file_path):
    csvFile = open(file_path)
    csv_reader = csv.reader(csvFile)
    c2_corpus = []
    for line in csv_reader:
        # 忽略第一行
        if csv_reader.line_num == 1:
            continue
        primary_question, similar_question, flow_code, key_word = line[1], line[2], line[4], line[5]
        if flow_code=="C2ZXKF":
            c2_corpus.append(primary_question)
            c2_corpus.extend([_ for _ in similar_question.split("###") if _])
    csvFile.close()
    return c2_corpus
