import pkuseg

"""
一个多领域中文分词工具包
1、支持了新闻领域，网络领域，医药领域，旅游领域
2、更高的分词准确率
3、支持用户自训练模型
4、支持词性标注

https://github.com/lancopku/pkuseg-python
"""


# 分词同时进行词性标注
# 下载postag.zip，解压到 postag\
seg = pkuseg.pkuseg(postag=True)    # , user_dict="./new_word.txt"
text = seg.cut("我爱北京天安门")  # 进行分词
print(text)

# # 对input.txt的文件分词输出到output.txt中
# # 开20个进程
# pkuseg.test('input.txt', 'output.txt', nthread=20)

# seg = pkuseg.pkuseg(model_name='medicine')  # 程序会自动下载所对应的细领域模型
# text = seg.cut('我爱北京天安门')              # 进行分词
# print(text)
