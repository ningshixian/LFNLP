import os
import time
from flask import Flask
from flask import request, url_for, make_response, send_from_directory
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.mlab as mlab
import matplotlib.image as mpimg
from matplotlib.pyplot import MultipleLocator
import numpy as np
import base64
import codecs
import traceback


"""
AskData的画图接口
测试环境：http://10.231.135.146:10086/plot
Request参数:
{
    "chart": {
        "type": "bar",
        "stack": false
    },
    "labels": [
        "",
    ],
    "dataset": [
        [
            "xxxx项目",
            19000
        ]
    ]
}
"""


# # 设置static静态目录
# app = Flask(__name__, static_folder="./static")

# 解决中文乱码问题
myfont = fm.FontProperties(fname="font/Songti.ttc", size=14)
matplotlib.rcParams["axes.unicode_minus"] = False

x_major_locator = MultipleLocator(1)
# 把x轴的刻度间隔设置为1，并存在变量里
y_major_locator = MultipleLocator(10)
# 把y轴的刻度间隔设置为10，并存在变量里
ax = plt.gca()
# ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# 把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)


class Plot4Parser(object):
    def __init__(self, dataset, labels, min_d, max_d):
        self.dataset = dataset
        self.labels = labels
        self.min_d = 0 if min_d >= 0 else min_d * 1.2
        self.max_d = max_d * 1.2  # 开根号×
        self.colors = ["yellowgreen", "gold", "lightskyblue", "lightcoral", "r", "g", "b"]  # --
        self.longth = 12
        self.width = 10

    def line_plot(self):
        """
        simple plot
        """
        # 生成画布，并设定标题
        plt.figure(figsize=(self.longth, self.width), dpi=80)
        plt.title("简单曲线图", fontproperties=myfont)
        plt.grid(True)

        # 设置X轴
        plt.xlabel(self.labels[0], fontproperties=myfont)
        plt.xlim(-1, len(self.dataset))  # 取值范围 [0, len-1]
        plt.xticks(
            range(len(self.dataset)), [x[0] for x in self.dataset], fontproperties=myfont, rotation=10
        )  # 坐标轴单位转换

        # 设置Y轴
        plt.ylabel("/".join(self.labels[1:]), fontproperties=myfont)
        plt.ylim(self.min_d, self.max_d)
        # plt.yticks(np.linspace(-1, 1, 9, endpoint=True))

        # 画多条曲线
        for i in range(1, len(self.labels)):
            x = np.arange(len(self.dataset))
            y = np.array([_[i] for _ in self.dataset])
            plt.plot(
                x,
                y,
                self.colors[i - 1],
                linewidth=1.5,
                label=self.labels[i],
                marker="o",
                markerfacecolor="blue",
                markersize=5,
            )
            # 折线图显示数据
            for j in range(len(self.dataset)):
                d = self.dataset[j]
                plt.text(j, int(d[i]), d[i], ha="center", va="bottom", fontsize=10)
        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="upper right", prop=myfont, shadow=True)
        plt.tight_layout()  # 自动调整绘图区的大小及间距

        # 图形显示
        # plt.show()
        img_local_path = "static/img_{}.png".format(time.strftime(r"%Y%m%d%H%M%S", time.localtime()))
        plt.savefig(img_local_path)
        return img_local_path

    # def simple_advanced_plot(self):
    #     """
    #     simple advanced plot
    #     """
    #     # 生成测试数据
    #     x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    #     y_cos, y_sin = np.cos(x), np.sin(x)

    #     # 生成画布, 并设定标题
    #     plt.figure(figsize=(8, 6), dpi=80)
    #     plt.title("复杂曲线图", fontproperties=myfont)
    #     plt.grid(True)

    #     # 画图的另外一种方式
    #     ax_1 = plt.subplot(111)
    #     ax_1.plot(x, y_cos, color="blue", linewidth=2.0, linestyle="--", label="左cos")
    #     ax_1.legend(loc="upper right", prop=myfont, shadow=True)

    #     # 设置Y轴(左边)
    #     ax_1.set_ylabel("左cos的y轴", fontproperties=myfont)
    #     ax_1.set_ylim(-1.0, 1.0)
    #     ax_1.set_yticks(np.linspace(-1, 1, 9, endpoint=True))

    #     # 画图的另外一种方式
    #     ax_2 = ax_1.twinx()
    #     ax_2.plot(x, y_sin, color="green", linewidth=2.0, linestyle="-", label="右sin")
    #     ax_2.legend(loc="upper right", prop=myfont, shadow=True)

    #     # 设置Y轴(右边)
    #     ax_2.set_ylabel("右sin的y轴", fontproperties=myfont)
    #     ax_2.set_ylim(-2.0, 2.0)
    #     ax_2.set_yticks(np.linspace(-2, 2, 9, endpoint=True))

    #     # 设置X轴(共同)
    #     ax_1.set_xlabel("x轴", fontproperties=myfont)
    #     ax_1.set_xlim(-4.0, 4.0)
    #     ax_1.set_xticks(np.linspace(-4, 4, 9, endpoint=True))

    #     # 图形显示
    #     # plt.show()
    #     return

    def bar_plot(self):
        # 生成画布，并设定标题
        plt.figure(figsize=(self.longth, self.width), dpi=80)
        plt.title("柱状图", fontproperties=myfont)
        plt.grid(True)

        # 设置X轴
        plt.xlabel(self.labels[0], fontproperties=myfont)
        plt.xlim(-1, len(self.dataset))  # 设置刻度范围 [0, len-1]
        plt.xticks(
            range(len(self.dataset)), [x[0] for x in self.dataset], fontproperties=myfont, rotation=10
        )  # 设置坐标轴名称

        # 设置Y轴
        plt.ylabel("/".join(self.labels[1:]), fontproperties=myfont)
        plt.ylim(self.min_d, self.max_d)
        # plt.yticks(np.linspace(-1, 1, 9, endpoint=True))

        # 设置相关参数
        bar_width = 0.35

        # 画多条曲线
        for i in range(1, len(self.labels)):
            x = np.arange(len(self.dataset)) + bar_width * (i - 1)
            y = np.array([_[i] for _ in self.dataset])
            plt.bar(x, y, width=bar_width, alpha=0.2, color=self.colors[i - 1], label=self.labels[i])
            # 折线图显示数据
            for j in range(len(self.dataset)):
                d = self.dataset[j]
                plt.text(j + bar_width * (i - 1), d[i] + 0.3, d[i], ha="center", va="bottom", fontsize=10)

        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="upper right", prop=myfont, shadow=True)
        plt.tight_layout()  # 自动调整绘图区的大小及间距

        # 图形显示
        # plt.show()
        img_local_path = "static/img_{}.png".format(time.strftime(r"%Y%m%d%H%M%S", time.localtime()))
        plt.savefig(img_local_path)
        return img_local_path

    def barh_plot(self):
        # 生成画布，并设定标题
        plt.figure(figsize=(self.longth, self.width), dpi=80)
        plt.title("横向柱状图", fontproperties=myfont)
        # plt.grid(True)

        # 设置y轴
        plt.ylabel(self.labels[0], fontproperties=myfont)
        plt.ylim(-1, len(self.dataset) - 0.5)  # 设置刻度范围 [0, len-1]
        plt.yticks(
            range(len(self.dataset)), [x[0] for x in self.dataset], fontproperties=myfont, rotation=10
        )  # 设置坐标轴名称

        # 设置x轴
        plt.xlabel("/".join(self.labels[1:]), fontproperties=myfont)
        plt.xlim(self.min_d, self.max_d)
        # plt.xticks(np.linspace(-1, 1, 9, endpoint=True))

        # 设置相关参数
        bar_height = 0.35

        # 画柱状图(水平方向)
        for i in range(1, len(self.labels)):
            x = np.arange(len(self.dataset)) - bar_height * (i - 1)
            y = np.array([_[i] for _ in self.dataset])
            alpha = 0.2 if i == 1 else 0.8
            plt.barh(x, y, height=bar_height, alpha=alpha, color=self.colors[i - 1], label=self.labels[i])
            for j in range(len(self.dataset)):
                d = self.dataset[j]
                plt.text(d[i] + 2, j - bar_height * (i - 1), d[i], ha="center", va="bottom", fontsize=10)

        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="upper right", prop=myfont, shadow=True)
        plt.tight_layout()  # 自动调整绘图区的大小及间距

        # 图形显示
        # plt.show()
        img_local_path = "static/img_{}.png".format(time.strftime(r"%Y%m%d%H%M%S", time.localtime()))
        plt.savefig(img_local_path)
        return img_local_path

    def barv_plot(self):
        return

    def table_plot(self):
        # 生成画布，并设定标题
        plt.figure(figsize=(self.longth, self.width), dpi=80)
        plt.title("层次柱状图", fontproperties=myfont)
        # plt.grid(True)

        # 设置x轴
        plt.xlabel(self.labels[0], fontproperties=myfont)
        plt.xlim(-1, len(self.dataset) - 0.5)  # 设置刻度范围 [0, len-1]
        plt.xticks(
            range(len(self.dataset)), [x[0] for x in self.dataset], fontproperties=myfont, rotation=10
        )  # 设置坐标轴名称

        # 设置y轴
        plt.ylabel("/".join(self.labels[1:]), fontproperties=myfont)
        plt.ylim(self.min_d, self.max_d)
        # plt.yticks(np.linspace(-1, 1, 9, endpoint=True))

        # 声明底部位置
        bottom = np.array([0.0] * len(self.dataset))

        # 依次画图,并更新底部位置
        for i in range(1, len(self.labels)):
            x = np.arange(len(self.dataset))
            y = np.array([_[i] for _ in self.dataset])
            plt.bar(x, y, width=0.5, alpha=0.8, color=self.colors[i - 1], bottom=bottom, label=self.labels[i])
            bottom += y
            # for j in range(len(dataset)):
            #     d = dataset[j]
            #     plt.text(d[i], j, d[i], ha="center", va="bottom", fontsize=10)

        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="upper right", prop=myfont, shadow=True)
        plt.tight_layout()  # 自动调整绘图区的大小及间距

        # 图形显示
        # plt.show()
        img_local_path = "static/img_{}.png".format(time.strftime(r"%Y%m%d%H%M%S", time.localtime()))
        plt.savefig(img_local_path)
        return img_local_path

    def histograms_plot(self, heights):
        """直方图"""
        # 第一个参数为待绘制的定量数据，不同于定性数据，这里并没有事先进行频数统计
        # 第二个参数为划分的区间个数
        plt.hist(heights, 100)
        plt.xlabel('Heights')
        plt.ylabel('Frequency')
        plt.title('Heights Of Male Students')
        plt.show()

    def pie_plot(self):
        """
        pie plot
        """
        nrows = 1 if len(self.labels[1:]) <= 2 else 2
        ncols = 2
        # 依次画图
        for i in range(1, len(self.labels)):
            # 生成测试数据
            total = sum([x[i] for x in self.dataset])
            sizes = [x[i] / total for x in self.dataset]
            labels = [x[0] for x in self.dataset]
            colors = self.colors[: len(self.dataset)]

            ax = plt.subplot(nrows, ncols, i)
            plt.title(self.labels[i], fontproperties=myfont)
            patches, l_text, p_text = ax.pie(
                sizes, labels=labels, colors=colors, autopct="%1.1f%%", shadow=True, startangle=90
            )
            for text in l_text:
                text.set_fontproperties(myfont)
            ax.axis("equal")

        # 图形显示
        # plt.show()
        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="upper right", prop=myfont, shadow=True)
        plt.tight_layout()  # 自动调整绘图区的大小及间距

        img_local_path = "static/img_{}.png".format(time.strftime(r"%Y%m%d%H%M%S", time.localtime()))
        plt.savefig(img_local_path)
        return img_local_path

    def scatter_plot(self):
        """
        scatter plot
        """
        # 生成画布，并设定标题
        plt.figure(figsize=(self.longth, self.width), dpi=80)
        plt.title("散点图", fontproperties=myfont)
        # plt.grid(True)

        # 设置x轴
        plt.xlabel(self.labels[0], fontproperties=myfont)
        plt.xlim(-1, len(self.dataset) - 0.5)  # 设置刻度范围 [0, len-1]
        plt.xticks(
            range(len(self.dataset)), [x[0] for x in self.dataset], fontproperties=myfont, rotation=10
        )  # 设置坐标轴名称

        # 设置y轴
        plt.ylabel("/".join(self.labels[1:]), fontproperties=myfont)
        plt.ylim(self.min_d, self.max_d)
        # plt.yticks(np.linspace(-1, 1, 9, endpoint=True))

        # 设置相关参数
        color_list = np.random.random(len(self.dataset))
        scale_list = np.random.random(len(self.dataset)) * 100

        # 依次画图,并更新底部位置
        for i in range(1, len(self.labels)):
            x = np.arange(len(self.dataset))
            y = np.array([_[i] for _ in self.dataset])
            plt.scatter(x, y, s=scale_list, c=color_list, marker="o")
            for j in range(len(self.dataset)):
                d = self.dataset[j]
                plt.text(d[i], j, d[i], ha="center", va="bottom", fontsize=10)

        # 图形显示
        # plt.show()
        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="upper right", prop=myfont, shadow=True)
        plt.tight_layout()  # 自动调整绘图区的大小及间距

        img_local_path = "static/img_{}.png".format(time.strftime(r"%Y%m%d%H%M%S", time.localtime()))
        plt.savefig(img_local_path)
        return img_local_path

    def area_plot(self):
        """
        area plot
        """
        # 生成画布，并设定标题
        plt.figure(figsize=(self.longth, self.width), dpi=80)
        plt.title("面积图", fontproperties=myfont)

        # 设置x轴
        plt.xlabel(self.labels[0], fontproperties=myfont)
        plt.xticks(
            range(len(self.dataset)), [x[0] for x in self.dataset], fontproperties=myfont, rotation=10
        )  # 设置坐标轴名称
        # 设置y轴
        plt.ylabel("/".join(self.labels[1:]), fontproperties=myfont)

        # 面积图
        colors = self.colors[: len(self.dataset)]
        x = np.arange(len(self.dataset))
        if len(self.dataset[0]) == 2:
            y = [x[1] for x in self.dataset]
        elif len(self.dataset[0]) == 3:
            y = [x[1] for x in self.dataset], [x[2] for x in self.dataset]
        else:
            y = ([x[1] for x in self.dataset], [x[2] for x in self.dataset], [x[3] for x in self.dataset])
        plt.stackplot(x, y, labels=self.labels[1:], colors=colors)

        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="upper right", prop=myfont, shadow=True)
        plt.tight_layout()  # 自动调整绘图区的大小及间距

        # 图形显示
        # plt.show()
        img_local_path = "static/img_{}.png".format(time.strftime(r"%Y%m%d%H%M%S", time.localtime()))
        plt.savefig(img_local_path)
        return img_local_path


def test_plot():
    dataset = [["a", 19.0], ["b", 42.0], ["c", 17.0], ["d", 114.5]]
    labels = ["项目名称", "签约货值"]  # , "供货货值"
    val_list = []
    for x in dataset:
        val_list.extend(x[1:])
    min_d = min(val_list)
    max_d = max(val_list)
    plotter = Plot4Parser(dataset, labels, min_d, max_d)
    # plotter.line_plot()
    # plotter.barh_plot()
    # plotter.bar_plot()
    # plotter.table_plot()
    # plotter.pie_plot()
    # plotter.scatter_plot()
    plotter.area_plot()


# test_plot()
# exit()


# def return_img_stream(img_local_path):
#     """
#     返回图片流给前端展示
#     :param img_local_path:文件单张图片的本地绝对路径
#     :return: 图片流
#     """
#     img_stream = ""
#     # img_stream = mpimg.imread(img_local_path)
#     # img_stream = base64.b64encode(img_stream)

#     with codecs.open(img_local_path, "rb") as img_f:
#         img_stream = img_f.read()
#         img_stream = base64.b64encode(img_stream)
#     return str(img_stream, encoding="utf-8")
