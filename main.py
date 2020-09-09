import sys
from PyQt5 import QtWidgets
from medical_ui import Ui_Medicalanalysis
import os
import numpy as np
import pydicom
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
import matplotlib.pyplot as plt
from predict_model import get_predict, BASE

# matplotlib绘图画布
class ImageView(FigureCanvas):
    def __init__(self, width, height, dpi):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super(ImageView, self).__init__(self.fig)


# 主程序类
class Main(QtWidgets.QMainWindow, Ui_Medicalanalysis):
    """ 这是程序主类，用于初始化控件，将控件关联functions，做数据处理。 """

    # 对程序gui初始化
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        # 设置一些linetext为不可写
        self.foldname.setDisabled(True)
        self.analysis.setDisabled(True)

        # 设置gui控件与function关联
        self.fold_select.clicked.connect(self.folddialog)
        self.imgfile.activated.connect(self.start_analysis)
        self.languages.activated.connect(self.change_language)
        self.analysis.clicked.connect(self.model_analysis)

        self.rawimg.clicked.connect(lambda: self.show_result_img(0))
        self.t1.clicked.connect(lambda: self.show_result_img(1))
        self.t2.clicked.connect(lambda: self.show_result_img(2))
        self.t3.clicked.connect(lambda: self.show_result_img(3))
        self.t4.clicked.connect(lambda: self.show_result_img(4))
        self.t5.clicked.connect(lambda: self.show_result_img(5))

        # 添加语言选项
        self.languages.addItem("中文")
        self.languages.addItem("English")

        # 添加model选项
        self.models.addItems(["Resnet-101", "Densenet-201", "VGG-19"])
        self.models.setCurrentIndex(1)

    # 打开文件夹对话框
    def folddialog(self):
        fold = QtWidgets.QFileDialog.getExistingDirectory(self)
        self.console.append(f"Currently fold -- {fold}")
        # 确定是否为有效的文件夹
        if os.path.isdir(fold):
            self.foldname.setText(fold)
            self.fold = fold
            self.imgfile.clear()
            # 设置读取文件的格式
            file_types = ["dcm", "png", "jpg"]
            files = [f for f in os.listdir(fold) if f[-3:] in file_types]
            if len(files) == 0:
                self.console.append("There is no image files.")
            else:
                files.sort()

                # 将文件加入文件选项控件
                self.imgfile.addItems(files)
                self.analysis.setDisabled(False)
                self.start_analysis()
        else:
            self.console.append("Chose a true fold.")

    # 开始对dicom文件解析，绘图
    def start_analysis(self):
        # 移除旧的画布
        try:
            self.imageshow.removeWidget(self.plot_figure)
            self.imageshow.removeWidget(self.tool)
        except:
            self.imageshow.removeWidget(self.photoview)
        finally:
            pass

        filename = self.imgfile.currentText()
        filepath = os.path.join(self.fold, filename)
        self.filepath = filepath
        self.plot_figure = ImageView(width=8, height=8, dpi=110)
        self.tool = NavigationToolbar2QT(self.plot_figure, self)
        if filepath[-3:] == "dcm":
            # 读取dicom文件
            _slice = pydicom.read_file(filepath)

            # 设置id, 长， 宽
            self.id.setText(_slice.PatientID)
            self.length.setText(str(_slice.pixel_array.shape[1]))
            self.width.setText(str(_slice.pixel_array.shape[0]))

            # 获取dicom数据，绘图
            image = np.stack(_slice.pixel_array)
            image = image.astype(np.int16)
            image = np.array(image)
        else:
            image = plt.imread(filepath)
            self.id.setText(" ")
            self.length.setText(str(image.shape[1]))
            self.width.setText(str(image.shape[0]))

        plt.imshow(image, cmap=plt.cm.gray)

        self.imageshow.addWidget(self.plot_figure)
        self.imageshow.addWidget(self.tool)
        _img = filepath.split("/")[-1]
        # 显示大标题
        self.plot_figure.fig.suptitle(_img)
        self.console.append(f"Start Precessing image:{_img}")
        self.t1_t5()

    # 程序语言，中文，英文
    def change_language(self):
        if self.languages.currentText() == "English":
            self.setWindowTitle("ICH deep learning detector")
            self.title.setText("ICH deep learning detector")
            try:
                self.foldname.setText(self.fold)
            except:
                self.foldname.setText("Fold Name")
            self.length_label.setText("Length")
            self.width_label.setText("Width")
            self.label_4.setText("Models")
            self.analysis.setText("Start")
            self.rawimg.setText("Any")
            self.t1.setText("Epidural")
            self.t2.setText("Intraparenchymal")
            self.t3.setText("Intraventricular")
            self.t4.setText("Subarachnoid")
            self.t5.setText("Subdural")
        else:
            self.setWindowTitle("医学影像分析")
            self.title.setText("医学影像诊断分析")
            try:
                self.foldname.setText(self.fold)
            except:
                self.foldname.setText("影像目录")
            self.length_label.setText("长")
            self.width_label.setText("宽")
            self.label_4.setText("模型选择：")
            self.analysis.setText("开始诊断")
            self.rawimg.setText("原图")
            self.t1.setText("硬膜外阻滞")
            self.t2.setText("脑实质")
            self.t3.setText("脑室内")
            self.t4.setText("膜下腔")
            self.t5.setText("硬脑膜下")

    # 程序model分析模块
    def model_analysis(self):
        self.console.append("Start detecting...")
        # 绘图
        try:
            self.resultlayer.removeWidget(self.result_figure)
        except:
            self.resultlayer.removeWidget(self.resultview)
        finally:
            pass
        # 获取model分析数据..
        filepath = self.filepath
        _img = filepath.split("/")[-1]
        if filepath[-3:] == "dcm":
            # 读取dicom文件
            _slice = pydicom.read_file(filepath)
            result_values = get_predict(_slice, flag=0)

        else:
            result_values = get_predict(filepath, flag=1)

        results = {
            "Any": 18.00,
            "Epidural": 32.00,
            "Intraparenchymal": 43.00,
            "Intraventricular": 17.00,
            "Subarachnoid": 90.00,
            "Subdural": 21.00,
        }
        i = 0

        prabobility_ = [
            1.0000e00,
            7.6311e-05,
            9.9868e-01,
            9.9763e-01,
            8.1377e-01,
            9.8775e-01,
        ]

        for (key, item) in results.items():
            results[key] = round(result_values[i].item(), 2)
            # results[key] = round(prabobility_[i] * 100, 2)

            i += 1

        self.result_figure = ImageView(width=3, height=2, dpi=80)
        labels = list(results.keys())
        data = np.array(list(results.values()))

        self.result_figure.ax.invert_yaxis()
        self.result_figure.ax.xaxis.set_visible(False)
        self.result_figure.ax.set_xlim(0, 220)
        self.result_figure.ax.barh(labels, data, left=0, height=0.5)
        self.result_figure.ax.set_yticklabels([])
        self.result_figure.ax.set_title("Probability")
        i = 0
        for (key, item) in results.items():
            self.result_figure.ax.annotate(
                f"{item}% -- {labels[i]}",
                xy=(item, i + 0.5 / 2),
                xytext=(3, 3),  # 3 points vertical offset
                textcoords="offset points",
                color="black",
                ha="left",
                va="center",
            )
            i += 1

        self.resultlayer.addWidget(self.result_figure)
        self.console.append("Results come out")
        self.t1_t5()

    def show_result_img(self, signal):
        # 移除旧的画布
        try:
            self.imageshow.removeWidget(self.plot_figure)
            self.imageshow.removeWidget(self.tool)
        except:
            self.imageshow.removeWidget(self.photoview)
        finally:
            pass

        self.t1_t5()
        if signal == 0:
            self.rawimg.setDisabled(True)
        elif signal == 1:
            self.t1.setDisabled(True)
        elif signal == 2:
            self.t2.setDisabled(True)
        elif signal == 3:
            self.t3.setDisabled(True)
        elif signal == 4:
            self.t4.setDisabled(True)

        else:
            self.t5.setDisabled(True)

        self.plot_figure = ImageView(width=8, height=8, dpi=110)
        self.tool = NavigationToolbar2QT(self.plot_figure, self)

        image = plt.imread(BASE + "/data/" + str(signal) + ".png")
        self.id.setText("631d342a4")
        self.length.setText(str(image.shape[1]))
        self.width.setText(str(image.shape[0]))

        plt.imshow(image)

        self.imageshow.addWidget(self.plot_figure)
        self.imageshow.addWidget(self.tool)
        # 显示大标题
        name_list = [
            "any",
            "epidural",
            "intraparenchymal",
            "intraventricular",
            "subarachnoid",
            "subdural",
        ]
        if signal == 1:
            title = f"{name_list[signal]} - NO"
        else:
            title = f"{name_list[signal]} - YES"
        self.plot_figure.fig.suptitle(title)

    def t1_t5(self):
        self.rawimg.setDisabled(False)
        self.t1.setDisabled(False)
        self.t2.setDisabled(False)
        self.t3.setDisabled(False)
        self.t4.setDisabled(False)
        self.t5.setDisabled(False)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
