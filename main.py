import sys
from PyQt5 import QtWidgets
import numpy as np
import os
import pydicom
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
import matplotlib.pyplot as plt
from skimage.io import imsave, imread

from functions import preprocess, grad_cam
from medical_ui import Ui_Medicalanalysis
from models import MainModel

is_windows = os.name == 'nt'
path_sign = '\\' if is_windows else '/'

BASE = os.path.dirname(os.path.abspath(__file__))
DENSENET_PATH = "/Users/wangzhongxuan/0QIU/trained_models/model_densenet201.pt"
RESNET_PATH = "/Users/wangzhongxuan/0QIU/trained_models/model_densenet201.pt"
VGG_PATH = "/Users/wangzhongxuan/0QIU/trained_models/model_densenet201.pt"

if (is_windows):
    DENSENET_PATH.replace('/', '\\')
    RESNET_PATH.replace('/', '\\')
    VGG_PATH.replace('/', '\\')

MODEL_PATH = [DENSENET_PATH, RESNET_PATH, VGG_PATH]

DenseNet_Model = MainModel('densenet201', 6)
Resnet_Model = MainModel('resnet101', 6)
VGG_Model = MainModel('vgg19', 6)
Models = [DenseNet_Model, Resnet_Model, VGG_Model]

ct_mean = 0.188
ct_std = 0.315

# state_dict = torch.load(PATH_MODEL)

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
        self.image = None

        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        # 设置一些linetext为不可写
        self.foldname.setDisabled(True)
        self.analysis.setDisabled(True)

        # 设置gui控件与function关联
        self.fold_select.clicked.connect(self.folddialog)
        self.imgfile.activated.connect(self.start_analysis)
        self.languages.activated.connect(self.change_language)
        self.analysis.clicked.connect(self.start_prediction)

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
        self.console.append(f"[INFO] Currently Directory is set to: {fold}")
        # 确定是否为有效的文件夹
        if os.path.isdir(fold):
            self.foldname.setText(fold)
            self.fold = fold
            self.imgfile.clear()
            # 设置读取文件的格式
            file_types = ["dcm", "png", "jpg"]
            files = [f for f in os.listdir(fold) if f[-3:] in file_types]
            if len(files) == 0:
                self.console.append("[WARNING] There is no jpg / png / dcm format files in this directory.")
            else:
                files.sort()

                # 将文件加入文件选项控件
                self.imgfile.addItems(files)
                self.analysis.setDisabled(False)
                self.start_analysis()
        else:
            self.console.append("[WARNING] Please choose a folder to start.")

    # 开始对dicom文件解析，绘图
    def start_analysis(self):
        # 移除旧的画布
        try:
            self.imageshow.removeWidget(self.plot_figure)
            self.imageshow.removeWidget(self.tool)
        except:
            self.imageshow.removeWidget(self.photoview)
        # finally:
        #     pass

        filename = self.imgfile.currentText()
        filepath = os.path.join(self.fold, filename)
        self.console.append(f"[INFO] Showing:{filename}")

        self.filepath = filepath
        self.plot_figure = ImageView(width=8, height=8, dpi=110)
        self.tool = NavigationToolbar2QT(self.plot_figure, self)
        if filepath[-3:] == "dcm":
            # FIXME
            # 读取dicom文件
            _slice = preprocess(filepath)

            # 设置id, 长， 宽
            self.id.setText(_slice.PatientID)
            self.length.setText(str(_slice.pixel_array.shape[1]))
            self.width.setText(str(_slice.pixel_array.shape[0]))

            # 获取dicom数据，绘图
            self.image = np.stack(_slice.pixel_array)
            self.image = self.image.astype(np.int16)
            self.image = np.array(self.image)
        else:
            self.image = plt.imread(filepath)
            self.id.setText(" ")
            self.length.setText(str(self.image.shape[1]))
            self.width.setText(str(self.image.shape[0]))

        plt.imshow(self.image, cmap=plt.cm.gray)

        self.imageshow.addWidget(self.plot_figure)
        self.imageshow.addWidget(self.tool)
        _img = filepath.split(path_sign)[-1]
        # 显示大标题
        self.plot_figure.fig.suptitle(_img)
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
            self.label_4.setText("模型选择:")
            self.analysis.setText("开始诊断")
            self.rawimg.setText("原图")
            self.t1.setText("硬膜外阻滞")
            self.t2.setText("脑实质")
            self.t3.setText("脑室内")
            self.t4.setText("膜下腔")
            self.t5.setText("硬脑膜下")

    # 程序model分析模块
    def start_prediction(self):
        self.console.append("[INFO] Process starts")
        try:
            self.resultlayer.removeWidget(self.result_figure)
        except:
            self.resultlayer.removeWidget(self.resultview)
        finally:
            pass
        filepath = self.filepath
        image = None
        # try:
        if filepath[-3:] == "dcm":
            # 读取dicom文件
            self.console.append(f"[INFO] Preprocessing {filepath.split(path_sign)[-1]} Image...")
            image = preprocess(filepath)
        else:
            image = imread(filepath)
        # except:
        #     self.console.append('ERROR Encountered while processing')
        #     return

        image = torch.tensor(image[None, None, ...], dtype=torch.float32) / 255
        image = (image - ct_mean) / ct_std
        try:
            image = image.expand(-1, 3, -1, -1)
        except:
            self.console.append('[ERROR] Unable to expand the image with shape ' + str(image.shape))
        current_model_index = self.models.currentIndex()



        results = {
            "Any": 0.0,
            "Epidural": 0.0,
            "Intraparenchymal": 0.0,
            "Intraventricular": 0.0,
            "Subarachnoid": 0.0,
            "Subdural": 0.0,
        }
        i = 0

        # for (key, item) in results.items():
        #     results[key] = round(result_values[i].item(), 2)
        #     # results[key] = round(prabobility_[i] * 100, 2)
        #
        #     i += 1

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
        self.console.append("Results:")
        self.t1_t5()

    def show_result_img(self, signal):

        if self.image is None:
            self.console.append('[WARNING] Image is not selected')
            return

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
        elif signal == 5:
            self.t5.setDisabled(True)
        else:
            raise Exception('Incorrect Signal Tossed!')

        self.plot_figure = ImageView(width=8, height=8, dpi=110)
        self.tool = NavigationToolbar2QT(self.plot_figure, self)

        selected_index = self.models.currentIndex()
        selected_model = Models[selected_index]
        if not selected_model.loaded:
            selected_model.load_state_dict(torch.load(MODEL_PATH[selected_index]))
        image = grad_cam(self.image, signal, selected_model)

        if image.shape[0] == 3:
            image = image[0]
        plt.imshow(image)

        self.imageshow.addWidget(self.plot_figure)
        self.imageshow.addWidget(self.tool)
        # 显示大标题
        name_list = [
            "Any",
            "Epidural",
            "Intraparenchymal",
            "Intraventricular",
            "Subarachnoid",
            "Subdural",
        ]
        if signal == 1:
            title = f"{name_list[signal]} - Not Diagnosed"
        else:
            title = f"{name_list[signal]} - Diagnosed"
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
