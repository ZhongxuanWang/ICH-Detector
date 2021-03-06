import sys
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
import matplotlib.pyplot as plt

from functions import *
from medical_ui import Ui_Medicalanalysis
from models import *

from copy import deepcopy

is_windows = os.name == 'nt'
path_sign = '\\' if is_windows else '/'

BASE = os.path.dirname(os.path.abspath(__file__))
DENSENET_PATH = "/Users/wangzhongxuan/0QIU/trained_models/model_densenet201.pt"
RESNET_PATH = "/Users/wangzhongxuan/0QIU/trained_models/model_resnet101.pt"
VGG_PATH = "/Users/wangzhongxuan/0QIU/trained_models/vgg19/model_vgg19.pt"

DENSENET_GRADCAM_PATH = '/Users/wangzhongxuan/0QIU/dn201_gradcam.pt'

if is_windows:
    DENSENET_PATH.replace('/', '\\')
    RESNET_PATH.replace('/', '\\')
    VGG_PATH.replace('/', '\\')
    DENSENET_GRADCAM_PATH.replace('/', '\\')

MODEL_PATH = [DENSENET_PATH, RESNET_PATH, VGG_PATH]

DenseNet_Model = MainModel('densenet201', 6)
Resnet_Model = MainModel('resnet101', 6)
VGG_Model = MainModel('vgg19', 6)
DenseNet_GradCAM_Model = GradCAMModel()
Models = [DenseNet_Model, Resnet_Model, VGG_Model, DenseNet_GradCAM_Model]

# Threshold of whether is diagnosed gained from ROC Curve
THRESHOLD = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
             ]  # FIXME: Get the curve, and get the threshold from there. 0.5 IS NOT THE OPTIMAL ONE!

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
    def __init__(self):
        self.fold = None
        self.start_btn_hit = False
        self.image = None
        self.plot_figure = None
        self.flag = [None, None, None, None, None, None]

        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        # 设置一些linetext为不可写
        self.foldname.setDisabled(True)
        self.analysis.setDisabled(True)

        # 设置gui控件与function关联
        self.fold_select.clicked.connect(self.folddialog)
        self.imgfile.activated.connect(self.show_fig)
        self.languages.activated.connect(self.change_language)
        self.analysis.clicked.connect(self.start_prediction_btn_clicked)

        self.original.clicked.connect(lambda: self.show_fig())
        self.rawimg.clicked.connect(lambda: self.show_gradcam(0))
        self.t1.clicked.connect(lambda: self.show_gradcam(1))
        self.t2.clicked.connect(lambda: self.show_gradcam(2))
        self.t3.clicked.connect(lambda: self.show_gradcam(3))
        self.t4.clicked.connect(lambda: self.show_gradcam(4))
        self.t5.clicked.connect(lambda: self.show_gradcam(5))

        self.gradcam_analysis.clicked.connect(lambda: self.gradcam_analysis_btn_clicked())

        # 添加语言选项
        self.languages.addItem("中文")
        self.languages.addItem("English")

        # 添加model选项
        self.models.addItems(["DenseNet-201", "Resnet-101", "VGG-19"])
        self.models.setCurrentIndex(0)

        self.change_to_english()

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
                self.show_fig()
        else:
            self.console.append("[WARNING] Please choose a folder to start.")

    # 开始对dicom文件解析，绘图
    def show_fig(self):
        if self.fold is None:
            self.console.append('[ERROR] You haven\'t opened an image!')
            return
        self.start_btn_hit = False
        try:
            self.resultlayer.removeWidget(self.result_figure)
        except:
            self.resultlayer.removeWidget(self.resultview)
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

        plt.close('all')

        self.plot_figure = ImageView(width=8, height=8, dpi=110)
        self.tool = NavigationToolbar2QT(self.plot_figure, self)
        if filepath[-3:] == "dcm":
            # 读取dicom文件
            _slice = preprocess(filepath)

            # 设置id, 长， 宽
            self.id.setText(_slice.PatientID)
            self.length.setText(str(_slice.pixel_array.shape[1]))
            self.width.setText(str(_slice.pixel_array.shape[0]))
            self.image = _slice
        else:
            self.image = plt.imread(filepath)
            # if self.image.ndim == 3:
            #     self.image = rgb_to_grey(self.image)
            # self.id.setText(" ")
            self.length.setText(str(self.image.shape[1]))
            self.width.setText(str(self.image.shape[0]))
        self.id.setText(str(self.image.shape))

        # while len(self.image) == 1:
        #     self.image = self.image[0]
        self.imageshow.addWidget(self.plot_figure)
        self.imageshow.addWidget(self.tool)
        _img = filepath.split(path_sign)[-1]
        if len(self.image.shape) == 3:
            plt.imshow(self.image)
        else:
            plt.imshow(self.image, plt.gray())

        # plt.imshow(np.around(torch.clamp(jet(image.expand(1,3,-1,-1)), 0, 1)[0,0].cpu().numpy()*255).astype(np.uint8))

        # 显示大标题
        self.plot_figure.fig.suptitle(_img)
        self.t1_t5()

    # 程序语言，中文，英文
    def change_to_english(self):
        self.languages.setCurrentIndex(1)
        self.change_language()

    def change_language(self):
        if self.languages.currentText() == "English":
            self.setWindowTitle("ICH deep learning detector")
            self.title.setText("ICH deep learning detector")
            try:
                self.foldname.setText(self.fold)
            except:
                self.foldname.setText("Folder")
            self.id_label.setText("Shape")
            self.length_label.setText("Slope")
            self.width_label.setText("Intercept")
            self.label_4.setText("Model")
            self.gradcam_analysis.setText('GradCam')
            self.analysis.setText("Diagnose")
            self.original.setText('Original')
            self.rawimg.setText("Any")
            self.t1.setText("Epidural")
            self.t2.setText("Intraparenchymal")
            self.t3.setText("Intraventricular")
            self.t4.setText("Subarachnoid")
            self.t5.setText("Subdural")
        else:
            self.setWindowTitle("脑出血CT使用GradCAM的深度学习诊断程序")
            self.title.setText("脑出血CT使用GradCAM的深度学习诊断程序")
            try:
                self.foldname.setText(self.fold)
            except:
                self.foldname.setText("目录")
            self.length_label.setText("Slope")
            self.width_label.setText("Intercept")
            self.label_4.setText("模型")
            self.gradcam_analysis.setText('热力图诊断')
            self.analysis.setText("原图诊断")
            self.original.setText('原图')
            self.rawimg.setText("总")
            self.t1.setText("硬膜外阻滞")
            self.t2.setText("脑实质")
            self.t3.setText("脑室内")
            self.t4.setText("膜下腔")
            self.t5.setText("硬脑膜下")

    def gradcam_analysis_btn_clicked(self):
        if self.models.currentIndex() != 0:
            self.console.append('[ERROR] Only Dense-Net 201 Model is supported for GradCAM result diagnoses.')
            return
        if not Models[3].loaded:
            Models[3].load_state_dict(torch.load(DENSENET_GRADCAM_PATH))
            Models[3].loaded = True
        image = deepcopy(self.image)
        if len(image.shape) == 3:
            self.console.append('[WARNING] For Heatmap, please use GradCAM Diagnose for optimal result')
            image = rgb_to_grey(image)
        results, flags = get_results(Models[3](gc(image, model=Models[2]).expand(1, -1, -1, -1)))
        labels = ['Any']
        data = [results['Any']]

        try:
            self.resultlayer.removeWidget(self.result_figure)
        except:
            self.resultlayer.removeWidget(self.resultview)

        self.result_figure = ImageView(width=3, height=2, dpi=80)
        self.result_figure.ax.invert_yaxis()
        self.result_figure.ax.xaxis.set_visible(False)
        self.result_figure.ax.set_xlim(0, 225)
        self.result_figure.ax.barh(labels, data, left=0, height=0.5)
        self.result_figure.ax.set_yticklabels([])
        self.result_figure.ax.set_title("GradCAM Results Reliability")
        for i, (key, item) in enumerate(results.items()):
            self.result_figure.ax.annotate(
                f"{item}% -- {key}",
                xy=(item, i + 0.5 / 2),
                xytext=(3, 3),  # 3 points vertical offset
                textcoords="offset points",
                color=flags[i],
                ha="left",
                va="center",
            )

        self.resultlayer.addWidget(self.result_figure)
        self.console.append(f"[INFO] Complete.")
        self.t1_t5()

    # 程序model分析模块
    def start_prediction_btn_clicked(self):
        self.start_btn_hit = True
        plt.close('all')
        if self.image is None:
            self.console.append('[INFO] No image selected')
            return

        # self.console.append("[INFO] Process starts")
        try:
            self.resultlayer.removeWidget(self.result_figure)
        except:
            self.resultlayer.removeWidget(self.resultview)
        # filepath = self.filepath
        # image = None
        # # try:
        # if filepath[-3:] == "dcm":
        #     # 读取dicom文件
        #     self.console.append(f"[INFO] Preprocessing {filepath.split(path_sign)[-1]} Image...")
        #     image = preprocess(filepath)
        #     self.id.setText(str(image.shape))
        # else:
        #     image = plt.imread(filepath)
        # except:
        #     self.console.append('ERROR Encountered while processing')
        #     return

        # self.
        image = deepcopy(self.image)
        if len(image.shape) == 3:
            self.console.append('[WARNING] For Heatmap, please use GradCAM Diagnose for optimal result')
            image = rgb_to_grey(image)

        current_model_index = self.models.currentIndex()

        selected_model = Models[current_model_index]
        if not selected_model.loaded:
            selected_model.load_state_dict(torch.load(MODEL_PATH[current_model_index]))
            selected_model.loaded = True
            selected_model.eval()

        results, self.flag = get_results(selected_model(torch.tensor(image).expand(1, 3, -1, -1)))

        self.result_figure = ImageView(width=3, height=2, dpi=80)
        labels = list(results.keys())
        data = list(results.values())

        self.result_figure.ax.invert_yaxis()
        self.result_figure.ax.xaxis.set_visible(False)
        self.result_figure.ax.set_xlim(0, 225)
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
                color=self.flag[i],
                ha="left",
                va="center",
            )
            i += 1

        self.resultlayer.addWidget(self.result_figure)
        self.console.append(f"[INFO] Complete.")
        self.t1_t5()

    def show_gradcam(self, signal):
        if not self.start_btn_hit:
            self.console.append('[ERROR] Please Diagnose the image first')
            return

        plt.close('all')

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
            selected_model.loaded = True
            selected_model.eval()
        image = self.image
        if isinstance(image, np.ndarray):
            image = torch.tensor(image)

        if len(image.shape) == 3:
            self.console.append('[WARNING] For Heatmap, please use GradCAM Diagnose for optimal result')
            image = rgb_to_grey(image)

        image = (image - 0.188) / 0.315
        image = image.expand(1, 3, -1, -1)
        ind = torch.tensor([[signal]])

        grad_cam = GradCAM(model=selected_model)
        cam = grad_cam(image.expand(1, 3, -1, -1), ind)
        cam = jet(cam)

        image = torch.clamp(image * ct_std + ct_mean, 0, 1) + cam
        image = np.moveaxis(image[0].cpu().numpy(), 0, 2)
        image = image / image.max()
        image = np.around(image * 255).astype(np.uint8)
        # if image.shape[0] == 3:
        #     image = image[0]
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
            "Subdural"
        ]
        if self.flag[signal] == 'green':
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
