import sys
import os
import yaml
import torch
import torchvision
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QComboBox, QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from DDPM.src.model.unet import Unet
from DDPM.src.noise_scheduler.noise_scheduler import NoiseScheduler

# 设置中文字体支持
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


class DDPMGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DDPM数字图像生成器")
        self.setGeometry(100, 100, 800, 600)

        # 模型和配置相关变量
        self.config = None
        self.model = None
        self.scheduler = None
        self.generated_image = None
        self.class_label = 0

        # 加载配置
        self.load_config()

        # 初始化UI
        self.init_ui()

        # 加载模型
        self.load_model()

    def load_config(self):
        config_path = "../config/default.yaml"
        try:
            with open(config_path) as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载配置文件: {str(e)}")
            sys.exit(1)

    def load_model(self):
        if self.config is None:
            return

        try:
            model_config = self.config['model_config']
            diffusion_config = self.config['diffusion_config']
            train_config = self.config['train_config']

            # 检查权重文件是否存在
            ckpt_dir = os.path.join(train_config['task_name'], train_config['ckpt_name'])
            if not os.path.exists(ckpt_dir):
                raise FileNotFoundError(f"权重文件不存在: {ckpt_dir}")

            # 加载模型
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = Unet(model_config).to(device)
            self.model.load_state_dict(torch.load(ckpt_dir, map_location=device))
            self.model.eval()

            # 初始化噪声调度器
            self.scheduler = NoiseScheduler(diffusion_config)

            # 保存配置
            self.model_config = model_config
            self.diffusion_config = diffusion_config
            self.train_config = train_config
            self.device = device

            QMessageBox.information(self, "成功", "模型加载成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            sys.exit(1)

    def init_ui(self):
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建顶部控制区域
        control_layout = QHBoxLayout()

        # 数字选择
        self.label = QLabel("选择数字 (0-9):")
        self.class_combo = QComboBox()
        for i in range(10):
            self.class_combo.addItem(str(i))
        self.class_combo.currentIndexChanged.connect(self.on_class_changed)

        # 生成按钮
        self.generate_btn = QPushButton("生成图片")
        self.generate_btn.clicked.connect(self.generate_image)

        # 重新生成按钮
        self.regenerate_btn = QPushButton("重新生成")
        self.regenerate_btn.clicked.connect(self.generate_image)
        self.regenerate_btn.setEnabled(False)

        # 保存按钮
        self.save_btn = QPushButton("保存图片")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)

        # 添加到控制布局
        control_layout.addWidget(self.label)
        control_layout.addWidget(self.class_combo)
        control_layout.addWidget(self.generate_btn)
        control_layout.addWidget(self.regenerate_btn)
        control_layout.addWidget(self.save_btn)

        # 图片显示区域
        self.image_label = QLabel("请点击'生成图片'按钮")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(400)

        # 添加到主布局
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.image_label)

    def on_class_changed(self):
        self.class_label = int(self.class_combo.currentText())

    def generate_image(self):
        if self.model is None or self.scheduler is None:
            QMessageBox.warning(self, "警告", "模型未加载，请先加载模型")
            return

        try:
            # 禁用按钮
            self.generate_btn.setEnabled(False)
            self.regenerate_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.image_label.setText("正在生成图片，请稍候...")

            # 处理类别标签
            class_label = self.class_label
            if not (0 <= class_label <= 9):
                raise ValueError("类别标签必须是0-9之间的整数")

            # 创建批次的类别标签
            class_labels = torch.tensor([class_label] * self.train_config['num_samples'], device=self.device)

            # 生成样本
            with torch.no_grad():
                xt = torch.randn((self.train_config['num_samples'], self.model_config['img_channels'],
                                  self.model_config['img_size'], self.model_config['img_size'])).to(self.device)

                for i in range(self.diffusion_config['num_timesteps'] - 1, -1, -1):
                    noise_pred = self.model(xt, torch.as_tensor(i).unsqueeze(0).to(self.device), class_labels)
                    xt, x0 = self.scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(self.device))

                # 处理生成的图像
                imgs = torch.clamp(xt, -1., 1.).detach().cpu()
                imgs = (imgs + 1) / 2

                # 对于单通道图像，确保make_grid不会将其转换为3通道
                if self.model_config['img_channels'] == 1:
                    # 保持单通道格式，使用与ddpm_sample.py相同的参数
                    grid_xt = torchvision.utils.make_grid(imgs, nrow=10)
                    # 确保是单通道
                    if grid_xt.shape[0] > 1:
                        # 如果make_grid创建了多通道，取第一个通道
                        grid_xt = grid_xt[0:1, :, :]
                else:
                    grid_xt = torchvision.utils.make_grid(imgs, nrow=10)

                # 保存生成的图像tensor
                self.generated_image = grid_xt

                # 转换为PIL图像，确保与保存的图像使用相同的处理流程
                grid_img = torchvision.transforms.ToPILImage()(grid_xt)

                # 再转换为NumPy数组用于Qt显示
                img_np = np.array(grid_img)
                height, width = img_np.shape[:2]
                channel = 1 if len(img_np.shape) == 2 else img_np.shape[2]

                # 根据通道数选择合适的格式
                if channel == 1:
                    # 单通道图像，转换为灰度图
                    bytes_per_line = width
                    q_img = QImage(img_np.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                else:
                    # 多通道图像，使用RGB格式
                    bytes_per_line = 3 * width
                    q_img = QImage(img_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)

                # 显示图像
                self.image_label.setPixmap(
                    pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

                # 启用按钮
                self.generate_btn.setEnabled(True)
                self.regenerate_btn.setEnabled(True)
                self.save_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成图片失败: {str(e)}")
            self.generate_btn.setEnabled(True)

    def save_image(self):
        if self.generated_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的图片，请先生成图片")
            return

        try:
            # 获取保存路径
            file_path, _ = QFileDialog.getSaveFileName(self, "保存图片",
                                                       os.path.join(os.getcwd(), "generated_image.png"),
                                                       "PNG图片 (*.png);;所有文件 (*)")
            if not file_path:
                return

            # 保存图片
            grid_img = torchvision.transforms.ToPILImage()(self.generated_image)
            grid_img.save(file_path)
            grid_img.close()

            QMessageBox.information(self, "成功", f"图片已保存到: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存图片失败: {str(e)}")

    def resizeEvent(self, event):
        # 当窗口大小改变时，重新缩放图片
        if hasattr(self, 'generated_image') and self.generated_image is not None:
            # 转换为PIL图像，确保与保存的图像使用相同的处理流程
            grid_img = torchvision.transforms.ToPILImage()(self.generated_image)
            
            # 再转换为NumPy数组用于Qt显示
            img_np = np.array(grid_img)
            height, width = img_np.shape[:2]
            channel = 1 if len(img_np.shape) == 2 else img_np.shape[2]
            
            # 根据通道数选择合适的格式
            if channel == 1:
                # 单通道图像，转换为灰度图
                bytes_per_line = width
                q_img = QImage(img_np.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                # 多通道图像，使用RGB格式
                bytes_per_line = 3 * width
                q_img = QImage(img_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        super().resizeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置应用程序样式
    app.setStyle('Fusion')
    # 创建并显示窗口
    window = DDPMGui()
    window.show()
    # 运行应用程序
    app.exec_()
