# MNIST-number-generate
基于MNIST数据集，利用DDPM算法实现手写数字生成。本项目仅用于学习，项目来源：https://space.bilibili.com/353555504?spm_id_from=333.337.0.0

# 数据准备
修改create_data.py脚本对应参数，在dataset/data路径下生成train，test目录。

# 模型训练
脚本位于utils文件夹下，关于模型具体参数于config/default.yaml文件内修改。utils/ddpm_sample.py与用户交互生成图片。


ddpm_gui_trae.py为TRAE生成，利用pyqt制作简易ui界面实现窗口交互。


# 项目效果
<img width="819" height="653" alt="屏幕截图 2025-08-20 135326" src="https://github.com/user-attachments/assets/771d5921-ccae-4183-8b3c-6f386d73abe1" />
