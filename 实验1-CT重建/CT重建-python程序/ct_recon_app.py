#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT图像重建系统主程序
功能概述：
    1. 生成标准Shepp-Logan幻影图像（CT重建算法测试基准图像）
    2. 模拟CT扫描过程，生成投影数据（正弦图）
    3. 支持4种CT重建算法：直接反投影/傅里叶重建/反投影滤波/滤波反投影
    4. 可视化展示原始图像、投影数据、重建结果
技术栈：
    - tkinter: 图形界面开发
    - numpy: 数值计算与矩阵操作
    - matplotlib: 数据可视化
    - scipy.ndimage: 图像旋转等形态学操作
    - PIL: 图像文件读写与格式转换
依赖模块：
    - ct_rec_algorithms下的4个重建算法实现文件
运行环境：
    Python 3.7+，需安装依赖：numpy, matplotlib, scipy, pillow
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os
from scipy.ndimage import rotate

# 导入CT重建核心算法模块
from ct_rec_algorithms.direct_backprojection import direct_backprojection
from ct_rec_algorithms.fourier_backprojection import fourier_backprojection
from ct_rec_algorithms.backprojection_filter import backprojection_filter
from ct_rec_algorithms.filtered_backprojection import filtered_backprojection


# ---------------------- 1. Shepp-Logan幻影生成模块 ----------------------
def shepp_logan_phantom(size=256):
    """
    生成标准Shepp-Logan幻影图像（CT重建算法的经典测试用例）
    完全对齐generate_shepp_logan_phantom.py的标准实现，保证物理参数一致性
    
    参数：
        size (int): 输出幻影图像的尺寸（正方形），默认256x256
    返回：
        np.ndarray: float32类型的幻影图像数组，shape=(size, size)
                    数值保留原始物理意义（未归一化到0-255），便于后续非线性增强
    椭圆参数说明（每一项对应一个解剖结构）：
        [x0, y0, a, b, angle_deg, rho]
        - x0/y0: 椭圆中心坐标（归一化到[-1,1]）
        - a/b: 椭圆长/短半轴长度
        - angle_deg: 椭圆旋转角度（度）
        - rho: 该区域的灰度值（CT值模拟）
    """
    # 标准Shepp-Logan幻影的椭圆参数（对应颅脑解剖结构）
    ellipses = [
        [0.0, 0.0, 0.92, 0.69, 90, 2.0],           # 主椭圆（大脑主体）
        [0.0, -0.0184, 0.874, 0.6624, 90, -0.98],  # 颅骨外层
        [0.22, 0.0, 0.31, 0.11, 72, -0.02],        # 左脑室
        [-0.22, 0.0, 0.41, 0.16, 108, -0.02],      # 右脑室
        [0.0, 0.35, 0.25, 0.21, 90, 0.01],         # 脑干
        [0.0, 0.1, 0.046, 0.046, 0, 0.01],         # 小脑区域小椭圆
        [0.0, -0.1, 0.046, 0.046, 0, 0.01],        # 小脑区域小椭圆
        [-0.08, -0.605, 0.046, 0.023, 0, 0.01],    # 左眼结构
        [0.0, -0.605, 0.023, 0.023, 0, 0.01],      # 眼部中心结构
        [0.06, -0.605, 0.046, 0.023, 90, 0.01]     # 右眼结构
    ]

    # 创建归一化坐标网格 [-1, 1]，保证与标准实现的坐标体系一致
    y, x = np.ogrid[-1:1:size * 1j, -1:1:size * 1j]

    # 初始化幻影图像数组
    phantom = np.zeros((size, size), dtype=np.float32)

    # 逐椭圆绘制解剖结构
    for item in ellipses:
        x0, y0, a, b, angle_deg, gray_val = item
        angle = np.deg2rad(angle_deg)  # 角度转弧度
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # 坐标平移：将椭圆中心移至目标位置
        x_shift = x - x0
        y_shift = y - y0

        # 坐标逆旋转：抵消椭圆自身旋转，便于后续椭圆方程判断
        xr = cos_a * x_shift + sin_a * y_shift
        yr = -sin_a * x_shift + cos_a * y_shift

        # 椭圆方程判断：筛选椭圆内部像素
        ellipse_mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0

        # 为椭圆区域赋值灰度值（叠加模式，模拟不同组织的CT值）
        phantom[ellipse_mask] += gray_val

    # 保留原始物理数值，不做0-255归一化（后续通过Sigmoid增强对比度）
    return phantom


def generate_sinogram(phantom, angles=180):
    """
    从Shepp-Logan幻影图像生成正弦图（模拟CT扫描的投影数据）
    核心逻辑：对幻影图像沿不同角度做X轴积分，模拟X射线穿透人体后的投影
    
    参数：
        phantom (np.ndarray): Shepp-Logan幻影图像，shape=(size, size)
        angles (int): 投影角度数量，默认180个角度（0~180度）
    返回：
        np.ndarray: 正弦图数组，shape=(size, angles)，float32类型
    """
    size = phantom.shape[0]
    theta = np.linspace(0, 180, angles, endpoint=False)  # 生成投影角度序列
    sinogram = np.zeros((size, angles), dtype=np.float32)

    # 创建投影坐标网格（归一化到[-1,1]）
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # 逐角度生成投影数据
    for i, angle in enumerate(theta):
        # 计算当前角度的旋转矩阵
        angle_rad = np.deg2rad(angle)
        X_rot = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
        Y_rot = -X * np.sin(angle_rad) + Y * np.cos(angle_rad)

        # 沿X轴积分（模拟X射线投影）
        for j in range(size):
            # 筛选有效投影区域（坐标在[-1,1]范围内）
            mask = (Y_rot[:, j] >= -1) & (Y_rot[:, j] <= 1)
            if np.any(mask):
                # 积分计算：对有效区域的像素值求和
                sinogram[j, i] = np.sum(phantom[mask, j])

    return sinogram


# ---------------------- 2. 重建算法配置模块 ----------------------
# 重建算法映射字典：键为界面显示名称，值为对应算法实现函数
# 便于界面选择与算法函数的解耦调用
RECONSTRUCTION_ALGORITHMS = {
    "直接反投影重建": direct_backprojection,
    "傅里叶重建": fourier_backprojection,
    "反投影滤波重建": backprojection_filter,
    "滤波反投影重建": filtered_backprojection
}


# ---------------------- 3. CT扫描模拟模块 ----------------------
def simulate_ct_scan(image, num_angles=180, callback=None):
    """
    对输入图像模拟CT扫描过程，生成物理意义正确的正弦图（投影数据）
    核心流程：图像旋转 → 垂直方向积分 → 投影数据归一化
    
    参数：
        image (np.ndarray): 输入图像数组，shape=(H, W)，支持非正方形图像
        num_angles (int): 投影角度数量，默认180（0~180度）
        callback (function): 扫描过程回调函数，用于实时更新动画
                             回调参数：(current_index, current_angle, total_angles, current_sinogram)
    返回：
        tuple: (sinogram, angles)
            - sinogram: 投影数据数组，shape=(num_angles, size)，float32类型
            - angles: 投影角度数组（弧度），shape=(num_angles,)
    处理逻辑：
        1. 统一图像为正方形（CT扫描探测器阵列通常为正方形）
        2. 逐角度旋转图像，模拟X射线不同方向入射
        3. 对旋转后的图像逐列求和，得到投影值
        4. 归一化投影数据，保证数值范围在0~255（适配显示与算法要求）
    """
    # 步骤1：统一图像尺寸为正方形（补零填充）
    size = max(image.shape)
    if image.shape[0] != size or image.shape[1] != size:
        pad_h = (size - image.shape[0]) // 2
        pad_w = (size - image.shape[1]) // 2
        # 对称补零，保证图像中心不变
        image = np.pad(image, ((pad_h, size - image.shape[0] - pad_h),
                               (pad_w, size - image.shape[1] - pad_w)),
                       mode='constant', constant_values=0)

    # 步骤2：生成投影角度（0~π弧度，对应0~180度，CT扫描常规角度范围）
    angles = np.linspace(0, np.pi, num_angles, endpoint=False, dtype=np.float32)
    sinogram = np.zeros((num_angles, size), dtype=np.float32)  # 初始化正弦图

    # 步骤3：生成探测器坐标（模拟CT探测器阵列的物理位置）
    detector_coords = np.linspace(-size / 2, size / 2, size)  # 像素单位物理坐标

    # 步骤4：逐角度生成投影数据
    for i, angle in enumerate(angles):
        # 旋转图像：模拟X射线沿当前角度入射
        rotated_img = rotate(image, angle * 180 / np.pi, reshape=False, mode='constant')

        # 垂直积分：对旋转后的图像逐列求和（沿Y轴），得到该角度的投影值
        projection = np.sum(rotated_img, axis=0)  # shape=(size,)

        # 赋值到正弦图对应角度位置
        sinogram[i] = projection

        # 执行回调（用于扫描过程动画更新）
        if callback:
            callback(i, angle, num_angles, sinogram)

    # 步骤5：投影数据归一化（适配Shepp-Logan数据特性）
    if np.max(sinogram) > 255.0:
        # 数据范围超过255时，按尺寸缩放至0~255
        sinogram = sinogram / size * 255
    else:
        # 数据范围在255内时，直接裁剪异常值（避免负数或超界值）
        sinogram = np.clip(sinogram, 0, 255)

    return sinogram, angles


# ---------------------- 4. 主程序界面类 ----------------------
class CTReconstructionApp:
    """
    CT图像重建系统主界面类
    负责界面布局、用户交互、数据流转、结果可视化
    """
    def __init__(self, root):
        """
        初始化界面
        
        参数：
            root (tk.Tk): tkinter根窗口对象
        """
        self.root = root
        self.root.title("CT图像重建系统")
        self.root.geometry("2500x1200")  # 初始窗口尺寸（适配多屏显示）
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)  # 窗口关闭回调

        # 字体配置（统一界面字体风格，适配中文显示）
        self.font_size_base = 20  # 基础字体大小（按钮、标签）
        self.font_size_large = 22  # 大号字体大小（标题、下拉框）
        self.font_family = "SimHei"  # 中文显示字体（黑体）

        # 封装字体样式（便于全局修改）
        self.font_base = (self.font_family, self.font_size_base)
        self.font_large = (self.font_family, self.font_size_large)

        # 数据状态标记（核心业务状态变量）
        self.data_type = None  # 数据类型：'image'/'sinogram'/'shepp_logan_image'/'shepp_logan_sinogram'
        self.sinogram_data = None  # 存储投影数据（正弦图）
        self.angles_data = None  # 存储投影角度（弧度）
        self.raw_data = None  # 存储原始图像/数据
        self.recon_result = None  # 存储重建结果
        self.data_source = None  # 标记数据来源（上传/生成）

        # 初始化界面样式与组件
        self._setup_style()  # 配置ttk控件样式
        self.root.option_add("*Font", self.font_base)  # 设置tk原生控件默认字体
        self._create_widgets()  # 创建界面组件

    def _create_widgets(self):
        """
        创建所有界面组件
        分区域构建：控制面板、图像显示区（原始/投影/重建）
        """
        # 1. 顶部控制面板（用户操作区）
        control_frame = ttk.LabelFrame(self.root, text="控制面板")
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # 第一行：文件选择 + Shepp-Logan生成
        row1_frame = ttk.Frame(control_frame)
        row1_frame.pack(fill=tk.X, padx=5, pady=5)

        # 文件选择按钮与路径显示
        self.file_path_var = tk.StringVar(value="未选择文件")
        ttk.Button(row1_frame, text="选择原始图像/投影数据",
                   command=self._select_file).pack(side=tk.LEFT, padx=5)
        ttk.Label(row1_frame, textvariable=self.file_path_var).pack(side=tk.LEFT, padx=(5,20))

        # Shepp-Logan生成控件组
        ttk.Label(row1_frame, text="生成Shepp-Logan数据：").pack(side=tk.LEFT, padx=(50,5))
        self.sl_size_var = tk.StringVar(value="256")  # 幻影图像尺寸默认值
        ttk.Label(row1_frame, text="图像尺寸：").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row1_frame, textvariable=self.sl_size_var, width=10).pack(side=tk.LEFT)
        ttk.Button(row1_frame, text="生成数据",
                   command=self._generate_shepp_logan).pack(side=tk.LEFT, padx=5)

        # 第二行：算法选择 + 滤波器选择 + 投影角度配置 + 功能按钮
        row2_frame = ttk.Frame(control_frame)
        row2_frame.pack(fill=tk.X, padx=5, pady=5)

        # 重建算法选择下拉框
        ttk.Label(row2_frame, text="选择重建算法：").pack(side=tk.LEFT, padx=5)
        self.algorithm_var = tk.StringVar()
        self.algorithm_list = ["直接反投影重建", "傅里叶重建", "反投影滤波重建", "滤波反投影重建"]
        algorithm_combobox = ttk.Combobox(row2_frame, textvariable=self.algorithm_var,
                                          values=self.algorithm_list, state="readonly")
        algorithm_combobox.pack(side=tk.LEFT, padx=5)
        algorithm_combobox.bind("<<ComboboxSelected>>", self._on_algorithm_change)  # 算法切换回调
        if self.algorithm_list:
            algorithm_combobox.current(0)  # 默认选中第一个算法

        # 滤波器选择下拉框（根据算法动态启用/禁用）
        ttk.Label(row2_frame, text="滤波器：").pack(side=tk.LEFT, padx=(20, 5))
        self.filter_type_var = tk.StringVar()
        self.filter_type_list = ["R-L", "S-L", "cosine", "hamming"]  # 支持的滤波器类型
        self.filter_combobox = ttk.Combobox(row2_frame, textvariable=self.filter_type_var,
                                       values=self.filter_type_list, state="disabled", width=12)
        self.filter_combobox.pack(side=tk.LEFT, padx=5)
        self.filter_combobox.current(0)  # 默认选中第一个滤波器
        self._on_algorithm_change(None)  # 初始化算法-滤波器关联状态

        # 投影角度数配置（仅对图像数据生效）
        ttk.Label(row2_frame, text="投影角度数：").pack(side=tk.LEFT, padx=(20,10))
        self.angle_num_var = tk.StringVar(value="180")  # 默认180个投影角度
        ttk.Entry(row2_frame, textvariable=self.angle_num_var, width=10).pack(side=tk.LEFT)

        # 功能按钮：模拟扫描、重建
        self.simulate_btn = ttk.Button(row2_frame, text="开始模拟扫描",
                                     command=self._run_simulation)
        self.simulate_btn.pack(side=tk.LEFT, padx=20)
        self.recon_btn = ttk.Button(row2_frame, text="开始重建",
                                   command=self._run_reconstruction)
        self.recon_btn.pack(side=tk.LEFT, padx=5)

        # 2. 图像显示区域（原始数据/投影数据/重建结果）
        display_frame = ttk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 原始数据/图像显示区
        raw_frame = ttk.LabelFrame(display_frame, text="原始数据/图像")
        raw_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 投影数据（正弦图）显示区
        sinogram_frame = ttk.LabelFrame(display_frame, text="CT投影数据（正弦图）")
        sinogram_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 重建结果显示区
        recon_frame = ttk.LabelFrame(display_frame, text="重建结果")
        recon_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建matplotlib绘图画布（原始图像）
        self.raw_fig, self.raw_ax = plt.subplots(figsize=(5, 5), dpi=100)
        self.raw_ax.set_title("未加载数据", fontsize=self.font_size_large, fontfamily=self.font_family)
        self.raw_canvas = FigureCanvasTkAgg(self.raw_fig, master=raw_frame)
        self.raw_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 创建matplotlib绘图画布（投影数据）
        self.sinogram_fig, self.sinogram_ax = plt.subplots(figsize=(5, 5), dpi=100)
        self.sinogram_ax.set_title("未生成投影数据", fontsize=self.font_size_large, fontfamily=self.font_family)
        self.sinogram_canvas = FigureCanvasTkAgg(self.sinogram_fig, master=sinogram_frame)
        self.sinogram_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 创建matplotlib绘图画布（重建结果）
        self.recon_fig, self.recon_ax = plt.subplots(figsize=(5, 5), dpi=100)
        self.recon_ax.set_title("未进行重建", fontsize=self.font_size_large, fontfamily=self.font_family)
        self.recon_canvas = FigureCanvasTkAgg(self.recon_fig, master=recon_frame)
        self.recon_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_style(self):
        """
        配置ttk控件的全局样式
        统一字体、内边距等样式属性，提升界面一致性
        """
        style = ttk.Style(self.root)

        # 基础控件样式
        style.configure("TLabel", font=self.font_base)
        style.configure("TButton", font=self.font_base)
        style.configure("TCombobox", font=self.font_large)
        style.configure("TRadiobutton", font=self.font_base)
        style.configure("TEntry", font=self.font_base)

        # 带标题框架样式
        style.configure(
            "TLabelframe.Label",
            font=self.font_large,
            padding=5  # 标题内边距，避免拥挤
        )

    def _on_closing(self):
        """
        窗口关闭时的资源清理操作
        确保Matplotlib资源释放、进程正常退出
        """
        plt.close('all')  # 关闭所有Matplotlib绘图窗口
        self.root.destroy()  # 销毁tkinter主窗口
        import sys
        sys.exit()  # 强制退出Python进程（避免资源泄漏）

    def _on_algorithm_change(self, event):
        """
        重建算法选择变更时的回调函数
        核心逻辑：根据选中算法动态控制滤波器下拉框的状态与可选值
        
        参数：
            event: tkinter事件对象（下拉框选择事件）
        算法-滤波器关联规则：
            - 滤波反投影重建：仅支持R-L、S-L滤波器
            - 反投影滤波重建：支持R-L、S-L、cosine、hamming滤波器
            - 直接反投影/傅里叶重建：禁用滤波器选择
        """
        selected_algo = self.algorithm_var.get()
        
        if selected_algo == "滤波反投影重建":
            # 滤波反投影（FBP）仅支持R-L、S-L滤波器
            self.filter_combobox.config(state="readonly")
            self.filter_combobox['values'] = ["R-L", "S-L"]
            # 重置滤波器选择（若当前选择不在允许列表）
            if self.filter_type_var.get() not in ["R-L", "S-L"]:
                self.filter_combobox.current(0)
                
        elif selected_algo == "反投影滤波重建":
            # 反投影滤波（BPF）支持全部4种滤波器
            self.filter_combobox.config(state="readonly")
            self.filter_combobox['values'] = ["R-L", "S-L", "cosine", "hamming"]
            
        else:
            # 直接反投影/傅里叶重建无需滤波器，禁用下拉框
            self.filter_combobox.config(state="disabled")

    def _generate_shepp_logan(self):
        """
        生成Shepp-Logan幻影图像并执行非线性增强
        流程：参数验证 → 生成原始幻影 → 非线性增强 → 界面状态更新
        非线性增强逻辑：Sigmoid函数拉伸对比度，适配可视化显示
        """
        try:
            # 验证尺寸参数有效性
            size = int(self.sl_size_var.get())
            if size <= 0 or size > 1024:
                raise ValueError("尺寸必须为1-1024之间的整数")

            # 设置等待光标，提示用户正在处理
            self.root.config(cursor="wait")
            self.root.update()

            # 1. 生成原始Shepp-Logan幻影图像（物理值）
            phantom_raw = shepp_logan_phantom(size)
            
            # 2. 非线性增强（Sigmoid函数），提升可视化对比度
            center = 1.0  # 增强中心值（适配Shepp-Logan数据分布）
            gain = 50.0   # 增强增益（控制对比度拉伸程度）
            z = np.clip(gain * (phantom_raw - center), -50, 50)  # 裁剪异常值，避免数值爆炸
            self.raw_data = 255 / (1 + np.exp(-z))  # Sigmoid归一化到0~255
            
            # 3. 更新数据状态标记
            self.data_type = "shepp_logan_image"
            self.data_source = f"生成的Shepp-Logan幻影图像（{size}x{size}，已内置增强）"
            
            # 4. 清空历史数据（避免干扰）
            self.sinogram_data = None
            self.angles_data = None
            self.recon_result = None
            
            # 5. 更新投影/重建画布提示
            self.sinogram_ax.clear()
            self.sinogram_ax.set_title("等待模拟扫描...", fontsize=self.font_size_large, fontfamily=self.font_family)
            self.sinogram_canvas.draw()
            self.recon_ax.clear()
            self.recon_ax.set_title("未进行重建", fontsize=self.font_size_large, fontfamily=self.font_family)
            self.recon_canvas.draw()

            # 恢复正常光标
            self.root.config(cursor="")

            # 6. 更新界面显示
            self.file_path_var.set(self.data_source)
            self._display_raw_data()  # 显示生成的幻影图像
            messagebox.showinfo("成功", f"{self.data_source}生成完成！")

        except ValueError as e:
            # 参数输入错误处理
            messagebox.showerror("错误", f"输入参数无效：{str(e)}")
        except Exception as e:
            # 通用异常处理（确保光标恢复）
            self.root.config(cursor="")
            messagebox.showerror("错误", f"生成数据失败：{str(e)}")

    def _select_file(self):
        """
        选择原始图像/投影数据文件，并解析数据类型
        支持文件类型：
            - 图像文件：png/jpg/jpeg/tif/tiff/bmp（转为灰度图）
            - 数据文件：npy/txt/csv（投影数据）
        处理逻辑：文件路径验证 → 数据加载 → 类型标记 → 界面更新
        """
        file_path = filedialog.askopenfilename(
            title="选择CT原始数据/图像",
            filetypes=[
                ("图像文件", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("投影数据文件", "*.npy *.txt *.csv"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.file_path_var.set(os.path.basename(file_path))
            try:
                # 判断文件类型并加载数据
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    # 加载图像文件（转为灰度图）
                    img = Image.open(file_path).convert('L')
                    self.raw_data = np.array(img, dtype=np.float32)
                    self.data_type = "image"
                    self.data_source = f"上传图像：{os.path.basename(file_path)}"
                    # 清空历史投影数据（需重新模拟扫描）
                    self.sinogram_data = None
                    self.angles_data = None
                    # 更新投影画布提示
                    self.sinogram_ax.clear()
                    self.sinogram_ax.set_title("等待模拟扫描...", fontsize=self.font_size_large, fontfamily=self.font_family)
                    self.sinogram_canvas.draw()
                else:
                    # 加载投影数据文件（npy/txt/csv）
                    if file_path.endswith('.npy'):
                        loaded_data = np.load(file_path, allow_pickle=True)
                        
                        # 补充：此处需根据实际数据格式完善投影数据加载逻辑
                        # （原代码裁剪，需根据实际业务补充）
                        self.sinogram_data = loaded_data
                        self.data_type = "sinogram"
                        self.data_source = f"上传投影数据：{os.path.basename(file_path)}"
                        # 显示投影数据
                        self._display_sinogram_data()
                
                # 显示原始数据/图像
                self._display_raw_data()
                messagebox.showinfo("成功", f"已加载：{self.data_source}")

            except Exception as e:
                messagebox.showerror("错误", f"加载文件失败：{str(e)}")

    def _display_raw_data(self):
        """
        显示原始数据/图像到对应画布
        核心逻辑：清空画布 → 绘制图像 → 更新画布
        补充说明：原代码裁剪，此处为核心逻辑占位，需根据实际业务完善
        """
        if self.raw_data is not None:
            self.raw_ax.clear()
            self.raw_ax.imshow(self.raw_data, cmap='gray', vmin=0, vmax=255)
            self.raw_ax.set_title("原始数据/图像", fontsize=self.font_size_large, fontfamily=self.font_family)
            self.raw_ax.axis('off')  # 隐藏坐标轴
            self.raw_canvas.draw()

    def _display_sinogram_data(self):
        """
        显示投影数据（正弦图）到对应画布
        补充说明：原代码裁剪，此处为核心逻辑占位，需根据实际业务完善
        """
        if self.sinogram_data is not None:
            self.sinogram_ax.clear()
            self.sinogram_ax.imshow(self.sinogram_data, cmap='gray', aspect='auto')
            self.sinogram_ax.set_title("CT投影数据（正弦图）", fontsize=self.font_size_large, fontfamily=self.font_family)
            self.sinogram_ax.set_xlabel("投影角度", fontsize=self.font_size_base, fontfamily=self.font_family)
            self.sinogram_ax.set_ylabel("探测器通道", fontsize=self.font_size_base, fontfamily=self.font_family)
            self.sinogram_canvas.draw()

    def _run_simulation(self):
        """
        执行CT扫描模拟，生成投影数据
        流程：参数验证 → 调用模拟函数 → 存储投影数据 → 界面更新
        补充说明：原代码裁剪，此处为核心逻辑占位，需根据实际业务完善
        """
        try:
            # 验证投影角度数参数
            num_angles = int(self.angle_num_var.get())
            if num_angles <= 0 or num_angles > 360:
                raise ValueError("投影角度数必须为1-360之间的整数")

            if self.raw_data is None:
                messagebox.showwarning("警告", "请先加载/生成原始图像数据")
                return

            # 设置等待光标
            self.root.config(cursor="wait")
            self.root.update()

            # 调用扫描模拟函数
            sinogram, angles = simulate_ct_scan(self.raw_data, num_angles=num_angles)
            self.sinogram_data = sinogram
            self.angles_data = angles

            # 恢复正常光标
            self.root.config(cursor="")

            # 显示投影数据
            self._display_sinogram_data()
            messagebox.showinfo("成功", f"模拟CT扫描完成，生成{num_angles}个角度的投影数据")

        except ValueError as e:
            messagebox.showerror("错误", f"参数无效：{str(e)}")
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("错误", f"模拟扫描失败：{str(e)}")

    def _run_reconstruction(self):
        """
        执行CT图像重建
        流程：数据验证 → 选择算法 → 调用重建函数 → 显示结果
        补充说明：原代码裁剪，此处为核心逻辑占位，需根据实际业务完善
        """
        try:
            # 验证前置条件
            if self.sinogram_data is None or self.angles_data is None:
                messagebox.showwarning("警告", "请先完成CT扫描模拟，生成投影数据")
                return

            selected_algo = self.algorithm_var.get()
            if not selected_algo:
                messagebox.showwarning("警告", "请选择重建算法")
                return

            # 设置等待光标
            self.root.config(cursor="wait")
            self.root.update()

            # 获取重建算法函数
            recon_func = RECONSTRUCTION_ALGORITHMS.get(selected_algo)
            if not recon_func:
                raise ValueError(f"不支持的重建算法：{selected_algo}")

            # 调用重建算法（根据算法类型传递参数）
            if selected_algo in ["滤波反投影重建", "反投影滤波重建"]:
                # 带滤波器的算法
                filter_type = self.filter_type_var.get()
                self.recon_result = recon_func(self.sinogram_data, self.angles_data, filter_type=filter_type)
            else:
                # 无滤波器的算法
                self.recon_result = recon_func(self.sinogram_data, self.angles_data)

            # 恢复正常光标
            self.root.config(cursor="")

            # 显示重建结果
            self._display_recon_result()
            messagebox.showinfo("成功", f"{selected_algo}完成！")

        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("错误", f"重建失败：{str(e)}")

    def _display_recon_result(self):
        """
        显示重建结果到对应画布
        补充说明：原代码裁剪，此处为核心逻辑占位，需根据实际业务完善
        """
        if self.recon_result is not None:
            self.recon_ax.clear()
            self.recon_ax.imshow(self.recon_result, cmap='gray', vmin=0, vmax=255)
            self.recon_ax.set_title("重建结果", fontsize=self.font_size_large, fontfamily=self.font_family)
            self.recon_ax.axis('off')
            self.recon_canvas.draw()


# ---------------------- 程序入口 ----------------------
if __name__ == "__main__":
    """
    程序主入口
    创建根窗口，初始化应用，启动主循环
    """
    root = tk.Tk()
    app = CTReconstructionApp(root)
    root.mainloop()