import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os

from ct_rec_algorithms.direct_backprojection import run_direct_reconstruction
from ct_rec_algorithms.fourier_backprojection import fourier_backprojection
from ct_rec_algorithms.backprojection_filter import backprojection_filter
from ct_rec_algorithms.filtered_backprojection import filtered_backprojection


# ---------------------- 1. 新增：Shepp-Logan 生成模块（对齐标准实现） ----------------------
def shepp_logan_phantom(size=256):
    """生成Shepp-Logan幻影图像（完全对齐generate_shepp_logan_phantom.py标准）"""
    # 椭圆参数: [x0, y0, a, b, angle_deg, rho]（与标准文件完全一致）
    ellipses = [
        [0.0, 0.0, 0.92, 0.69, 90, 2.0],  # 主椭圆（头部）
        [0.0, -0.0184, 0.874, 0.6624, 90, -0.98],  # 颅骨
        [0.22, 0.0, 0.31, 0.11, 72, -0.02],  # 左脑室
        [-0.22, 0.0, 0.41, 0.16, 108, -0.02],  # 右脑室
        [0.0, 0.35, 0.25, 0.21, 90, 0.01],  # 脑干
        [0.0, 0.1, 0.046, 0.046, 0, 0.01],  # 小椭圆
        [0.0, -0.1, 0.046, 0.046, 0, 0.01],  # 小椭圆
        [-0.08, -0.605, 0.046, 0.023, 0, 0.01],  # 眼球
        [0.0, -0.605, 0.023, 0.023, 0, 0.01],  # 眼球
        [0.06, -0.605, 0.046, 0.023, 90, 0.01]  # 眼球
    ]

    # 创建归一化坐标网格 [-1, 1]（与标准文件完全一致）
    y, x = np.ogrid[-1:1:size * 1j, -1:1:size * 1j]

    phantom = np.zeros((size, size), dtype=np.float32)

    for item in ellipses:
        x0, y0, a, b, angle_deg, gray_val = item
        angle = np.deg2rad(angle_deg)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # 平移（与标准文件完全一致）
        x_shift = x - x0
        y_shift = y - y0

        # 逆旋转（与标准文件完全一致）
        xr = cos_a * x_shift + sin_a * y_shift
        yr = -sin_a * x_shift + cos_a * y_shift

        # 椭圆方程（与标准文件完全一致）
        ellipse_mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0

        # 设置灰度值（与标准文件完全一致）
        phantom[ellipse_mask] += gray_val

    return phantom


def generate_sinogram(phantom, angles=180):
    """从Shepp-Logan幻影生成正弦图（投影数据），模拟真实CT原始数据"""
    size = phantom.shape[0]
    theta = np.linspace(0, 180, angles, endpoint=False)  # 投影角度
    sinogram = np.zeros((size, angles), dtype=np.float32)

    # 创建投影坐标
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    for i, angle in enumerate(theta):
        # 旋转坐标
        angle_rad = np.deg2rad(angle)
        X_rot = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
        Y_rot = -X * np.sin(angle_rad) + Y * np.cos(angle_rad)

        # 沿X轴积分（投影）
        for j in range(size):
            mask = (Y_rot[:, j] >= -1) & (Y_rot[:, j] <= 1)
            if np.any(mask):
                sinogram[j, i] = np.sum(phantom[mask, j])

    return sinogram


# ---------------------- 2. 算法模块 ----------------------
# 算法字典：更新为四种指定算法（键为显示名称，值为对应函数）
RECONSTRUCTION_ALGORITHMS = {
    "直接反投影重建": run_direct_reconstruction,
    "傅里叶反投影重建": fourier_backprojection,
    "反投影滤波重建": backprojection_filter,
    "滤波反投影重建": filtered_backprojection
}


# ---------------------- 3. 模拟CT扫描模块 ----------------------
def simulate_ct_scan(image, num_angles=180):
    """
    对输入图像模拟CT扫描，生成投影数据（正弦图）和投影角度
    :param image: 输入图像，numpy数组，shape=(H, W)
    :param num_angles: 投影角度数量（默认180）
    :return: sinogram（投影数据）、angles（投影角度，弧度）
    """
    # 统一图像尺寸为正方形（CT扫描常规处理）
    size = max(image.shape)
    if image.shape[0] != size or image.shape[1] != size:
        # 补零到正方形
        pad_h = (size - image.shape[0]) // 2
        pad_w = (size - image.shape[1]) // 2
        image = np.pad(image, ((pad_h, size - image.shape[0] - pad_h),
                               (pad_w, size - image.shape[1] - pad_w)),
                       mode='constant', constant_values=0)

    # 生成投影角度（0~π弧度，对应0~180度）
    angles = np.linspace(0, np.pi, num_angles, endpoint=False, dtype=np.float32)
    sinogram = np.zeros((size, num_angles), dtype=np.float32)

    # 创建归一化坐标网格
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # 逐角度计算投影（沿旋转后的X轴积分）
    for i, angle in enumerate(angles):
        # 旋转坐标
        angle_rad = angle
        X_rot = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
        Y_rot = -X * np.sin(angle_rad) + Y * np.cos(angle_rad)

        # 沿X轴积分（投影）
        for j in range(size):
            mask = (Y_rot[:, j] >= -1) & (Y_rot[:, j] <= 1)
            if np.any(mask):
                sinogram[j, i] = np.sum(image[mask, j])

    # 转置为(角度数, 探测器数)（匹配算法输入要求）
    sinogram = sinogram.T
    return sinogram, angles

# ---------------------- 4. 主程序界面类（关键修改：对齐显示标准） ----------------------
class CTReconstructionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CT图像重建系统")
        self.root.geometry("2500x1200")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.font_size_base = 20  # 基础字体（按钮、标签），改这个数字调整大小
        self.font_size_large = 22  # 大号字体（标题、下拉框），改这个数字调整大小
        self.font_family = "SimHei"  # 字体族（已简化为你能正常显示的SimHei）

        # 定义字体配置（同时支持ttk和tk原生控件）
        self.font_base = (self.font_family, self.font_size_base)
        self.font_large = (self.font_family, self.font_size_large)

        # Shepp-Logan显示参数（完全对齐generate_shepp_logan_phantom.py）
        self.sl_display_config = {
            "cmap": 'gray',
            "extent": [-1, 1, -1, 1],
            "vmin": 0.95,
            "vmax": 1.25,
            "origin": 'lower',
            "alpha": 1.0
        }

        # 新增：数据类型标记 + 中间变量（存储投影数据/角度）
        self.data_type = None  # 'image'/'sinogram'/'shepp_logan_image'/'shepp_logan_sinogram'
        self.sinogram_data = None  # 投影数据
        self.angles_data = None  # 投影角度

        self._setup_style()  # 配置ttk控件样式
        # 新增：配置tkinter原生控件的默认字体
        self.root.option_add("*Font", self.font_base)

        # 初始化变量
        self.raw_data = None
        self.recon_result = None
        self.data_source = None  # 新增：标记数据来源（上传/生成）

        # 创建界面组件
        self._create_widgets()

    def _create_widgets(self):
        """创建界面组件（新增投影角度数配置）"""
        # 1. 顶部控制面板
        control_frame = ttk.LabelFrame(self.root, text="控制面板")
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # 第一行：文件选择 + Shepp-Logan生成
        row1_frame = ttk.Frame(control_frame)
        row1_frame.pack(fill=tk.X, padx=5, pady=5)

        # 选择文件按钮
        self.file_path_var = tk.StringVar(value="未选择文件")
        ttk.Button(row1_frame, text="选择原始图像/投影数据",
                   command=self._select_file).pack(side=tk.LEFT, padx=5)
        ttk.Label(row1_frame, textvariable=self.file_path_var).pack(side=tk.LEFT, padx=5)

        # Shepp-Logan生成相关控件
        ttk.Label(row1_frame, text="生成Shepp-Logan数据：").pack(side=tk.LEFT, padx=10)
        self.sl_data_type = tk.StringVar(value="幻影图像")
        ttk.Radiobutton(row1_frame, text="幻影图像", variable=self.sl_data_type,
                        value="幻影图像").pack(side=tk.LEFT)
        ttk.Radiobutton(row1_frame, text="正弦图(投影数据)", variable=self.sl_data_type,
                        value="正弦图").pack(side=tk.LEFT)

        self.sl_size_var = tk.StringVar(value="256")
        ttk.Label(row1_frame, text="图像尺寸：").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row1_frame, textvariable=self.sl_size_var, width=10).pack(side=tk.LEFT)

        ttk.Button(row1_frame, text="生成数据",
                   command=self._generate_shepp_logan).pack(side=tk.LEFT, padx=5)

        # 第二行：算法选择 + 投影角度数配置 + 重建按钮
        row2_frame = ttk.Frame(control_frame)
        row2_frame.pack(fill=tk.X, padx=5, pady=5)

        # 算法选择下拉框（仅保留direct_reconstruction，可扩展其他算法）
        ttk.Label(row2_frame, text="选择重建算法：").pack(side=tk.LEFT, padx=5)
        self.algorithm_var = tk.StringVar()
        self.algorithm_list = ["直接反投影重建","傅里叶反投影重建","反投影滤波重建","滤波反投影重建"]  # 可扩展：["直接反投影重建",...]
        algorithm_combobox = ttk.Combobox(row2_frame, textvariable=self.algorithm_var,
                                          values=self.algorithm_list, state="readonly")
        algorithm_combobox.pack(side=tk.LEFT, padx=5)
        if self.algorithm_list:
            algorithm_combobox.current(0)

        # 新增：投影角度数配置（仅对图像数据生效）
        ttk.Label(row2_frame, text="投影角度数：").pack(side=tk.LEFT, padx=10)
        self.angle_num_var = tk.StringVar(value="180")
        ttk.Entry(row2_frame, textvariable=self.angle_num_var, width=10).pack(side=tk.LEFT)

        # 重建按钮
        ttk.Button(row2_frame, text="开始重建",
                   command=self._run_reconstruction).pack(side=tk.LEFT, padx=20)

        # 2. 图像显示区域（新增投影数据显示区）
        display_frame = ttk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 原始数据/图像显示区
        raw_frame = ttk.LabelFrame(display_frame, text="原始数据/图像")
        raw_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 投影数据显示区（新增）
        sinogram_frame = ttk.LabelFrame(display_frame, text="CT投影数据（正弦图）")
        sinogram_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 重建结果显示区
        recon_frame = ttk.LabelFrame(display_frame, text="重建结果")
        recon_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建matplotlib绘图区域
        # 原始图像画布
        self.raw_fig, self.raw_ax = plt.subplots(figsize=(5, 5), dpi=100)
        self.raw_ax.set_title("未加载数据", fontsize=self.font_size_large, fontfamily=self.font_family)
        self.raw_canvas = FigureCanvasTkAgg(self.raw_fig, master=raw_frame)
        self.raw_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 投影数据画布（新增）
        self.sinogram_fig, self.sinogram_ax = plt.subplots(figsize=(5, 5), dpi=100)
        self.sinogram_ax.set_title("未生成投影数据", fontsize=self.font_size_large, fontfamily=self.font_family)
        self.sinogram_canvas = FigureCanvasTkAgg(self.sinogram_fig, master=sinogram_frame)
        self.sinogram_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 重建结果画布
        self.recon_fig, self.recon_ax = plt.subplots(figsize=(5, 5), dpi=100)
        self.recon_ax.set_title("未进行重建", fontsize=self.font_size_large, fontfamily=self.font_family)
        self.recon_canvas = FigureCanvasTkAgg(self.recon_fig, master=recon_frame)
        self.recon_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_style(self):
        """配置ttk控件的全局样式（字体+内边距）"""
        # 创建ttk样式对象
        style = ttk.Style(self.root)

        # 设置不同类型ttk控件的字体
        style.configure("TLabel", font=self.font_base)  # 普通标签
        style.configure("TButton", font=self.font_base)  # 按钮
        style.configure("TCombobox", font=self.font_large)  # 下拉框
        style.configure("TRadiobutton", font=self.font_base)  # 单选按钮
        style.configure("TEntry", font=self.font_base)  # 输入框

        # 设置LabelFrame（带标题的框架）的样式
        style.configure(
            "TLabelframe.Label",
            font=self.font_large,  # 框架标题用大号字体
            padding=5  # 内边距，让标题不拥挤
        )

    def _on_closing(self):
        """窗口关闭时的清理操作"""
        # 1. 关闭所有Matplotlib的绘图窗口/释放资源
        plt.close('all')
        # 2. 销毁tkinter主窗口
        self.root.destroy()
        # 3. 强制退出Python进程（确保彻底终止）
        import sys
        sys.exit()

    # ---------------------- 新增：Shepp-Logan生成方法 ----------------------
    def _generate_shepp_logan(self):
        """生成Shepp-Logan数据，并标记数据类型"""
        try:
            size = int(self.sl_size_var.get())
            if size <= 0 or size > 1024:
                raise ValueError("尺寸必须为1-1024之间的整数")

            self.root.config(cursor="wait")
            self.root.update()

            if self.sl_data_type.get() == "幻影图像":
                self.raw_data = shepp_logan_phantom(size)
                self.data_type = "shepp_logan_image"
                self.data_source = f"生成的Shepp-Logan幻影图像（{size}x{size}）"
            else:
                # 先生成幻影图像，再生成正弦图
                phantom = shepp_logan_phantom(size)
                self.sinogram_data, self.angles_data = simulate_ct_scan(phantom, num_angles=180)
                self.raw_data = self.sinogram_data
                self.data_type = "shepp_logan_sinogram"
                self.data_source = f"生成的Shepp-Logan正弦图（{size}x{size}）"

            self.root.config(cursor="")

            # 更新显示
            self.file_path_var.set(self.data_source)
            self._display_raw_data()
            if self.data_type == "shepp_logan_sinogram":
                self._display_sinogram_data()
            messagebox.showinfo("成功", f"{self.data_source}生成完成！")

        except ValueError as e:
            messagebox.showerror("错误", f"输入参数无效：{str(e)}")
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("错误", f"生成数据失败：{str(e)}")

    # ---------------------- 原有方法（关键修改：对齐Shepp-Logan显示标准） ----------------------
    def _select_file(self):
        """选择原始图像/投影数据文件，并判断数据类型"""
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
                # 判断文件类型：图像文件→标记为image；数据文件→标记为sinogram
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    # 加载图像
                    img = Image.open(file_path).convert('L')
                    self.raw_data = np.array(img, dtype=np.float32)
                    self.data_type = "image"
                    self.data_source = f"上传图像：{os.path.basename(file_path)}"
                else:
                    # 加载投影数据（需确保是(sinogram, angles)格式，或单独sinogram）
                    if file_path.endswith('.npy'):
                        loaded_data = np.load(file_path, allow_pickle=True)
                        if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                            # 若为(sinogram, angles)元组
                            self.sinogram_data, self.angles_data = loaded_data
                        else:
                            # 若为单独sinogram，自动生成角度
                            self.sinogram_data = loaded_data
                            self.angles_data = np.linspace(0, np.pi, self.sinogram_data.shape[0], endpoint=False)
                    else:
                        # 文本文件仅加载sinogram
                        self.sinogram_data = np.loadtxt(file_path, dtype=np.float32)
                        self.angles_data = np.linspace(0, np.pi, self.sinogram_data.shape[0], endpoint=False)
                    self.raw_data = self.sinogram_data  # 原始数据显示投影数据
                    self.data_type = "sinogram"
                    self.data_source = f"上传投影数据：{os.path.basename(file_path)}"

                # 显示原始数据
                self._display_raw_data()
                # 若为投影数据，同步显示投影数据
                if self.data_type == "sinogram":
                    self._display_sinogram_data()
                messagebox.showinfo("成功", f"{self.data_source}加载成功！")

            except Exception as e:
                messagebox.showerror("错误", f"加载文件失败：{str(e)}")
                self.raw_data = None
                self.data_type = None

    def _display_raw_data(self):
        """显示原始数据"""
        if self.raw_data is not None:
            self.raw_ax.clear()

            # 图像类型数据显示
            if self.data_type in ["image", "shepp_logan_image"]:
                if self.data_type == "shepp_logan_image":
                    # Shepp-Logan图像用标准参数显示
                    self.raw_ax.imshow(
                        self.raw_data,
                        cmap=self.sl_display_config["cmap"],
                        extent=self.sl_display_config["extent"],
                        vmin=self.sl_display_config["vmin"],
                        vmax=self.sl_display_config["vmax"],
                        origin=self.sl_display_config["origin"],
                        alpha=self.sl_display_config["alpha"]
                    )
                else:
                    # 普通图像用默认灰度显示
                    self.raw_ax.imshow(self.raw_data, cmap='gray')
                self.raw_ax.set_title(
                    f"{self.data_source} (尺寸: {self.raw_data.shape})",
                    fontsize=self.font_size_large, fontfamily=self.font_family
                )
            # 投影数据类型显示
            else:
                self.raw_ax.imshow(self.raw_data, cmap='gray')
                self.raw_ax.set_title(
                    f"{self.data_source} (角度数: {self.raw_data.shape[0]}, 探测器数: {self.raw_data.shape[1]})",
                    fontsize=self.font_size_large, fontfamily=self.font_family
                )
            self.raw_ax.axis('off')
            self.raw_canvas.draw()

    def _display_sinogram_data(self):
        """显示投影数据（正弦图）"""
        if self.sinogram_data is not None:
            self.sinogram_ax.clear()
            self.sinogram_ax.imshow(self.sinogram_data.T, cmap='gray')  # 转置后显示更直观
            self.sinogram_ax.set_title(
                f"CT投影数据 (角度数: {len(self.angles_data)}, 探测器数: {self.sinogram_data.shape[1]})",
                fontsize=self.font_size_large, fontfamily=self.font_family
            )
            self.sinogram_ax.axis('off')
            self.sinogram_canvas.draw()

    def _run_reconstruction(self):
        """执行重建算法（核心：根据数据类型自动处理）"""
        if self.raw_data is None:
            messagebox.warning("警告", "请先加载或生成原始数据/图像！")
            return

        selected_algorithm = self.algorithm_var.get()
        if not selected_algorithm:
            messagebox.warning("警告", "请选择重建算法！")
            return

        try:
            self.root.config(cursor="wait")
            self.root.update()

            # 步骤1：根据数据类型生成投影数据（若需要）
            if self.data_type in ["image", "shepp_logan_image"]:
                # 图像类型→先模拟CT扫描生成投影数据
                num_angles = int(self.angle_num_var.get())
                if num_angles <= 0 or num_angles > 360:
                    raise ValueError("投影角度数必须为1-360之间的整数")

                # 模拟CT扫描
                self.sinogram_data, self.angles_data = simulate_ct_scan(self.raw_data, num_angles=num_angles)
                # 显示投影数据
                self._display_sinogram_data()
                messagebox.showinfo("提示", f"已对图像完成CT模拟扫描（{num_angles}个投影角度）")

            # 步骤2：检查投影数据是否存在
            if self.sinogram_data is None or self.angles_data is None:
                raise ValueError("投影数据缺失！无法执行重建")

            # 步骤3：调用重建算法（以direct_reconstruction为例）
            if selected_algorithm == "直接反投影重建":
                # 自定义参数（可根据需要调整）
                custom_params = {
                    "filter_type": "shepp-logan",
                    "sigma": 0.8,
                    "interpolation": "linear"
                }
                # 调用算法接口
                self.recon_result, status, msg = run_direct_reconstruction(
                    self.sinogram_data, self.angles_data, custom_params
                )
            else:
                # 可扩展其他算法调用逻辑
                raise NotImplementedError(f"暂未实现{selected_algorithm}的调用逻辑")

            self.root.config(cursor="")

            # 步骤4：判断重建结果并显示
            if status == "success":
                self._display_recon_result()
                messagebox.showinfo("成功", f"{selected_algorithm} 重建完成！")
            else:
                raise ValueError(f"算法执行失败：{msg}")

        except ValueError as e:
            self.root.config(cursor="")
            messagebox.showerror("错误", f"参数错误：{str(e)}")
        except NotImplementedError as e:
            self.root.config(cursor="")
            messagebox.showerror("错误", str(e))
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("错误", f"重建失败：{str(e)}")

    def _display_recon_result(self):
        """显示重建结果"""
        if self.recon_result is not None:
            self.recon_ax.clear()

            # 如果是Shepp-Logan幻影重建结果，沿用相同的显示参数
            if self.data_source and "Shepp-Logan" in self.data_source:
                self.recon_ax.imshow(
                    self.recon_result,
                    cmap=self.sl_display_config["cmap"],
                    extent=self.sl_display_config["extent"],
                    vmin=self.sl_display_config["vmin"],
                    vmax=self.sl_display_config["vmax"],
                    origin=self.sl_display_config["origin"],
                    alpha=self.sl_display_config["alpha"]
                )
            else:
                self.recon_ax.imshow(self.recon_result, cmap='gray')

            self.recon_ax.set_title(
                f"重建结果 (尺寸: {self.recon_result.shape})",
                fontsize=self.font_size_large,
                fontfamily=self.font_family
            )
            self.recon_ax.axis('off')
            self.recon_canvas.draw()


# ---------------------- 4. 程序入口 ----------------------
if __name__ == "__main__":
    # 确保中文显示正常（与generate_shepp_logan_phantom.py对齐）
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False

    root = tk.Tk()
    app = CTReconstructionApp(root)
    root.mainloop()