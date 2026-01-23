import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os


# ---------------------- 1. 新增：Shepp-Logan 生成模块 ----------------------
def shepp_logan_phantom(size=256):
    """生成Shepp-Logan幻影图像（标准CT测试图像）"""
    # 椭圆参数: [x0, y0, a, b, angle_deg, rho]
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

    # 创建归一化坐标网格 [-1, 1]
    y, x = np.ogrid[-1:1:size * 1j, -1:1:size * 1j]
    phantom = np.zeros((size, size), dtype=np.float32)

    for item in ellipses:
        x0, y0, a, b, angle_deg, gray_val = item
        angle = np.deg2rad(angle_deg)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # 平移
        x_shift = x - x0
        y_shift = y - y0

        # 逆旋转
        xr = cos_a * x_shift + sin_a * y_shift
        yr = -sin_a * x_shift + cos_a * y_shift

        # 椭圆方程
        ellipse_mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
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


# ---------------------- 2. 算法模块（更新为四种算法） ----------------------
def direct_backprojection(raw_data):
    """直接反投影算法 - 模拟实现，替换为你的实际代码"""
    reconstructed = raw_data  # 模拟逻辑，替换为真实算法
    return reconstructed

def fourier_backprojection(raw_data):
    """傅里叶反投影算法 - 模拟实现，替换为你的实际代码"""
    reconstructed = np.flipud(raw_data)  # 模拟逻辑，替换为真实算法
    return reconstructed

def backprojection_filter(raw_data):
    """反投影滤波算法 - 模拟实现，替换为你的实际代码"""
    reconstructed = np.fliplr(raw_data)  # 模拟逻辑，替换为真实算法
    return reconstructed

def filtered_backprojection(raw_data):
    """滤波反投影算法 - 模拟实现，替换为你的实际代码"""
    reconstructed = np.rot90(raw_data)  # 模拟逻辑，替换为真实算法
    return reconstructed

# 算法字典：更新为四种指定算法（键为显示名称，值为对应函数）
RECONSTRUCTION_ALGORITHMS = {
    "直接反投影重建": direct_backprojection,
    "傅里叶反投影重建": fourier_backprojection,
    "反投影滤波重建": backprojection_filter,
    "滤波反投影重建": filtered_backprojection
}


# ---------------------- 3. 主程序界面类（关键修改） ----------------------
class CTReconstructionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CT图像重建系统")
        self.root.geometry("1900x1200")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.font_size_base = 20  # 基础字体（按钮、标签），改这个数字调整大小
        self.font_size_large = 22  # 大号字体（标题、下拉框），改这个数字调整大小
        self.font_family = "SimHei"  # 字体族（已简化为你能正常显示的SimHei）

        # 定义字体配置（同时支持ttk和tk原生控件）
        self.font_base = (self.font_family, self.font_size_base)
        self.font_large = (self.font_family, self.font_size_large)

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
        """创建所有界面组件（新增Shepp-Logan生成相关控件）"""
        # 1. 顶部控制面板
        control_frame = ttk.LabelFrame(self.root, text="控制面板")
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # 第一行：文件选择 + Shepp-Logan生成
        row1_frame = ttk.Frame(control_frame)
        row1_frame.pack(fill=tk.X, padx=5, pady=5)

        # 选择文件按钮
        self.file_path_var = tk.StringVar(value="未选择文件")
        ttk.Button(row1_frame, text="选择原始图像/数据",
                   command=self._select_file).pack(side=tk.LEFT, padx=5)
        ttk.Label(row1_frame, textvariable=self.file_path_var).pack(side=tk.LEFT, padx=5)

        # 新增：Shepp-Logan生成相关控件
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

        # 第二行：算法选择 + 重建按钮
        row2_frame = ttk.Frame(control_frame)
        row2_frame.pack(fill=tk.X, padx=5, pady=5)

        # 算法选择下拉框
        ttk.Label(row2_frame, text="选择重建算法：").pack(side=tk.LEFT, padx=5)
        self.algorithm_var = tk.StringVar()
        algorithm_combobox = ttk.Combobox(row2_frame, textvariable=self.algorithm_var,
                                          values=list(RECONSTRUCTION_ALGORITHMS.keys()),
                                          state="readonly")
        algorithm_combobox.pack(side=tk.LEFT, padx=5)
        if RECONSTRUCTION_ALGORITHMS:
            algorithm_combobox.current(0)

        # 重建按钮
        ttk.Button(row2_frame, text="开始重建",
                   command=self._run_reconstruction).pack(side=tk.LEFT, padx=5)

        # 2. 图像显示区域（保持不变）
        display_frame = ttk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 原始图像显示区
        raw_frame = ttk.LabelFrame(display_frame, text="原始数据/图像")
        raw_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 重建结果显示区
        recon_frame = ttk.LabelFrame(display_frame, text="重建结果")
        recon_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建matplotlib绘图区域
        # 原始图像画布
        self.raw_fig, self.raw_ax = plt.subplots(figsize=(5, 5), dpi=100)
        # ========== 关键修改：添加fontsize和fontfamily ==========
        self.raw_ax.set_title("未加载数据",
                              fontsize=self.font_size_large,  # 用全局定义的大号字体大小
                              fontfamily=self.font_family)  # 用全局定义的字体族（SimHei）
        self.raw_canvas = FigureCanvasTkAgg(self.raw_fig, master=raw_frame)
        self.raw_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 重建结果画布
        self.recon_fig, self.recon_ax = plt.subplots(figsize=(5, 5), dpi=100)
        # ========== 关键修改：添加fontsize和fontfamily ==========
        self.recon_ax.set_title("未进行重建",
                                fontsize=self.font_size_large,  # 全局大号字体大小
                                fontfamily=self.font_family)  # 全局字体族
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
        """生成Shepp-Logan数据（幻影图像或正弦图）"""
        try:
            # 获取用户输入的尺寸
            size = int(self.sl_size_var.get())
            if size <= 0 or size > 1024:
                raise ValueError("尺寸必须为1-1024之间的整数")

            # 根据选择生成对应数据
            self.root.config(cursor="wait")
            self.root.update()

            if self.sl_data_type.get() == "幻影图像":
                self.raw_data = shepp_logan_phantom(size)
                self.data_source = "生成的Shepp-Logan幻影图像"
            else:
                # 先生成幻影图像，再生成正弦图
                phantom = shepp_logan_phantom(size)
                self.raw_data = generate_sinogram(phantom)
                self.data_source = "生成的Shepp-Logan正弦图"

            self.root.config(cursor="")

            # 更新显示和状态
            self.file_path_var.set(self.data_source)
            self._display_raw_data()
            messagebox.showinfo("成功", f"{self.data_source}生成完成！尺寸：{size}x{size}")

        except ValueError as e:
            messagebox.showerror("错误", f"输入参数无效：{str(e)}")
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("错误", f"生成数据失败：{str(e)}")

    # ---------------------- 原有方法（保持不变） ----------------------
    def _select_file(self):
        """选择原始图像/数据文件"""
        file_path = filedialog.askopenfilename(
            title="选择CT原始数据/图像",
            filetypes=[
                ("图像文件", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("数据文件", "*.npy *.txt *.csv"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.file_path_var.set(os.path.basename(file_path))
            try:
                if file_path.endswith(('.npy', '.txt', '.csv')):
                    if file_path.endswith('.npy'):
                        self.raw_data = np.load(file_path)
                    else:
                        self.raw_data = np.loadtxt(file_path)
                else:
                    img = Image.open(file_path).convert('L')
                    self.raw_data = np.array(img)

                self.data_source = "上传文件"
                self._display_raw_data()
                messagebox.showinfo("成功", "原始数据/图像加载成功！")

            except Exception as e:
                messagebox.showerror("错误", f"加载文件失败：{str(e)}")
                self.raw_data = None

    def _display_raw_data(self):
        if self.raw_data is not None:
            self.raw_ax.clear()
            self.raw_ax.imshow(self.raw_data, cmap='gray')
            # 关键修改：指定字体大小（用self.font_size_large）
            self.raw_ax.set_title(
                f"{self.data_source} (尺寸: {self.raw_data.shape})",
                fontsize=self.font_size_large,
                fontfamily=self.font_family
            )
            self.raw_ax.axis('off')
            self.raw_canvas.draw()

    def _run_reconstruction(self):
        """执行重建算法"""
        if self.raw_data is None:
            messagebox.warning("警告", "请先加载或生成原始数据/图像！")
            return

        selected_algorithm = self.algorithm_var.get()
        if not selected_algorithm:
            messagebox.warning("警告", "请选择重建算法！")
            return

        try:
            recon_func = RECONSTRUCTION_ALGORITHMS[selected_algorithm]

            self.root.config(cursor="wait")
            self.root.update()
            self.recon_result = recon_func(self.raw_data)
            self.root.config(cursor="")

            self._display_recon_result()
            messagebox.showinfo("成功", f"{selected_algorithm} 重建完成！")

        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("错误", f"重建失败：{str(e)}")

    def _display_recon_result(self):
        if self.recon_result is not None:
            self.recon_ax.clear()
            self.recon_ax.imshow(self.recon_result, cmap='gray')
            # 关键修改：指定字体大小
            self.recon_ax.set_title(
                f"重建结果 (尺寸: {self.recon_result.shape})",
                fontsize=self.font_size_large,
                fontfamily=self.font_family
            )
            self.recon_ax.axis('off')
            self.recon_canvas.draw()

# ---------------------- 4. 程序入口 ----------------------
if __name__ == "__main__":
    # 确保中文显示正常
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False

    root = tk.Tk()
    app = CTReconstructionApp(root)
    root.mainloop()