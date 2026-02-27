# -*- coding: utf-8 -*-
# 文件名: ul_recon_app_01.py
# 功能: B超图像重建系统 - 支持PGM(P5)格式解析、DAT二进制处理、凸阵/线阵图像重建
# 核心流程: PGM文件解析 → DAT格式转换 → 信号处理(FFT/正交解调/FIR滤波) → 图像重建 → GUI显示对比
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import struct
from scipy import signal, ndimage
import re

# -------------------------- PGM(P5)格式解析与数据转换模块 --------------------------
def read_pgm_p5(pgm_path):
    """
    解析PGM P5格式二进制文件（B超原始数据存储格式）
    格式说明：
    - 前544字节为文件头（ASCII格式，包含标识、尺寸、注释、最大值等）
    - 544字节后为数据区，每个像素为4字节int32类型（RF原始信号）
    - 注释行可能包含HalfAngle(rad)（凸阵探头半张角，单位弧度）
    
    参数:
        pgm_path: str - PGM文件路径
    返回:
        img: np.ndarray(int32) - 二维图像数组，shape=(height, width)
        header: dict - 解析后的头信息，包含：
            width/height: 图像宽/高
            maxval: 像素最大值（PGM格式字段）
            comments: 所有注释行列表
            half_angle_rad: 凸阵探头半张角（从注释中提取，无则为None）
    异常:
        ValueError - P5标识缺失、尺寸/最大值解析失败
    """
    with open(pgm_path, 'rb') as f:
        # 读取固定长度头信息（544字节），兼容ASCII编码（忽略无法解码的字节）
        header_blob = f.read(544)
        header_text = header_blob.decode('ascii', errors='ignore')
        
        # 提取注释行（PGM注释行以#开头）
        comments = [l.strip() for l in header_text.splitlines() if l.strip().startswith('#')]
        
        # 提取非注释行（过滤空行和注释行），验证P5标识
        lines = [l.strip() for l in header_text.splitlines() if l.strip() and not l.startswith('#')]
        if not lines or not lines[0].startswith('P5'):
            raise ValueError("PGM格式错误：头信息中未检测到P5标识，非标准P5格式文件")
        
        # 解析图像尺寸（第2行）和像素最大值（第3行）
        try:
            dims = lines[1].split()
            width, height = int(dims[0]), int(dims[1])
            maxval = int(lines[2])
        except (IndexError, ValueError) as e:
            raise ValueError(f"PGM头信息解析失败：尺寸/最大值字段格式错误 - {e}")
        
        # 从注释行提取凸阵探头半张角（匹配"HalfAngle(rad): 数值"格式）
        half_angle_rad = None
        for comment in comments:
            match = re.search(r'HalfAngle\(rad\):\s*([\d\.]+)', comment)
            if match:
                half_angle_rad = float(match.group(1))
                break
        
        # 定位到544字节位置，读取数据区（int32格式）
        f.seek(544)
        data = np.fromfile(f, dtype=np.int32)
    
    # 数据尺寸校验与容错：确保数据区长度匹配头信息定义的宽高
    expected_size = width * height
    if data.size < expected_size:
        # 数据不足时用0填充，避免维度不匹配导致后续处理崩溃
        print(f"警告：PGM数据区长度({data.size})小于头信息定义尺寸({expected_size})，将用0填充缺失部分")
        img = np.zeros((height, width), dtype=np.int32)
        available = min(data.size, expected_size)
        img.flat[:available] = data[:available]
    else:
        # 数据充足时截取并重塑为二维数组
        img = data[:expected_size].reshape((height, width))
    
    # 封装头信息字典
    header = {
        "width": width, 
        "height": height, 
        "maxval": maxval,
        "comments": comments,
        "half_angle_rad": half_angle_rad
    }
    return img, header

def pgm_pixels_to_int16(img, maxval):
    """
    PGM像素数据类型转换（兼容后续信号处理算法）
    设计背景：
    - 原始PGM数据为int32类型（RF原始信号），无需像8位图像做-128偏移
    - 保持int32精度以适配FIR滤波等算法的动态范围要求
    - 函数命名保留历史兼容（实际返回int32，避免修改调用端）
    
    参数:
        img: np.ndarray - 原始PGM像素数组（int32）
        maxval: int - PGM头信息中的像素最大值（未实际使用，保留参数兼容）
    返回:
        np.ndarray(int32) - 类型转换后的数组（与输入一致，仅做类型显式声明）
    """
    return img.astype(np.int32)

def save_dat_from_array(arr_int32, out_path):
    """
    将二维int32数组写入DAT二进制文件（B超数据处理中间格式）
    参数:
        arr_int32: np.ndarray(int32) - 待写入的二维数组
        out_path: str - DAT文件输出路径
    返回:
        str - 实际输出路径（与out_path一致，便于链式调用）
    """
    arr_int32.tofile(out_path)
    return out_path

# -------------------------- 核心信号处理与算法模块 --------------------------
def changeB2T(file_path, n_lines=None, n_samples=None):
    """
    读取DAT二进制文件并重塑为二维扫描数据（行=扫描线数，列=采样点数）
    格式说明：DAT文件中每个数据点为4字节int32类型（RF原始信号）
    
    参数:
        file_path: str - DAT文件路径
        n_lines: int/None - 扫描线数（指定则按该值重塑，None则使用默认值）
        n_samples: int/None - 单条扫描线采样点数（指定则按该值重塑，None则使用默认值）
    返回:
        np.ndarray(int32) - 二维扫描数据，shape=(实际扫描线数, 采样点数)
    异常:
        ValueError - 无法推断尺寸（数据量不足且未指定n_lines/n_samples）
    容错:
        数据量不足时，按实际可提取的完整扫描线数返回（避免维度不匹配）
    """
    with open(file_path, 'rb') as fid:
        # 读取全部数据（int32格式）
        data = np.fromfile(fid, dtype=np.int32)
    
    # 优先使用指定的扫描线数和采样点数重塑
    if n_lines is not None and n_samples is not None:
        total = n_lines * n_samples
        if data.size < total:
            # 数据不足时，计算实际可提取的完整扫描线数
            actual_lines = data.size // n_samples
            da = data[:actual_lines * n_samples].reshape((actual_lines, n_samples))
            return da
        da = data[:total].reshape((n_lines, n_samples))
        return da
    
    # 未指定尺寸时，回退到历史默认值（240行×7618列）
    if data.size >= 240 * 7618:
        da = data[:240 * 7618].reshape((240, 7618))
        return da
    
    # 既未指定尺寸，数据量也不足默认值，无法推断维度
    raise ValueError("DAT文件尺寸推断失败：数据量不足默认值(240×7618)，且未指定n_lines/n_samples参数")

def processF(Data):
    """
    单条扫描线RF信号处理（核心算法）
    处理流程：FFT频域峰值提取 → 正交解调 → FIR低通滤波 → 降采样 → 对数压缩
    设计背景：
    - 对齐MATLAB原版算法索引（MATLAB为1起始，Python为0起始，需手动+1/-1适配）
    - 降采样逻辑严格复刻MATLAB循环，保证结果一致性
    - 对数压缩用于提升图像视觉对比度
    
    参数:
        Data: np.ndarray(int32) - 单条扫描线原始RF数据（一维数组）
    返回:
        np.ndarray(float) - 处理后的单条扫描线数据（降采样+对数压缩后）
    """
    N = len(Data)
    # 1. FFT变换并提取前半段频域峰值（避免共轭对称部分）
    mag = np.abs(np.fft.fft(Data))
    m = mag.shape[0]
    # 查找前半段最大值索引（+1适配MATLAB 1起始索引）
    max_val = np.max(mag[:int(m/2)])
    k = np.where(mag[:int(m/2)] == max_val)[0][0] + 1
    # 计算角频率（用于正交解调）
    w = -2 * np.pi * k / N

    # 2. 正交解调（分离同相/正交分量）
    x = np.arange(N)
    Q = np.cos(w * x) * Data  # 正交分量
    I = np.sin(w * x) * Data  # 同相分量

    # 3. FIR低通滤波（滤除高频噪声）
    T = 1 / 40000          # 采样周期（40kHz采样率）
    f = 100                # 截止频率（100Hz）
    wn = 2 * T * f * 1.001 # 归一化截止频率（+0.001避免边界值）
    n = 70                 # 滤波器阶数
    fir_coeff = signal.firwin(n + 1, wn)  # 生成FIR低通滤波器系数
    # FFT卷积实现滤波（mode='same'保证输出长度与输入一致）
    Qf = signal.fftconvolve(Q, fir_coeff, mode='same')
    If = signal.fftconvolve(I, fir_coeff, mode='same')
    # 计算解调后信号幅度（未滤波的I/Q用于幅度计算，保持与原版一致）
    Fdata = np.sqrt(Q**2 + I**2)

    # 4. 降采样（16倍抽取，严格对齐MATLAB循环逻辑）
    Sdata = []
    index = 1
    while True:
        p = 16 * (index - 1) + 1  # MATLAB 1起始索引计算
        if p > N:
            break
        Sdata.append(Fdata[p-1])  # 转换为Python 0起始索引
        index += 1
    Sdata = np.array(Sdata) / 255.0  # 归一化到0~1范围

    # 5. 对数压缩（提升图像对比度）
    a = 1.5  # 压缩系数（经验值）
    b = 0.02 # 偏移量（避免0值对数无意义）
    for i in range(len(Sdata)):
        Sdata[i] = a * np.log(Sdata[i] + 1) + b
        # 限幅（避免超过255导致显示异常）
        if Sdata[i] > 255:
            Sdata[i] = 255

    return Sdata

def frameProcess(file_path, n_lines=None, n_samples=None):
    """
    单帧DAT数据批量处理（逐行调用processF）
    参数:
        file_path: str - DAT文件路径
        n_lines: int/None - 扫描线数（传递给changeB2T）
        n_samples: int/None - 单条扫描线采样点数（传递给changeB2T）
    返回:
        np.ndarray(float) - 处理后的帧数据，shape=(扫描线数, 降采样后采样点数)
    """
    # 读取并重塑DAT数据为二维数组
    data = changeB2T(file_path, n_lines=n_lines, n_samples=n_samples)
    # 取绝对值（RF信号幅度处理）
    data = np.abs(data)

    # 逐行处理扫描线数据
    first_line = processF(data[0, :])  # 先处理首行获取降采样后长度
    Cdata = np.zeros((data.shape[0], len(first_line)))  # 初始化结果数组
    for n in range(data.shape[0]):
        Cdata[n, :] = processF(data[n, :])

    # 转置（匹配MATLAB转置操作，行列互换以适配后续重建逻辑）
    CData = Cdata.T
    return CData

def image_reconstruct(d, half_angle_rad=None):
    """
    图像重建统一入口（适配凸阵/线阵，封装底层重建逻辑）
    参数:
        d: np.ndarray - 处理后的帧数据（frameProcess输出）
        half_angle_rad: float/None - 凸阵探头半张角（线阵时无效）
    返回:
        np.ndarray - 重建后的图像数组
    """
    return image_reconstruct_convex(d, half_angle_rad)

# -------------------------- 图像重建算法（凸阵/线阵） --------------------------
def image_reconstruct_linear(d):
    """
    线阵探头图像重建（双三次插值放大，提升显示效果）
    设计背景：
    - 线阵扫描为矩形区域，无需极坐标转换
    - 双三次插值（order=3）相比线性插值更平滑，消除马赛克感
    - 垂直/水平放大倍率为经验值，平衡分辨率与显示效率
    
    参数:
        d: np.ndarray - 处理后的帧数据（frameProcess输出）
    返回:
        np.ndarray - 插值放大后的重建图像
    """
    # 插值倍率（垂直2倍，水平4倍，可根据需求调整）
    scale_h = 2.0
    scale_w = 4.0
    # 双三次插值放大
    recon = ndimage.zoom(d, (scale_h, scale_w), order=3)
    return recon

def image_reconstruct_convex(d, half_angle_rad=None):
    """
    凸阵探头图像重建（极坐标→笛卡尔坐标转换，双线性插值消除空洞）
    核心逻辑：
    1. 极坐标（扫描线/采样点）转换为笛卡尔坐标（像素坐标）
    2. 双线性插值填充像素值，避免极坐标转换后的空洞问题
    3. 背景区域（无效值）设为-1，后续显示为黑色
    
    参数:
        d: np.ndarray - 处理后的帧数据（frameProcess输出）
        half_angle_rad: float/None - 凸阵探头半张角（None则使用默认68°）
    返回:
        np.ndarray - 重建后的凸阵图像（包含背景掩码值-1）
    """
    m, n = d.shape
    # 半张角默认值（68°转换为弧度）
    if half_angle_rad is None:
        total_angle_deg = 68
        half_angle_rad = (total_angle_deg / 2) / 180 * np.pi
    
    total_angle_rad = 2 * half_angle_rad  # 总扫描角度
    r_start = 70.0  # 起始扫描半径（经验值，单位：mm）
    
    # 1. 计算重建图像的目标尺寸（基于极坐标范围）
    R_max = r_start + m  # 最大扫描半径
    max_width = 2 * R_max * np.sin(half_angle_rad)  # 图像宽度（笛卡尔坐标）
    max_height = R_max - r_start * np.cos(half_angle_rad)  # 图像高度
    
    # 分辨率缩放系数（1.0为原始分辨率，可调整）
    res_scale = 1.0
    new_w = int(np.ceil(max_width * res_scale)) + 2  # 宽度向上取整+2避免边界截断
    new_h = int(np.ceil(max_height * res_scale)) + 2  # 高度向上取整+2避免边界截断
    
    # 2. 生成笛卡尔坐标网格（目标图像像素坐标）
    xv, yv = np.meshgrid(np.arange(new_w), np.arange(new_h))
    
    # 3. 坐标偏移校正（以探头顶点为原点）
    X_center = (new_w - 1) / 2.0  # 图像水平中心
    Y_offset = r_start * np.cos(half_angle_rad)  # 垂直偏移（补偿起始半径）
    # 转换为物理坐标（px: 水平，py: 垂直）
    px = xv - X_center
    py = -yv - Y_offset
    
    # 4. 笛卡尔坐标→极坐标转换（映射到原始数据索引）
    R = np.sqrt(px**2 + py**2)  # 极径（映射到原始数据行索引）
    theta = np.arctan2(px, -py)  # 极角（映射到原始数据列索引）
    
    # 5. 极坐标→原始数据索引映射
    row_idx = R - r_start  # 行索引（扫描深度方向）
    col_idx = (theta + half_angle_rad) / (2 * half_angle_rad) * (n - 1)  # 列索引（角度方向）
    
    # 6. 双线性插值填充像素值（order=1），背景设为-1
    recon = ndimage.map_coordinates(
        d, 
        [row_idx, col_idx], 
        order=1,          # 双线性插值（平衡精度与速度）
        mode='constant',  # 边界外值设为常数
        cval=-1           # 背景填充值（后续显示为黑色）
    )
    
    return recon

# -------------------------- GUI交互与显示模块 --------------------------
class BUSReconstructionApp:
    """
    B超图像重建系统GUI主类
    界面布局：
    - 顶部控制面板：PGM文件上传、参考图像上传、探头类型选择、重建执行
    - 底部显示区域：重建图像（左）、参考图像（右）对比显示
    核心属性：
        root: tk.Tk - 主窗口对象
        frame_data_list: list - 存储处理后的帧数据（支持多帧平均）
        recon_data: np.ndarray - 重建后的图像数据
        reference_image: np.ndarray - 参考图像数据
        pgm_header: dict - 解析后的PGM头信息
        probe_type: tk.StringVar - 探头类型（convex/linear）
    """
    def __init__(self, root):
        self.root = root
        self.root.title("B超图像重建系统")
        self.root.geometry("1600x900")  # 初始窗口尺寸
        
        # 全局UI配置（集中管理，便于调整）
        self.UI_FONT_SIZE = 18          # 全局字体大小
        self.BTN_PADDING_XY = (20, 8)   # 按钮内边距（左右，上下）

        # 初始化样式
        self.style = ttk.Style()
        self.style.configure(".", font=("SimHei", self.UI_FONT_SIZE))  # 全局字体
        self.style.configure("TButton", padding=self.BTN_PADDING_XY)   # 按钮样式

        # 绑定窗口关闭事件（确保彻底退出程序）
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 数据存储初始化
        self.frame_data_list = []       # 处理后的帧数据列表
        self.recon_data = None          # 重建后的图像数据
        self.reference_image = None     # 参考图像数据
        self.pgm_header = None          # PGM头信息
        self.probe_type = tk.StringVar(value="convex")  # 默认探头类型：凸阵

        # 初始化界面控件
        self._create_widgets()

    def _create_widgets(self):
        """
        构建GUI控件布局
        布局结构：
        - top_frame: 顶部控制面板（文件上传+参数选择）
        - 底部：Matplotlib画布（重建图像+参考图像对比）
        """
        # 顶部控制面板（填充水平方向）
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # 第一行：文件上传控件
        file_frame = ttk.Frame(top_frame)
        file_frame.pack(side=tk.LEFT, padx=10)
        
        # PGM文件上传按钮
        upload_btn = ttk.Button(file_frame, text="上传PGM数据", command=self.upload_pgm_file)
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        # 参考图像上传按钮
        ref_btn = ttk.Button(file_frame, text="上传参考图像", command=self.upload_reference_image)
        ref_btn.pack(side=tk.LEFT, padx=5)

        # 第二行：参数选择与执行控件
        action_frame = ttk.Frame(top_frame)
        action_frame.pack(side=tk.LEFT, padx=30)
        
        # 探头类型选择（单选按钮）
        ttk.Label(action_frame, text="探头类型:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(action_frame, text="凸阵", variable=self.probe_type, value="convex").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(action_frame, text="线阵", variable=self.probe_type, value="linear").pack(side=tk.LEFT, padx=2)
        
        # 重建执行按钮
        recon_btn = ttk.Button(action_frame, text="开始重建", command=self.reconstruct_image)
        recon_btn.pack(side=tk.LEFT, padx=20)

        # 底部图像显示区域（1行2列布局，左右各占50%）
        self.fig = plt.figure(figsize=(28, 14))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.1)
        
        # 子图1：重建后图像
        self.ax2 = self.fig.add_subplot(gs[0, 0])
        # 子图2：参考图像
        self.ax3 = self.fig.add_subplot(gs[0, 1])
        
        # 子图标题与样式初始化
        self.ax2.set_title("重建后图像", fontsize=20)
        self.ax3.set_title("参考图像", fontsize=20)
        
        # 强制正方形显示（便于图像对比）
        self.ax2.set_box_aspect(1)
        self.ax3.set_box_aspect(1)
        
        # 隐藏坐标轴刻度，保留黑色外框（提升视觉效果）
        for ax in [self.ax2, self.ax3]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.5)
        
        # 绑定Matplotlib画布到Tkinter窗口
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # 调整子图边距（减少白边，最大化图像显示区域）
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)

    def upload_reference_image(self):
        """
        上传参考图像并显示到右侧子图
        支持格式：BMP/JPG/JPEG
        异常处理：文件选择取消、图像读取失败（弹窗提示）
        """
        # 打开文件选择对话框
        ref_path = filedialog.askopenfilename(
            title="选择参考图像文件",
            filetypes=[("图像文件", "*.bmp *.jpg *.jpeg"), ("所有文件", "*.*")]
        )
        if not ref_path:  # 用户取消选择
            return
        
        try:
            # 读取参考图像
            img = plt.imread(ref_path)
            self.reference_image = img
            
            # 清空右侧子图并显示参考图像
            self.ax3.clear()
            self.ax3.imshow(img)
            self.ax3.set_title(f"参考图像\n({ref_path.split('/')[-1]})")
            
            # 恢复子图样式（clear会重置spines）
            self.ax3.set_xticks([])
            self.ax3.set_yticks([])
            for spine in self.ax3.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.5)
            
            self.ax3.set_box_aspect(1)
            self.canvas.draw()  # 刷新画布
        except Exception as e:
            messagebox.showerror("错误", f"参考图像加载失败：{str(e)}")

    def upload_pgm_file(self):
        """
        上传PGM文件并执行预处理：
        1. 解析PGM头信息和数据
        2. 转换为int32格式
        3. 生成DAT文件
        4. 调用frameProcess处理帧数据
        5. 初始化重建图像显示区域
        异常处理：文件选择取消、PGM解析失败（弹窗提示）
        """
        # 打开文件选择对话框
        pgm_path = filedialog.askopenfilename(
            title="选择PGM格式B超数据文件",
            filetypes=[("PGM文件", "*.pgm"), ("所有文件", "*.*")]
        )
        if not pgm_path:  # 用户取消选择
            return
        
        try:
            # 解析PGM文件
            img, header = read_pgm_p5(pgm_path)
            self.pgm_header = header
            
            # 数据类型转换（兼容后续处理）
            signed = pgm_pixels_to_int16(img, header["maxval"])
            
            # 获取扫描线数和采样点数
            n_lines, n_samples = signed.shape[0], signed.shape[1]
            
            # 生成DAT文件（路径与PGM文件相同，后缀替换为.dat）
            dat_path = pgm_path[:-4] + ".dat"
            save_dat_from_array(signed, dat_path)
            
            # 处理帧数据
            frame_data = frameProcess(dat_path, n_lines=n_lines, n_samples=n_samples)
            self.frame_data_list = [frame_data]
            
            # 组装提示信息（包含PGM头信息摘要）
            angle_info = f"\n识别到扫描角度 (HalfAngle): {header['half_angle_rad']:.4f} rad" if header.get('half_angle_rad') else "\n未识别到角度信息，将使用默认值"
            meta_info = ""
            if header["comments"]:
                meta_info = "\n".join(header["comments"][:5])  # 只显示前5条注释
            
            # 弹窗提示处理成功
            messagebox.showinfo(
                "成功",
                f"PGM加载完成：{pgm_path}\n尺寸：{header['width']}×{header['height']}\nmaxval：{header['maxval']}\n已生成DAT：{dat_path}{angle_info}\n\n文件头摘要：\n{meta_info}"
            )
            
            # 初始化重建图像显示区域（显示"待重建"提示）
            self.show_reconstructed_placeholder()
        except Exception as e:
            messagebox.showerror("错误", f"PGM处理失败：{str(e)}")

    def show_reconstructed_placeholder(self):
        """
        初始化重建图像显示区域（上传PGM后，重建前）
        清空左侧子图，显示"待重建"提示，恢复子图样式
        """
        self.ax2.clear()
        self.ax2.set_title("重建后图像 (待重建)")
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        
        # 恢复黑色外框
        for spine in self.ax2.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        
        self.canvas.draw()  # 刷新画布

    def reconstruct_image(self):
        """
        执行图像重建并显示到左侧子图
        流程：
        1. 校验数据是否已加载
        2. 多帧平均（抵消噪声）
        3. 根据探头类型选择重建算法
        4. 显示重建图像（掩码无效值为黑色）
        异常处理：未加载数据（弹窗提示）
        """
        # 数据校验
        if not self.frame_data_list:
            messagebox.showwarning("警告", "请先上传PGM数据文件！")
            return

        # 多帧平均（当前仅单帧，后续可扩展多帧）
        d_avg = np.mean(self.frame_data_list, axis=0)
        
        # 根据探头类型执行重建
        if self.probe_type.get() == "linear":
            self.recon_data = image_reconstruct_linear(d_avg)
            aspect_type = 'auto'  # 线阵自适应显示比例
        else:
            # 提取凸阵探头半张角（从PGM头信息）
            half_angle = self.pgm_header.get("half_angle_rad") if self.pgm_header else None
            self.recon_data = image_reconstruct_convex(d_avg, half_angle_rad=half_angle)
            aspect_type = 'equal'  # 凸阵强制等比例显示（保证几何准确性）

        # 显示重建图像（掩码小于0的无效值为黑色）
        self.ax2.clear()
        img = np.ma.masked_less(self.recon_data, 0)  # 创建掩码数组（<0为掩码）
        cm = plt.cm.gray.copy()                      # 复制灰度色图
        cm.set_bad(color='black')                    # 掩码区域显示为黑色
        
        # 绘制重建图像（origin='upper'匹配图像坐标系）
        self.ax2.imshow(
            img, 
            cmap=cm, 
            origin='upper', 
            aspect=aspect_type, 
            interpolation='nearest',  # 最近邻插值（保留细节）
            extent=[0, img.shape[1], img.shape[0], 0]  # 坐标范围匹配图像尺寸
        )
        
        # 组装标题（包含探头类型和角度信息）
        angle_str = ""
        if self.probe_type.get() == "convex":
            angle = self.pgm_header.get("half_angle_rad") if self.pgm_header else None
            if angle:
                angle_str = f" (角度: {angle:.4f} rad)"
            else:
                angle_str = " (角度: 默认 68°)"
        
        self.ax2.set_title(f"重建后图像 ({'线阵' if self.probe_type.get() == 'linear' else '凸阵'}){angle_str}")
        
        # 恢复子图样式
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        for spine in self.ax2.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        
        self.ax2.margins(0)  # 移除边距
        # 调整子图布局（避免tight_layout警告）
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
        self.canvas.draw()  # 刷新画布

    def on_closing(self):
        """
        窗口关闭事件处理（彻底退出程序）
        解决Tkinter主线程残留问题：
        1. 停止事件循环
        2. 销毁窗口组件
        3. 强制退出Python解释器
        """
        try:
            self.root.quit()     # 停止Tkinter事件循环
            self.root.destroy()  # 销毁所有窗口组件
            import sys
            sys.exit(0)         # 强制退出（释放终端）
        except Exception:
            import sys
            sys.exit(0)

# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":
    # 配置Matplotlib中文显示（解决中文乱码）
    plt.rcParams["font.family"] = "SimHei"        # 设置中文字体
    plt.rcParams["axes.unicode_minus"] = False    # 解决负号显示问题

    # 初始化Tkinter主窗口
    root = tk.Tk()
    # 实例化GUI应用
    app = BUSReconstructionApp(root)
    # 启动事件循环
    root.mainloop()