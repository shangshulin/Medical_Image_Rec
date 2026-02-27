#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 功能模块：MRI K空间图像重建可视化工具
# 核心能力：读取FID格式MRI原始数据、执行二维傅里叶反变换重建、多切片切换、参考图像对比
# 适配场景：临床MRI影像后处理、科研级K空间数据解析
# 依赖库：tkinter(UI)、numpy(数值计算)、matplotlib(可视化)、scipy(图像重采样)
# 维护人：xxx
# 最后修改时间：2024-XX-XX

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

# ===================== 全局配置与初始化 =====================
# 强制指定matplotlib后端为TkAgg，避免与tkinter GUI框架冲突
plt.switch_backend('TkAgg')
# 配置matplotlib中文显示：设置默认字体为黑体，解决中文标签乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# ===================== 核心工具函数 =====================
def read_fid(file_path):
    """
    读取FID格式MRI原始数据文件（二进制），解析K空间复数数据
    FID文件结构说明：
    - 前5个int32：文件版本+5个数据段大小
    - 第5段：维度信息（D1~D4）+ 复数K空间数据（实部+虚部）
    - 数据存储：float32类型，按维度(D1,D2,D3,D4)逐层存储，每个点包含实部+虚部
    
    参数：
        file_path (str): FID文件绝对/相对路径
    返回值：
        tau (np.ndarray): 时间轴数组，形状(D1,1)，用于后续谱分析
        Data (np.ndarray): K空间复数数据，形状(D1,D2,D3,D4)，dtype=complex64
        pp (int): 预留参数，暂固定为0
        section (int): 切片数量（即D3维度大小），用于UI切片选择控件初始化
    异常处理：
        文件打开失败时弹出错误提示框，返回4个None
    """
    # 采样宽度(SW)：单位kHz，与MATLAB解析逻辑保持一致，用于计算时间轴
    SAMPLING_WIDTH = 20  
    
    # 尝试以二进制只读模式打开文件
    try:
        fid = open(file_path, 'rb')
    except Exception as e:
        messagebox.showerror("文件读取错误", f"无法打开FID文件：{str(e)}")
        return None, None, None, None
    
    # 内部辅助函数：从文件当前指针位置读取一个32位有符号整数（小端序）
    def read_int32():
        # 读取4字节数据并转换为int32，[0]取数组唯一元素
        return np.frombuffer(fid.read(4), dtype=np.int32)[0]
    
    # 读取文件头基础信息
    file_version = read_int32()          # FID文件版本号（预留字段）
    section1_size = read_int32()         # 第1数据段字节数
    section2_size = read_int32()         # 第2数据段字节数
    section3_size = read_int32()         # 第3数据段字节数
    section4_size = read_int32()         # 第4数据段字节数
    section5_size = read_int32()         # 第5数据段字节数（K空间数据段）
    
    # 计算前4个数据段总字节数，定位到第5段（K空间数据）起始位置
    offset_bytes = section1_size + section2_size + section3_size + section4_size
    fid.read(offset_bytes)  # 跳过前4段字节，移动文件指针到第5段
    
    # 读取K空间数据维度信息（核心参数）
    dim1 = read_int32()  # D1：频率编码方向点数（如256，频率编码步长）
    dim2 = read_int32()  # D2：相位编码方向点数（如256，相位编码步长）
    dim3 = read_int32()  # D3：切片数量（如32，3D扫描的层数）
    dim4 = read_int32()  # D4：回波数/接收通道数（如1，单通道采集）
    
    # 初始化实部/虚部数组：float32类型，匹配FID文件数据存储精度
    data_real = np.zeros((dim1, dim2, dim3, dim4), dtype=np.float32)
    data_imag = np.zeros((dim1, dim2, dim3, dim4), dtype=np.float32)
    
    # 按维度循环读取复数数据（实部+虚部成对存储）
    # 循环顺序：D4(通道) → D3(切片) → D2(相位) → D1(频率)，匹配文件存储顺序
    for l in range(dim4):          # 遍历通道/回波维度
        for k in range(dim3):      # 遍历切片维度
            for j in range(dim2):  # 遍历相位编码维度
                # 读取当前相位编码层的所有频率点数据：
                # 每个频率点占2个float32（实部+虚部），共2*dim1个float32
                # 每个float32占4字节，因此单次读取字节数=2*dim1*4
                raw_data = np.frombuffer(
                    fid.read(2 * dim1 * 4), 
                    dtype=np.float32
                ).reshape(-1, 2)  # 重塑为(N,2)数组：[:,0]实部，[:,1]虚部
                
                # 赋值到对应维度数组
                data_real[:, j, k, l] = raw_data[:, 0]  # 频率维度所有实部
                data_imag[:, j, k, l] = raw_data[:, 1]  # 频率维度所有虚部
    
    # 关闭文件句柄，释放资源
    fid.close()
    
    # 生成时间轴：用于后续频谱分析，形状(dim1,1)保证维度匹配
    tau = np.arange(dim1)[:, np.newaxis] / SAMPLING_WIDTH
    # 合并实部虚部为复数K空间数据（核心输出）
    k_space_data = data_real + 1j * data_imag
    # 预留参数，保持与原始解析逻辑兼容
    pp = 0
    # 切片数量（用于UI层控制）
    section = dim3
    
    return tau, k_space_data, pp, section

def gray_trans(img1, para):
    """
    灰度拉伸函数（对标MATLAB gray_trans实现）
    将图像像素值线性映射到指定灰度范围，解决低对比度图像可视化问题
    
    参数：
        img1 (np.ndarray): 输入图像数组（任意维度，任意数值范围）
        para (list/tuple): 目标灰度范围，格式[min_val, max_val]，需满足0≤min_val≤max_val≤255
    返回值：
        img2 (np.ndarray): 拉伸后的图像，dtype=uint8，像素值范围[para[0], para[1]]
    异常：
        输入参数不合法时抛出ValueError
    """
    # 解包目标灰度范围
    min_target, max_target = para
    
    # 参数合法性校验
    if min_target > max_target:
        raise ValueError("灰度拉伸参数错误：最小阈值不能大于最大阈值")
    if min_target < 0 or max_target > 255:
        raise ValueError("灰度拉伸参数错误：阈值需在[0,255]范围内")
    
    # 转换为float64避免精度丢失，后续归一化计算
    img = img1.astype(np.float64)
    # 计算输入图像的像素值极值
    img_min = np.min(img)
    img_max = np.max(img)
    
    # 处理极值差为0的特殊情况（图像所有像素值相同）
    if np.abs(img_max - img_min) < 1e-8:
        # 所有像素赋值为最小目标灰度
        img2 = np.ones_like(img) * min_target
    else:
        # 线性归一化到[0,1] → 映射到[min_target, max_target]
        img_normalized = (img - img_min) / (img_max - img_min)
        img2 = (max_target - min_target) * img_normalized + min_target
    
    # 转换为uint8（符合图像存储标准）并返回
    return img2.astype(np.uint8)

# ===================== GUI主类 =====================
class MRIReconGUI:
    """
    MRI K空间图像重建可视化界面类
    核心功能：
    1. FID文件读取与解析
    2. 多切片K空间数据可视化
    3. 二维傅里叶反变换重建图像
    4. 参考图像加载与对比显示
    5. 切片切换交互
    """
    def __init__(self, root):
        """
        初始化GUI界面与全局变量
        参数：
            root (tk.Tk): tkinter主窗口对象
        """
        # 主窗口配置
        self.root = root
        self.root.title("MRI K空间图像重建工具")
        self.root.geometry("1500x950")  # 固定初始窗口大小
        
        # 字体配置：统一UI控件字体风格
        self.default_font = ("Microsoft YaHei", 12)    # 基础字体
        self.btn_font = ("Microsoft YaHei", 11, "bold")# 按钮字体
        
        # 绑定窗口关闭事件：释放matplotlib资源+退出进程
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 全局状态变量初始化
        self.fid_path = None          # 已加载的FID文件路径
        self.ref_img_path = None      # 已加载的参考图像路径
        self.tau = None               # 时间轴数组（来自FID解析）
        self.data = None              # K空间复数数据（核心数据）
        self.section = None           # 总切片数量
        self.current_section = 1      # 当前选中切片（1起始）
        self.is_reconstructed = False # 重建状态标记：False-未重建，True-已执行傅里叶反变换
        
        # ========== 顶部控制区UI布局 ==========
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # 按钮：选择FID文件
        self.btn_select_fid = tk.Button(
            self.control_frame, 
            text="选择FID文件", 
            command=self.select_fid_file,
            font=self.btn_font, 
            padx=10, 
            pady=5
        )
        self.btn_select_fid.pack(side=tk.LEFT, padx=10)
        
        # 按钮：执行MRI重建（傅里叶反变换）
        self.btn_recon = tk.Button(
            self.control_frame, 
            text="执行MRI重建", 
            command=self.run_recon, 
            state=tk.DISABLED,  # 初始禁用，选完FID文件后启用
            font=self.btn_font, 
            padx=10, 
            pady=5
        )
        self.btn_recon.pack(side=tk.LEFT, padx=10)
        
        # 按钮：选择参考图像
        self.btn_select_ref = tk.Button(
            self.control_frame, 
            text="选择参考图像", 
            command=self.select_ref_img,
            font=self.btn_font, 
            padx=10, 
            pady=5
        )
        self.btn_select_ref.pack(side=tk.LEFT, padx=10)
        
        # 下拉框：切片选择（初始仅显示切片1，加载FID后更新）
        self.section_var = tk.StringVar(value="切片 1")
        self.section_menu = tk.OptionMenu(self.control_frame, self.section_var, "切片 1")
        self.section_menu.config(
            font=self.btn_font, 
            width=10, 
            indicatoron=True,
            padx=10,
            pady=10
        )
        self.section_menu.pack(side=tk.LEFT, padx=10, pady=5)
        self.section_menu.config(state=tk.DISABLED)  # 初始禁用
        # 绑定切片切换事件
        self.section_var.trace('w', self.on_section_change)
        
        # ========== 绘图区域UI布局 ==========
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建matplotlib画布：1行3列子图（K空间/重建图像/参考图像）
        # figsize=(15,8)：画布尺寸，dpi=100：分辨率，constrained_layout：自动调整子图间距
        self.fig, (self.ax_k, self.ax_recon, self.ax_ref) = plt.subplots(
            1, 3, figsize=(15, 8), dpi=100, constrained_layout=True
        )
        
        # 强制所有子图为1:1正方形（匹配MRI图像原始比例）
        for ax in [self.ax_k, self.ax_recon, self.ax_ref]:
            ax.set_aspect('equal', adjustable='box')
        
        # 设置画布总标题
        self.fig.suptitle("MRI图像重建结果", fontsize=28, fontweight='bold', y=1.05)
        
        # 初始化子图标题与坐标轴
        self.ax_k.set_title("K空间数据", fontsize=20)
        self.ax_recon.set_title("重建图像", fontsize=20)
        self.ax_ref.set_title("参考图像", fontsize=20)
        # 隐藏坐标轴（图像可视化无需显示坐标）
        for ax in [self.ax_k, self.ax_recon, self.ax_ref]:
            ax.axis('off')
        
        # 将matplotlib画布嵌入tkinter窗口
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()  # 初始绘制
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def select_fid_file(self):
        """
        选择FID文件并解析数据：
        1. 弹出文件选择对话框，过滤文件类型
        2. 解析选中的FID文件，加载K空间数据
        3. 启用重建按钮和切片选择下拉框
        4. 初始化显示当前切片的K空间数据
        """
        # 弹出文件选择对话框
        file_path = filedialog.askopenfilename(
            title="选择FID文件",
            filetypes=[("FID文件", "*.*"), ("所有文件", "*.*")]
        )
        
        if file_path:
            # 更新全局FID路径，重置重建状态
            self.fid_path = file_path
            self.is_reconstructed = False
            # 启用重建按钮
            self.btn_recon.config(state=tk.NORMAL)
            
            # 解析FID文件
            tau, data, pp, section = read_fid(self.fid_path)
            # 解析成功则更新全局数据
            if data is not None:
                self.tau = tau
                self.data = data
                self.section = section
                # 更新切片选择下拉框选项
                self.update_section_menu(section)
                # 显示当前切片的K空间数据
                self.on_section_change()

    def update_section_menu(self, section):
        """
        更新切片选择下拉框的选项列表
        参数：
            section (int): 总切片数量（来自FID解析的dim3）
        """
        # 启用下拉框
        self.section_menu.config(state=tk.NORMAL)
        # 获取下拉框菜单对象，清空原有选项
        menu = self.section_menu['menu']
        menu.delete(0, 'end')
        # 新增切片选项（1起始，符合用户习惯）
        for i in range(1, section + 1):
            menu.add_command(
                label=f"切片 {i}", 
                command=lambda v=f"切片 {i}": self.section_var.set(v)
            )
        # 默认选中第1切片
        self.section_var.set(f"切片 {1}")

    def select_ref_img(self):
        """
        选择参考图像文件（支持png/jpg/jpeg/bmp格式）
        选中后立即调用显示函数更新参考图像子图
        """
        file_path = filedialog.askopenfilename(
            title="选择参考图像",
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.ref_img_path = file_path
            # 显示选中的参考图像
            self.show_ref_img()

    def show_ref_img(self):
        """
        加载并显示参考图像到指定子图
        自动适配图像尺寸，保持1:1比例，隐藏坐标轴
        """
        if self.ref_img_path:
            # 读取参考图像（matplotlib自动处理不同格式）
            img = plt.imread(self.ref_img_path)
            # 清空原有内容，避免重叠
            self.ax_ref.clear()
            # 显示灰度图像
            self.ax_ref.imshow(img, cmap='gray')
            # 重置标题和坐标轴
            self.ax_ref.set_title("参考图像", fontsize=20)
            self.ax_ref.axis('off')
            # 刷新画布
            self.canvas.draw()

    def run_recon(self):
        """
        执行MRI图像重建：
        1. 校验K空间数据是否已加载
        2. 标记重建状态为True
        3. 触发切片切换事件，显示重建后图像
        """
        # 数据校验
        if self.data is None:
            messagebox.showwarning("操作警告", "请先选择并加载FID文件后再执行重建！")
            return
        
        # 标记重建状态
        self.is_reconstructed = True
        # 触发切片切换事件，更新重建图像显示
        self.on_section_change()

    def on_section_change(self, *args):
        """
        切片切换事件处理函数（核心可视化逻辑）
        触发时机：
        1. 加载FID文件后初始化
        2. 用户切换切片下拉框选项
        3. 执行重建操作后
        处理逻辑：
        1. 更新画布总标题（显示当前文件名）
        2. 解析当前选中切片索引
        3. 提取对应切片的K空间数据并可视化
        4. 若已重建，执行傅里叶反变换并显示重建图像
        """
        # 数据未加载时直接返回
        if self.data is None:
            return
        
        # 更新画布总标题：显示当前加载的FID文件名
        file_name = os.path.basename(self.fid_path) if self.fid_path else "未加载FID文件"
        self.fig.suptitle(
            f"MRI图像重建结果 - 当前文件: {file_name}", 
            fontsize=16, 
            fontweight='bold'
        )
        
        # 解析当前选中切片（转换为0起始索引）
        section_str = self.section_var.get()
        try:
            # 从"切片 X"字符串中提取数字，转换为0起始索引
            self.current_section = int(section_str.split()[1]) - 1
        except (ValueError, IndexError):
            # 解析失败时返回，避免程序崩溃
            return

        # 提取当前切片的K空间数据：取第0个通道/回波（D4=0）
        # 原始数据形状(D1,D2,D3,D4) → 切片后(D1,D2)
        raw_slice = self.data[:, :, self.current_section, 0]
        
        # ========== 1. 显示K空间数据（幅度图） ==========
        # 计算K空间数据幅度（复数的模）
        k_data = np.abs(raw_slice)
        # 旋转90度（匹配临床MRI图像显示习惯）
        k_data_rot = np.rot90(k_data)
        # 重采样到256x256（统一显示尺寸）
        k_data_resize = self.resize_img(k_data_rot, (256, 256))
        
        # 更新K空间子图
        self.ax_k.clear()
        self.ax_k.imshow(k_data_resize, cmap='gray')
        self.ax_k.set_title("K空间数据", fontsize=20)
        self.ax_k.axis('off')
        
        # ========== 2. 显示重建图像（若已执行重建） ==========
        if self.is_reconstructed:
            # 二维傅里叶反变换重建流程（标准MRI重建步骤）：
            # step1: fftshift → 将K空间低频分量移到中心
            # step2: ifft2 → 二维逆傅里叶变换
            # step3: ifftshift → 将重建图像中心移回视觉正中
            k_shifted = np.fft.fftshift(raw_slice)
            recon_data = np.fft.ifftshift(np.fft.ifft2(k_shifted))
            
            # 提取重建图像幅度（复数的模）
            img = np.abs(recon_data)
            # 灰度拉伸到0-255（优化可视化效果）
            img_stretch = gray_trans(img, [0, 255])
            # 旋转+水平翻转（匹配临床MRI图像方位）
            img_rot = np.rot90(img_stretch)
            img_flip = np.fliplr(img_rot)
            # 重采样到256x256（统一显示尺寸）
            img_resize = self.resize_img(img_flip, (256, 256))
            
            # 更新重建图像子图
            self.ax_recon.clear()
            self.ax_recon.imshow(img_resize, cmap='gray')
            self.ax_recon.set_title("重建图像", fontsize=20)
            self.ax_recon.axis('off')
        else:
            # 未重建时显示提示文字
            self.ax_recon.clear()
            self.ax_recon.set_title("待重建...", fontsize=20)
            self.ax_recon.axis('off')
        
        # 刷新画布，更新所有子图显示
        self.canvas.draw()

    def resize_img(self, img, target_size):
        """
        图像重采样（对标MATLAB imresize）
        使用scipy zoom实现任意尺寸缩放，保持图像比例
        
        参数：
            img (np.ndarray): 输入图像数组（2D）
            target_size (tuple): 目标尺寸，格式(height, width)
        返回值：
            resized_img (np.ndarray): 重采样后的图像
        """
        from scipy.ndimage import zoom
        
        # 获取输入图像尺寸
        h, w = img.shape
        # 获取目标尺寸
        target_h, target_w = target_size
        # 计算缩放因子
        zoom_h = target_h / h
        zoom_w = target_w / w
        # 执行重采样
        return zoom(img, (zoom_h, zoom_w))

    def on_closing(self):
        """
        窗口关闭回调函数：
        1. 关闭所有matplotlib画布，释放资源
        2. 停止tkinter主循环
        3. 销毁窗口对象
        4. 强制退出进程，避免资源泄漏
        """
        plt.close('all')       # 关闭matplotlib所有绘图资源
        self.root.quit()       # 终止tkinter主循环
        self.root.destroy()    # 销毁主窗口
        sys.exit(0)            # 退出Python进程

# ===================== 程序入口 =====================
if __name__ == "__main__":
    # 创建tkinter主窗口
    root = tk.Tk()
    # 初始化GUI应用
    app = MRIReconGUI(root)
    # 启动主循环
    root.mainloop()