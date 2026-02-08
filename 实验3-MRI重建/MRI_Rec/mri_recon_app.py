import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

# 配置matplotlib后端，避免tkinter冲突
plt.switch_backend('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# FID文件读取函数
def read_fid(file_path):
    SW = 20  # 采样宽度，与MATLAB保持一致
    try:
        fid = open(file_path, 'rb')
    except:
        messagebox.showerror("错误", "无法打开指定的FID文件！")
        return None, None, None, None
    
    # 读取文件头信息（int32类型，小端）
    def read_int32():
        return np.frombuffer(fid.read(4), dtype=np.int32)[0]
    
    FileVersion = read_int32()
    Section1Size = read_int32()
    Section2Size = read_int32()
    Section3Size = read_int32()
    Section4Size = read_int32()
    Section5Size = read_int32()
    
    Position = Section1Size + Section2Size + Section3Size + Section4Size
    # 跳过偏移字节
    fid.read(Position)
    
    # 读取维度信息
    Dimension1 = read_int32()
    Dimension2 = read_int32()
    Dimension3 = read_int32()
    Dimension4 = read_int32()
    
    # 初始化实部和虚部数组
    DataReal = np.zeros((Dimension1, Dimension2, Dimension3, Dimension4), dtype=np.float32)
    DataImaginary = np.zeros((Dimension1, Dimension2, Dimension3, Dimension4), dtype=np.float32)
    
    # 读取复数数据（float32，实部+虚部）
    for l in range(Dimension4):
        for k in range(Dimension3):
            for j in range(Dimension2):
                # 读取一行数据：2*Dimension1个float32
                data = np.frombuffer(fid.read(2 * Dimension1 * 4), dtype=np.float32).reshape(-1, 2)
                DataReal[:, j, k, l] = data[:, 0]
                DataImaginary[:, j, k, l] = data[:, 1]
    
    fid.close()
    
    section = Dimension3
    tau = np.arange(Dimension1)[:, np.newaxis] / SW  # 时间轴
    Data = DataReal + 1j * DataImaginary  # 复数K空间数据
    pp = 0
    
    return tau, Data, pp, section

# 灰度拉伸函数（对应MATLAB的gray_trans）
def gray_trans(img1, para):
    a, b = para
    if a > b:
        raise ValueError("最小灰度值a不能超过最大灰度值b！")
    if a < 0 or b > 255:
        raise ValueError("灰度值范围需在[0,255]之间！")
    
    # 转换为double并归一化
    img = img1.astype(np.float64)
    min_i = np.min(img)
    max_i = np.max(img)
    
    if max_i - min_i < 1e-8:  # 避免除零
        img2 = np.ones_like(img) * a
    else:
        img2 = (img - min_i) / (max_i - min_i)
        img2 = (b - a) * img2 + a
    
    return img2.astype(np.uint8)

# MRI重建主类
class MRIReconGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MRI K空间图像重建工具")
        self.root.geometry("1500x950")
        
        # 设置全局字体
        self.default_font = ("Microsoft YaHei", 12)
        self.btn_font = ("Microsoft YaHei", 11, "bold")
        
        # 处理窗口关闭协议
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 初始化变量
        self.fid_path = None
        self.ref_img_path = None
        self.tau = None
        self.data = None
        self.section = None
        self.current_section = 1
        self.is_reconstructed = False  # 新增：标记是否已执行重建
        
        # ========== 顶部控制区域 ==========
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # 按钮1：选择FID文件
        self.btn_select_fid = tk.Button(
            self.control_frame, text="选择FID文件", command=self.select_fid_file,
            font=self.btn_font, padx=10, pady=5
        )
        self.btn_select_fid.pack(side=tk.LEFT, padx=10)
        
        # 按钮2：执行重建
        self.btn_recon = tk.Button(
            self.control_frame, text="执行MRI重建", command=self.run_recon, 
            state=tk.DISABLED, font=self.btn_font, padx=10, pady=5
        )
        self.btn_recon.pack(side=tk.LEFT, padx=10)
        
        # 按钮3：选择参考图像
        self.btn_select_ref = tk.Button(
            self.control_frame, text="选择参考图像", command=self.select_ref_img,
            font=self.btn_font, padx=10, pady=5
        )
        self.btn_select_ref.pack(side=tk.LEFT, padx=10)
        
        # 下拉框：选择切片
        self.section_var = tk.StringVar(value="切片 1")
        self.section_menu = tk.OptionMenu(self.control_frame, self.section_var, "切片 1")
        self.section_menu.config(
            font=self.btn_font, 
            width=10, 
            indicatoron=True,
            padx=10,
            pady=10  # 略小于按钮的pady以平衡视觉高度
        )
        self.section_menu.pack(side=tk.LEFT, padx=10, pady=5)
        self.section_menu.config(state=tk.DISABLED)
        self.section_var.trace('w', self.on_section_change)
        
        # ========== 绘图区域 ==========
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建matplotlib画布
        # 1. 设置DPI为100，这样 figsize=(15, 8) 对应 1500x800 像素
        # 2. 图像核心区域将更接近 256x256 的原始比例
        self.fig, (self.ax_k, self.ax_recon, self.ax_ref) = plt.subplots(
            1, 3, figsize=(15, 8), dpi=100, constrained_layout=True
        )
        # 强制设置子图比例为1:1（正方形）
        for ax in [self.ax_k, self.ax_recon, self.ax_ref]:
            ax.set_aspect('equal', adjustable='box')
            
        # 设置标题字号为28，并通过y参数上移位置以增加段后间距
        self.fig.suptitle("MRI图像重建结果", fontsize=28, fontweight='bold', y=1.05)
        
        # 初始化子图标题
        self.ax_k.set_title("K空间数据", fontsize=20)
        self.ax_recon.set_title("重建图像", fontsize=20)
        self.ax_ref.set_title("参考图像", fontsize=20)
        
        # 禁用坐标轴
        for ax in [self.ax_k, self.ax_recon, self.ax_ref]:
            ax.axis('off')
        
        # 嵌入tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # 选择FID文件
    def select_fid_file(self):
        file_path = filedialog.askopenfilename(
            title="选择FID文件",
            filetypes=[("FID文件", "*.*"), ("所有文件", "*.*")]
        )
        if file_path:
            self.fid_path = file_path
            self.btn_recon.config(state=tk.NORMAL)
            self.is_reconstructed = False  # 重置重建状态
            
            # 读取FID数据
            tau, data, pp, section = read_fid(self.fid_path)
            if data is not None:
                self.tau = tau
                self.data = data
                self.section = section
                
                # 更新切片下拉框
                self.update_section_menu(section)
                # 仅显示K空间
                self.on_section_change()
    
    def update_section_menu(self, section):
        self.section_menu.config(state=tk.NORMAL)
        menu = self.section_menu['menu']
        menu.delete(0, 'end')
        for i in range(1, section+1):
            menu.add_command(label=f"切片 {i}", command=lambda v=f"切片 {i}": self.section_var.set(v))
        self.section_var.set(f"切片 {1}")
    
    # 选择参考图像
    def select_ref_img(self):
        file_path = filedialog.askopenfilename(
            title="选择参考图像",
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp"), ("所有文件", "*.*")]
        )
        if file_path:
            self.ref_img_path = file_path
            # 显示参考图像
            self.show_ref_img()
    
    # 显示参考图像
    def show_ref_img(self):
        if self.ref_img_path:
            img = plt.imread(self.ref_img_path)
            self.ax_ref.clear()
            self.ax_ref.imshow(img, cmap='gray')
            self.ax_ref.set_title("参考图像", fontsize=20)
            self.ax_ref.axis('off')
            self.canvas.draw()
    
    # 执行MRI重建
    def run_recon(self):
        if self.data is None:
            messagebox.showwarning("警告", "请先选择FID文件！")
            return
        
        self.is_reconstructed = True
        self.on_section_change()
    
    # 切换切片
    def on_section_change(self, *args):
        if self.data is None:
            return
        
        # 更新总标题，包含文件名
        file_name = os.path.basename(self.fid_path) if self.fid_path else "未加载"
        self.fig.suptitle(f"MRI图像重建结果 - 当前文件: {file_name}", fontsize=16, fontweight='bold')
        
        # 获取当前切片索引
        section_str = self.section_var.get()
        try:
            self.current_section = int(section_str.split()[1]) - 1  # 转0索引
        except:
            return

        # 获取当前切片数据并去除多余维度 (D1, D2, D3, D4) -> (D1, D2)
        # 假设我们取第1个Echo/Channel (index 0)
        raw_slice = self.data[:, :, self.current_section, 0]
        
        # 1. 显示K空间数据（幅度，旋转90度）
        k_data = np.abs(raw_slice)
        k_data_rot = np.rot90(k_data)
        # 对K空间数据也进行重采样到256x256，确保显示比例一致
        k_data_resize = self.resize_img(k_data_rot, (256, 256))
        
        self.ax_k.clear()
        self.ax_k.imshow(k_data_resize, cmap='gray')
        self.ax_k.set_title("K空间数据", fontsize=20)
        self.ax_k.axis('off')
        
        # 2. 如果已点击重建，则显示重建图像
        if self.is_reconstructed:
            # 正确的二维傅里叶反变换流程：
            # 1. 对原始K空间数据进行 fftshift（确保低频在中心）
            # 2. 执行 ifft2
            # 3. 对结果执行 ifftshift（将图像中心移回正中）
            k_shifted = np.fft.fftshift(raw_slice)
            recon_data = np.fft.ifftshift(np.fft.ifft2(k_shifted))
            
            img = np.abs(recon_data)
            # 灰度拉伸（0-255）
            img_stretch = gray_trans(img, [0, 255])
            # 旋转、翻转、重采样
            img_rot = np.rot90(img_stretch)
            img_flip = np.fliplr(img_rot)
            # 重采样到256x256（使用numpy插值）
            img_resize = self.resize_img(img_flip, (256, 256))
            
            self.ax_recon.clear()
            self.ax_recon.imshow(img_resize, cmap='gray')
            self.ax_recon.set_title("重建图像", fontsize=20)
            self.ax_recon.axis('off')
        else:
            self.ax_recon.clear()
            self.ax_recon.set_title("待重建...", fontsize=20)
            self.ax_recon.axis('off')
        
        # 刷新画布
        self.canvas.draw()
    
    # 图像重采样（替代MATLAB的imresize）
    def resize_img(self, img, target_size):
        from scipy.ndimage import zoom
        h, w = img.shape
        target_h, target_w = target_size
        zoom_h = target_h / h
        zoom_w = target_w / w
        return zoom(img, (zoom_h, zoom_w))

    # 窗口关闭回调
    def on_closing(self):
        plt.close('all')  # 关闭所有Matplotlib资源
        self.root.quit()   # 停止主循环
        self.root.destroy() # 销毁窗口
        sys.exit(0)        # 强制退出进程

# 主函数
if __name__ == "__main__":
    root = tk.Tk()
    app = MRIReconGUI(root)
    root.mainloop()