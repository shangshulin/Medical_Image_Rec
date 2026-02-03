# filtered_backprojection.py - 滤波反投影算法
import numpy as np

def ram_lak_filter(size):
    """
    生成 Ram-Lak 滤波器 (Ramp filter)
    """
    n = np.arange(-size // 2, size // 2)
    filter_kernel = np.zeros(size)
    for i, freq in enumerate(n):
        if freq == 0:
            filter_kernel[i] = 0.25
        elif freq % 2 == 1:
            filter_kernel[i] = -1 / (np.pi ** 2 * freq ** 2)
        else:
            filter_kernel[i] = 0
    return filter_kernel


def shepp_logan_filter(size):
    """
    生成 Shepp-Logan 滤波器
    """
    n = np.arange(-size // 2, size // 2)
    filter_kernel = np.zeros(size)
    for i, freq in enumerate(n):
        filter_kernel[i] = -2 / (np.pi ** 2 * (4 * freq ** 2 - 1))
    return filter_kernel


def apply_filter(projection, filter_type='R-L'):
    """
    对单角度投影数据进行滤波 (一维卷积)
    :param projection: 1D array, 单个角度的投影数据
    :param filter_type: 'R-L' or 'S-L'
    :return: 滤波后的投影数据
    """
    size = len(projection)
    if filter_type == 'R-L':
        kernel = ram_lak_filter(size)
    elif filter_type == 'S-L':
        kernel = shepp_logan_filter(size)
    else:
        # 默认使用 Ram-Lak (R-L)
        kernel = ram_lak_filter(size)

    # 使用 'same' 模式保持尺寸一致
    filtered = np.convolve(projection, kernel, mode='same')
    return filtered


def backprojection_rad(projection, angles_rad, image_size):
    """
    执行反投影 (Backprojection)
    :param projection: 滤波后的投影数据, shape=(detectors, angles)
    :param angles_rad: 投影角度(弧度)
    :param image_size: 重建图像尺寸
    :return: 重建图像
    """
    N_d = projection.shape[0]
    num_angles = len(angles_rad)
    
    recon = np.zeros((image_size, image_size))
    center = image_size // 2
    
    # 构建图像坐标网格
    x = np.arange(image_size) - center
    y = np.arange(image_size) - center
    X, Y = np.meshgrid(x, y)

    for i in range(num_angles):
        theta_rad = angles_rad[i]
        
        # 计算每个像素在探测器上的投影位置 t
        # 几何关系: t = x * cos(theta) + y * sin(theta)
        t = X * np.cos(theta_rad) + Y * np.sin(theta_rad)
        
        # 转换为数组索引 (探测器中心为原点)
        t_indices = np.round(t + N_d // 2).astype(int)
        
        # 边界处理 (Clip防止越界)
        t_indices = np.clip(t_indices, 0, N_d - 1)

        # 获取当前角度的投影数据
        proj = projection[:, i]
        
        # 反投影累加
        backproj = proj[t_indices]
        recon += backproj

    # 归一化
    recon *= np.pi / num_angles
    return recon


def filtered_backprojection(sinogram, angles, image_size=None, filter_type='R-L'):
    """
    滤波反投影重建 (FBP) 主接口
    
    算法流程：
    1. 投影数据滤波 (Filtering) -> 1D Filter
    2. 反投影 (Backprojection)
    
    :param sinogram: 投影数据。支持两种输入形状：
                     1. (角度数, 探测器数) -> 主程序默认格式
                     2. (探测器数, 角度数) -> 算法内部计算格式
                     函数会自动根据 angles 长度进行适配。
    :param angles: 投影角度 (弧度)
    :param image_size: 重建图像尺寸 (int)。若为None，则默认为探测器数量。
    :param filter_type: 滤波器类型 ('ram_lak' 或 'shepp_logan')
    :return: 重建图像 (numpy 2D array)
    """
    
    # --- 1. 参数适配 ---
    sinogram = np.array(sinogram, dtype=np.float32)
    angles = np.array(angles, dtype=np.float32)
    
    # 确保 P 的形状为 (探测器数, 角度数)
    if sinogram.shape[0] == len(angles):
        # 输入是 (角度数, 探测器数)，需要转置
        P = sinogram.T
        N_d = sinogram.shape[1]
    elif sinogram.shape[1] == len(angles):
        # 输入是 (探测器数, 角度数)，无需转置
        P = sinogram
        N_d = sinogram.shape[0]
    else:
        raise ValueError(f"投影数据维度 {sinogram.shape} 与角度数量 {len(angles)} 不匹配")

    if image_size is None:
        image_size = N_d

    num_angles = len(angles)
    
    # --- 2. 投影数据滤波 (Filtering) ---
    filtered_projections = np.zeros_like(P)
    for i in range(num_angles):
        # 对每个角度的投影数据进行1D滤波
        filtered_projections[:, i] = apply_filter(P[:, i], filter_type)

    # --- 3. 反投影 (Backprojection) ---
    recon = backprojection_rad(filtered_projections, angles, image_size)
    
    return recon