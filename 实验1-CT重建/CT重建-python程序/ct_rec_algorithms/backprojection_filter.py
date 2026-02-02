# backprojection_filter.py - 反投影滤波算法
import numpy as np

def backprojection_unfiltered(P, angles_rad, image_size):
    """
    未滤波反投影 (BPF第一步)
    :param P: 投影数据, shape=(探测器数, 角度数)
    :param angles_rad: 投影角度(弧度), shape=(角度数,)
    :param image_size: 重建图像尺寸 (N)
    :return: 反投影图像 (N, N)
    """
    N_d, num_angles = P.shape
    
    recon = np.zeros((image_size, image_size))
    center = (image_size - 1) / 2.0
    
    # 预计算网格
    y, x = np.mgrid[:image_size, :image_size]
    x = x - center
    y = y - center
    
    # 反投影累加
    for i in range(num_angles):
        angle = angles_rad[i]
        proj = P[:, i]
        
        # 计算每个像素在探测器上的投影位置 (t)
        # 几何关系：t = x * cos(theta) + y * sin(theta)
        t = x * np.cos(angle) + y * np.sin(angle)
        
        # 转换到数组索引 (假设探测器中心为原点)
        t_idx = t + (N_d - 1) / 2.0
        
        # 线性插值
        # 使用 numpy.interp 进行插值 (需要展平处理)
        val = np.interp(t_idx.ravel(), np.arange(N_d), proj, left=0, right=0)
        recon += val.reshape(image_size, image_size)
        
    # 归一化 (根据角度密度调整幅度，经验系数)
    return recon * (np.pi / num_angles)


def frequency_domain_processing(image, filter_type='ram_lak'):
    """
    频域滤波处理 (BPF第二步)
    :param image: 反投影后的图像
    :param filter_type: 滤波器类型 ('ram_lak', 'shepp_logan', 'cosine', 'hamming')
    :return: 滤波后的图像
    """
    rows, cols = image.shape
    
    # 1. 转换到频域，并移位到中心
    F_image = np.fft.fftshift(np.fft.fft2(image))
    
    # 2. 构建规则的频域笛卡尔网格
    u = np.fft.fftfreq(rows)
    v = np.fft.fftfreq(cols)
    u_shifted = np.fft.fftshift(u)
    v_shifted = np.fft.fftshift(v)
    
    U, V = np.meshgrid(u_shifted, v_shifted)
    
    # 计算频率半径 rho (归一化频率 0~0.5)
    rho = np.sqrt(U**2 + V**2)
    
    # 3. 设计2D滤波器
    if filter_type == 'ram_lak':
        H = rho
    elif filter_type == 'shepp_logan':
        # rho=0时 sinc(0)=1, rho*sinc(rho)=0
        H = rho * np.sinc(rho) 
    elif filter_type == 'cosine':
        H = rho * np.cos(np.pi * rho)
    elif filter_type == 'hamming':
        H = rho * (0.54 + 0.46 * np.cos(2 * np.pi * rho))
    else:
        H = rho

    # 4. 应用滤波器
    F_filtered = F_image * H
    
    # 5. 转换回空域
    recon = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))
    
    return recon


def backprojection_filter(sinogram, angles, image_size=None, filter_type='ram_lak'):
    """
    反投影滤波重建 (BPF) 主接口
    
    算法流程：
    1. 反投影 (Backprojection) -> 得到模糊图像
    2. 2D频域滤波 (2D Filtering) -> 恢复清晰图像
    
    :param sinogram: 投影数据。支持两种输入形状：
                     1. (角度数, 探测器数) -> 主程序默认格式
                     2. (探测器数, 角度数) -> 算法内部计算格式
                     函数会自动根据 angles 长度进行适配。
    :param angles: 投影角度 (弧度)
    :param image_size: 重建图像尺寸 (int)。若为None，则默认为探测器数量。
    :param filter_type: 滤波器类型
    :return: 重建图像 (numpy 2D array)
    """
    
    # --- 1. 参数适配 ---
    sinogram = np.array(sinogram, dtype=np.float32)
    angles = np.array(angles, dtype=np.float32)
    
    # 确保 P 的形状为 (探测器数, 角度数) 以匹配 backprojection_unfiltered 逻辑
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

    # --- 2. 执行反投影 (Step 1) ---
    # 这一步得到的是未滤波的反投影图像 (模糊)
    bp_image = backprojection_unfiltered(P, angles, image_size)
    
    # --- 3. 执行频域滤波 (Step 2) ---
    # 对反投影图像进行二维滤波
    recon = frequency_domain_processing(bp_image, filter_type)
    
    # --- 4. 后处理 ---
    # 去除负值 (CT值通常非负)
    recon = np.maximum(0, recon)
    
    return recon