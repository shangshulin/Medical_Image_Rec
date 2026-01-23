import numpy as np
from scipy.interpolate import griddata


def fourier_backprojection(sinogram, angles):
    """
    傅里叶反投影重建算法（Fourier Backprojection）
    核心步骤：
    1. 对每个角度的投影做1D FFT + fftshift
    2. 构建频域极坐标点 (fx, fy) 和对应的频域值 F
    3. 笛卡尔网格插值
    4. ifftshift + 2D IFFT 得到重建图像

    参数:
        sinogram: np.ndarray, 投影数据（正弦图），shape=(num_angles, num_detectors)
        angles: np.ndarray, 投影角度（弧度），shape=(num_angles,)

    返回:
        recon_image: np.ndarray, 重建后的图像，shape=(num_detectors, num_detectors)
    """
    # ========== 步骤0：参数初始化与预处理 ==========
    num_angles = len(angles)  # 投影角度数
    num_detectors = sinogram.shape[1]  # 探测器数量（图像尺寸）
    recon_size = num_detectors  # 重建图像尺寸（与探测器数一致）

    # 归一化探测器坐标（频域X轴，对应投影方向）
    # 探测器位置范围：[-num_detectors/2, num_detectors/2]
    detector_coords = np.linspace(-num_detectors / 2, num_detectors / 2, num_detectors, dtype=np.float32)

    # ========== 步骤1：对每个角度的投影做1D FFT + fftshift ==========
    # 初始化频域数据存储
    fft_projections = np.zeros((num_angles, num_detectors), dtype=np.complex64)

    for i in range(num_angles):
        # 单个角度的投影数据
        proj = sinogram[i].astype(np.float32)

        # 1D FFT（沿探测器方向）
        fft_proj = np.fft.fft(proj)

        # fftshift：将零频率分量移到中心
        fft_proj_shifted = np.fft.fftshift(fft_proj)

        # 存储频域结果
        fft_projections[i] = fft_proj_shifted

    # ========== 步骤2：构建频域极坐标点 (fx, fy) 和对应的频域值 F ==========
    # 极坐标参数：
    # - 径向频率 r (对应探测器频域坐标)
    # - 角度 theta (投影角度 + 90度，因为投影是沿垂直于角度方向)
    r = detector_coords  # 径向频率（与探测器坐标一致）
    theta = angles + np.pi / 2  # 投影角度旋转90度（投影方向与角度垂直）

    # 生成极坐标网格 (r_grid, theta_grid)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing='xy')

    # 极坐标转笛卡尔坐标 (fx, fy)
    fx = r_grid * np.cos(theta_grid)
    fy = r_grid * np.sin(theta_grid)

    # 展平用于插值
    fx_flat = fx.flatten()
    fy_flat = fy.flatten()
    F_flat = fft_projections.flatten()  # 频域值

    # ========== 步骤3：创建规则笛卡尔网格并插值 ==========
    # 定义笛卡尔网格范围（与重建图像尺寸匹配）
    # 网格中心点为(0,0)，范围覆盖所有极坐标点
    cartesian_coords = np.linspace(-num_detectors / 2, num_detectors / 2, recon_size, dtype=np.float32)
    fx_cart, fy_cart = np.meshgrid(cartesian_coords, cartesian_coords, indexing='xy')

    # 插值：将极坐标频域值映射到笛卡尔网格
    # 采用线性插值，超出范围的点设为0（避免NaN）
    F_cart = griddata(
        points=np.column_stack((fx_flat, fy_flat)),
        values=F_flat,
        xi=(fx_cart, fy_cart),
        method='linear',
        fill_value=0.0
    ).astype(np.complex64)

    # ========== 步骤4：ifftshift + 2D IFFT 得到重建图像 ==========
    # ifftshift：将零频率分量移回角落（与fftshift逆操作）
    F_cart_unshifted = np.fft.ifftshift(F_cart)

    # 2D IFFT：频域转空域
    recon_image = np.fft.ifft2(F_cart_unshifted)

    # 取实部（虚部是数值误差，应忽略）
    recon_image = np.real(recon_image)

    # ========== 后处理：数值归一化与调整 ==========
    # 移除均值偏移（背景归零）
    recon_image = recon_image - np.min(recon_image)
    # 归一化到0~255范围（与主程序显示标准对齐）
    if np.max(recon_image) > 1e-6:
        recon_image = recon_image / np.max(recon_image) * 255.0
    # 转换为float32（匹配主程序数据类型）
    recon_image = recon_image.astype(np.float32)

    return recon_image