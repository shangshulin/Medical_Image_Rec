import numpy as np
from scipy.interpolate import griddata


def fourier_backprojection(sinogram, angles_deg, image_size):
    """
    傅里叶反投影重建算法（Fourier Backprojection）
    核心步骤：
    1. 对每个角度的投影做1D FFT，并fftshift到中心
    2. 构建频域极坐标点 (fx, fy) 和对应的频域值 F
    3. 笛卡尔网格插值（线性插值）
    4. ifftshift + 2D IFFT 得到重建图像

    参数:
        sinogram: np.ndarray, 投影数据（正弦图），shape=(探测器数量, 投影角度数量)
        angles_deg: np.ndarray, 投影角度（角度制），shape=(投影角度数量,)
        image_size: int, 重建图像的尺寸（N×N）

    返回:
        recon_image: np.ndarray, 重建后的图像，shape=(image_size, image_size)
    """
    # ========== 步骤0：参数初始化与预处理 ==========
    N_d, num_angles = sinogram.shape  # 探测器数量、投影角度数
    angles_rad = np.deg2rad(angles_deg)  # 角度转弧度

    # ========== 步骤1：对每个角度的投影做1D FFT + fftshift ==========
    # 沿探测器维度（axis=0）做FFT，先ifftshift再FFT最后fftshift（对齐fourier_reconstruction逻辑）
    F_proj = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(sinogram, axes=0), axis=0),
        axes=0
    )

    # ========== 步骤2：构建频域极坐标点 (fx, fy) 和对应的F值 ==========
    # 计算归一化频率（[-0.5, 0.5)，与fourier_reconstruction一致）
    freqs = np.fft.fftfreq(N_d)
    freqs_shifted = np.fft.fftshift(freqs)  # 零频率移到中心

    # 遍历所有角度，生成极坐标对应的笛卡尔频域点
    fx_list, fy_list, F_list = [], [], []
    for i in range(num_angles):
        theta = angles_rad[i]
        # 极坐标转笛卡尔坐标（径向频率×cos/sin(投影角度)）
        fx = freqs_shifted * np.cos(theta)
        fy = freqs_shifted * np.sin(theta)
        F_vals = F_proj[:, i]

        fx_list.append(fx)
        fy_list.append(fy)
        F_list.append(F_vals)

    # 拼接所有频域点和对应值
    fx_all = np.concatenate(fx_list)
    fy_all = np.concatenate(fy_list)
    F_all = np.concatenate(F_list)

    # ========== 步骤3：创建规则笛卡尔网格并插值 ==========
    # 构建与重建图像尺寸匹配的归一化频域网格（[-0.5, 0.5) × [-0.5, 0.5)）
    x = np.linspace(-0.5, 0.5, image_size, endpoint=False)
    y = np.linspace(-0.5, 0.5, image_size, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # 线性插值：极坐标频域点 → 笛卡尔频域网格
    points = np.column_stack((fx_all, fy_all))
    values = F_all
    F_2d = griddata(
        points=points,
        values=values,
        xi=(X, Y),
        method='linear',
        fill_value=0.0
    )

    # ========== 步骤4：ifftshift + 2D IFFT 得到重建图像 ==========
    # 逆移位后做2D IFFT，取实部（虚部为数值误差）
    f_recon = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(F_2d))
    )
    recon_image = np.real(f_recon)

    # ========== 后处理：归一化（对齐fourier_reconstruction的归一化逻辑） ==========
    recon_min, recon_max = recon_image.min(), recon_image.max()
    recon_image = (recon_image - recon_min) / (recon_max - recon_min + 1e-8)  # 加小值避免除零

    return recon_image


# 可选：添加测试主函数（对齐fourier_reconstruction的测试逻辑）
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from phantominator import shepp_logan
    from fourier_reconstruction import forward_projection

    # 配置参数
    configs = {
        "N": 180,          # 图像大小
        "N_d": 180,        # 探测器数量
        "result_path": "./image_result"
    }

    # 生成投影数据
    theta = np.arange(0, 180, 1)  # 0~179度，步长1度
    P = forward_projection(theta, configs["N"], configs["N_d"])

    # 傅里叶反投影重建
    recon = fourier_backprojection(P, theta, image_size=configs["N"])

    # 显示结果
    plt.figure(figsize=(15, 4))
    # 原始Shepp-Logan幻影
    I = shepp_logan(configs["N"])
    plt.subplot(1, 3, 1)
    plt.imshow(I, cmap='gray')
    plt.title('Original Phantom')
    plt.axis('off')
    # 正弦图
    plt.subplot(1, 3, 2)
    plt.imshow(P, cmap='gray', aspect='auto')
    plt.title('Sinogram')
    plt.xlabel('Angle')
    plt.ylabel('Detector')
    # 重建结果
    plt.subplot(1, 3, 3)
    plt.imshow(recon, cmap='gray')
    plt.title('Fourier Backprojection')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(configs["result_path"])
    plt.show()