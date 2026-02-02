import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from scipy.interpolate import griddata
from scipy.ndimage import rotate


def forward_projection(theta_proj, N, N_d):
    # Shepp-Logan 椭圆参数 (rho, a, b, x0, y0, phi_deg)
    shep = np.array([[1, 0.69, 0.92, 0, 0, 0],
                     [-0.8, 0.6624, 0.8740, 0, -0.0184, 0],
                     [-0.2, 0.1100, 0.3100, 0.22, 0, -18],
                     [-0.2, 0.1600, 0.4100, -0.22, 0, 18],
                     [0.1, 0.2100, 0.2500, 0, 0.35, 0],
                     [0.1, 0.0460, 0.0460, 0, 0.1, 0],
                     [0.1, 0.0460, 0.0460, 0, 0.1, 0],
                     [0.1, 0.0460, 0.0230, -0.08, -0.605, 0],
                     [0.1, 0.0230, 0.0230, 0, -0.606, 0],
                     [0.1, 0.0230, 0.0460, 0.06, -0.605, 0]])

    theta_num = len(theta_proj)
    P = np.zeros((int(N_d), theta_num))
    rho = shep[:, 0]
    ae = 0.5 * N * shep[:, 1]
    be = 0.5 * N * shep[:, 2]
    xe = 0.5 * N * shep[:, 3]
    ye = 0.5 * N * shep[:, 4]
    alpha = shep[:, 5] * np.pi / 180  # 转为弧度
    theta_rad = theta_proj * np.pi / 180

    # 探测器位置（中心对称）
    t = np.arange(-(N_d - 1) / 2, (N_d - 1) / 2 + 1)

    for k1 in range(theta_num):
        P_theta = np.zeros(int(N_d))
        for k2 in range(len(xe)):
            cos_term = np.cos(theta_rad[k1] - alpha[k2])
            sin_term = np.sin(theta_rad[k1] - alpha[k2])
            a_sq = (ae[k2] * cos_term) ** 2 + (be[k2] * sin_term) ** 2
            t0 = xe[k2] * np.cos(theta_rad[k1]) + ye[k2] * np.sin(theta_rad[k1])
            temp = a_sq - (t - t0) ** 2
            valid = temp > 0
            P_theta[valid] += rho[k2] * (2 * ae[k2] * be[k2] * np.sqrt(temp[valid])) / a_sq
        P[:, k1] = P_theta

    # 归一化
    P_min, P_max = P.min(), P.max()
    P = (P - P_min) / (P_max - P_min + 1e-8)
    return P


def fourier_reconstruction(P, angles_deg, image_size):
    """
    傅里叶重建算法
    :param P: 投影结果，大小为 (投影器数量, 投影角度数量)
    :param angles_deg:
    :param image_size: 图像大小
    :return:
    """
    N_d, num_angles = P.shape
    # 角度转弧度
    angles_rad = np.deg2rad(angles_deg)

    # Step 1: 对每个角度做 1D FFT，并 fftshift 到中心
    F_proj = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(P, axes=0), axis=0), axes=0)

    # Step 2: 构建频域极坐标点 (fx, fy) 和对应的 F 值
    freqs = np.fft.fftfreq(N_d)  # 归一化频率：[-0.5, 0.5)
    freqs_shifted = np.fft.fftshift(freqs)  # 移动后：[-0.5, 0.5)，中心为0

    fx_list = []
    fy_list = []
    F_list = []

    for i in range(num_angles):
        theta = angles_rad[i]
        fx = freqs_shifted * np.cos(theta)
        fy = freqs_shifted * np.sin(theta)
        F_vals = F_proj[:, i]

        fx_list.append(fx)
        fy_list.append(fy)
        F_list.append(F_vals)

    fx_all = np.concatenate(fx_list)
    fy_all = np.concatenate(fy_list)
    F_all = np.concatenate(F_list)

    # Step 3: 创建规则的笛卡尔网格用于插值
    # 注意：我们希望网格覆盖 [-0.5, 0.5) × [-0.5, 0.5)，对应于归一化频率
    x = np.linspace(-0.5, 0.5, image_size, endpoint=False)
    y = np.linspace(-0.5, 0.5, image_size, endpoint=False)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((fx_all, fy_all))
    values = F_all

    # 插值到二维网格（使用线性插值）
    F_2d = griddata(points, values, (X, Y), method='linear', fill_value=0)

    # Step 4: 2D IFFT 得到重建图像
    # 注意：IFFT 需要先 ifftshift，因为我们的 F_2d 是 fftshifted 的
    f_recon = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F_2d)))
    image_recon = np.real(f_recon)

    # Step 5: 裁剪并调整大小（可选）
    # center = image_size // 2
    # half_N = min(center, N_d // 2)
    # cropped = image_recon[center - half_N:center + half_N,
    #           center - half_N:center + half_N]
    #
    # pad = (image_size - cropped.shape[0]) // 2
    # if pad > 0:
    #     image_recon = np.pad(cropped, ((pad, pad), (pad, pad)), mode='constant')
    # else:
    #     image_recon = cropped
    return image_recon


if __name__ == "__main__":
    configs = {
        "N": 180, # 图像大小
        "N_d": 180, # 投影器数量
        "fourier_reconstruction_result_path": "./image_result/fourier_reconstruction_result/fourier_reconstruction_result.png"
    }

    theta = np.arange(0, 180, 1)  # 180个角度
    I = shepp_logan(configs["N"])

    # 投影
    P = forward_projection(theta, configs["N"], configs["N_d"])

    # 傅里叶重建
    recon = fourier_reconstruction(P, theta, image_size=configs["N"])

    # 显示结果
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(I, cmap='gray')
    plt.title('Original Phantom')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(P, cmap='gray', aspect='auto')
    plt.title('Sinogram')
    plt.xlabel('Angle')
    plt.ylabel('Detector')

    plt.subplot(1, 3, 3)
    plt.imshow(recon, cmap='gray')
    plt.title('Fourier Reconstruction')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(configs["fourier_reconstruction_result_path"])
    plt.show()
