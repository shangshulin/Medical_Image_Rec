import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan

def forward_projection(theta_proj, N, N_d):
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
    alpha = shep[:, 5]
    alpha = alpha * np.pi / 180
    theta_proj = theta_proj * np.pi / 180
    TT = np.arange(-(N_d - 1) / 2, (N_d - 1) / 2 + 1)

    for k1 in range(theta_num):
        P_theta = np.zeros(int(N_d))
        for k2 in range(len(xe)):
            a = (ae[k2] * np.cos(theta_proj[k1] - alpha[k2])) ** 2 + (be[k2] * np.sin(theta_proj[k1] - alpha[k2])) ** 2
            temp = a - (TT - xe[k2] * np.cos(theta_proj[k1]) - ye[k2] * np.sin(theta_proj[k1])) ** 2
            ind = temp > 0
            P_theta[ind] += rho[k2] * (2 * ae[k2] * be[k2] * np.sqrt(temp[ind])) / a
        P[:, k1] = P_theta

    P_min = np.min(P)
    P_max = np.max(P)
    P = (P - P_min) / (P_max - P_min + 1e-8)
    return P


def backprojection_unfiltered(P, angles_deg, image_size):
    """
    未滤波反投影
    """
    N_d, num_angles = P.shape
    angles_rad = np.deg2rad(angles_deg)
    
    recon = np.zeros((image_size, image_size))
    center = (image_size - 1) / 2.0
    
    # 预计算网格
    y, x = np.mgrid[:image_size, :image_size]
    x = x - center
    y = y - center
    
    # 简单的反投影
    for i in range(num_angles):
        angle = angles_rad[i]
        proj = P[:, i]
        
        # 计算每个像素在探测器上的投影位置
        t = x * np.cos(angle) + y * np.sin(angle)
        
        # 简单的线性插值
        t_idx = t + (N_d - 1) / 2.0
        
        # 使用 numpy.interp 进行插值 (需要展平处理)
        val = np.interp(t_idx.ravel(), np.arange(N_d), proj, left=0, right=0)
        recon += val.reshape(image_size, image_size)
        
    return recon * (np.pi / num_angles)


def frequency_domain_processing(image, filter_type='R-L'):
    """
    中间处理模块：频域处理
    """
    rows, cols = image.shape
    
    # 转换到频域，并移位到中心
    F_image = np.fft.fftshift(np.fft.fft2(image))
    
    # 构建规则的频域笛卡尔网格
    u = np.fft.fftfreq(rows)
    v = np.fft.fftfreq(cols)
    u_shifted = np.fft.fftshift(u)
    v_shifted = np.fft.fftshift(v)
    
    U, V = np.meshgrid(u_shifted, v_shifted)
    
    # 计算滤波器响应并应用
    rho = np.sqrt(U**2 + V**2)
    
    # 设计滤波器
    if filter_type == 'R-L':
        H = rho
    elif filter_type == 'S-L':
        H = rho * np.sinc(rho) 
    elif filter_type == 'cosine':
        H = rho * np.cos(np.pi * rho)
    elif filter_type == 'hamming':
        H = rho * (0.54 + 0.46 * np.cos(2 * np.pi * rho))
    else:
        H = rho

    # 应用滤波器
    F_filtered = F_image * H
    
    # 转换回空域
    recon = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))
    
    return recon


def bpf_reconstruction(P, angles_deg, image_size, filter_type='R-L'):
    """
    反投影滤波重建 (Back Projection Filtering) 主流程
    """
    # 反投影 (Spatial Domain Operation)
    print("Step 1: 执行反投影 (Backprojection)...")
    bp_image = backprojection_unfiltered(P, angles_deg, image_size)
    
    # 频域处理 (Frequency Domain Processing)
    print(f"Step 2: 执行频域处理 (Filter: {filter_type})...")
    recon = frequency_domain_processing(bp_image, filter_type)
    
    # 后处理 (Post-processing)
    recon = np.maximum(0, recon)
    
    return recon


def interactive_display(phantom, sinogram, recon):
    """
    交互式展示函数
    :param phantom: 原始 Phantom 图像
    :param sinogram: 正弦图数据
    :param recon: 重建结果
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    axes[0].imshow(phantom, cmap='gray')
    axes[0].set_title('Original Phantom')
    axes[0].axis('off')

    im_sinogram = axes[1].imshow(sinogram, cmap='gray', aspect='auto')
    axes[1].set_title('Sinogram')
    axes[1].set_xlabel('Angle')
    axes[1].set_ylabel('Detector')

    im_recon = axes[2].imshow(recon, cmap='gray')
    axes[2].set_title('BPF Reconstruction')
    axes[2].axis('off')

    print("交互式界面已启动。")
    plt.show()


if __name__ == "__main__":
    configs = {
        "N": 180,
        "N_d": 180,
        "bpf_reconstruction_result_path": "./image_result/bpf_reconstruction_result/bpf_reconstruction_result.png"
    }

    import os
    os.makedirs(os.path.dirname(configs["bpf_reconstruction_result_path"]), exist_ok=True)

    theta = np.arange(0, 180, 1)
    I = shepp_logan(configs["N"])

    print("Generating projections...")
    P = forward_projection(theta, configs["N"], configs["N_d"])
    
    print("Reconstructing with Ram-Lak filter...")# 计算重建
    recon = bpf_reconstruction(P, theta, configs["N"], filter_type='R-L')

    interactive_display(I, P, recon)
