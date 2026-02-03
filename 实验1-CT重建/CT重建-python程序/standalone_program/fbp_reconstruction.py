import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
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

    return P


def ram_lak_filter(size):
    n = np.arange(-size // 2, size // 2)
    filter = np.zeros(size)
    for i, freq in enumerate(n):
        if freq == 0:
            filter[i] = 0.25
        elif freq % 2 == 1:
            filter[i] = -1 / (np.pi ** 2 * freq ** 2)
        else:
            filter[i] = 0
    return filter


def shepp_logan_filter(size):
    n = np.arange(-size // 2, size // 2)
    filter = np.zeros(size)
    for i, freq in enumerate(n):
        filter[i] = -2 / (np.pi ** 2 * (4 * freq ** 2 - 1))
    return filter


def apply_filter(projection, filter_type='R-L'):
    size = len(projection)
    if filter_type == 'R-L':
        kernel = ram_lak_filter(size)
    elif filter_type == 'S-L':
        kernel = shepp_logan_filter(size)
    else:
        kernel = ram_lak_filter(size)

    filtered = np.convolve(projection, kernel, mode='same')
    return filtered


def backprojection(projection, angles, image_size):
    N_d = len(projection)
    recon = np.zeros((image_size, image_size))
    center = image_size // 2
    x = np.arange(image_size) - center
    y = np.arange(image_size) - center
    X, Y = np.meshgrid(x, y)

    dt = 1.0
    for i, theta in enumerate(angles):
        theta_rad = theta * np.pi / 180
        t = X * np.cos(theta_rad) + Y * np.sin(theta_rad)
        t_indices = np.round(t + N_d // 2).astype(int)
        t_indices = np.clip(t_indices, 0, N_d - 1)

        proj = projection[:, i]
        backproj = proj[t_indices]

        recon += backproj

    recon *= np.pi / len(angles)
    return recon


def fbp_reconstruction(P, angles_deg, image_size, filter_type='R-L'):
    angles_rad = np.deg2rad(angles_deg)
    N_d, num_angles = P.shape

    filtered_projections = np.zeros_like(P)
    for i in range(num_angles):
        filtered_projections[:, i] = apply_filter(P[:, i], filter_type)

    recon = backprojection(filtered_projections, angles_deg, image_size)
    return recon, filtered_projections


def interactive_display(phantom, sinogram_dict, recon_dict):
    """
    交互式展示函数
    :param phantom: 原始 Phantom 图像
    :param sinogram_dict: 包含不同类型 Sinogram 的字典 {'label': data}
    :param recon_dict: 包含不同类型重建结果的字典 {'label': data}
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25) # 为按钮留出空间

    # 原始图像
    axes[0].imshow(phantom, cmap='gray')
    axes[0].set_title('Original Phantom')
    axes[0].axis('off')

    # Sinogram (可变)
    keys = list(sinogram_dict.keys())
    current_key = keys[0] # 默认显示第一个

    im_sinogram = axes[1].imshow(sinogram_dict[current_key], cmap='gray', aspect='auto')
    title_sinogram = axes[1].set_title(f'Sinogram ({current_key})')
    axes[1].set_xlabel('Angle')
    axes[1].set_ylabel('Detector')

    # 重建结果 (可变)
    im_recon = axes[2].imshow(recon_dict[current_key], cmap='gray')
    title_recon = axes[2].set_title(f'Reconstruction ({current_key})')
    axes[2].axis('off')

    # 添加交互式按钮 (RadioButtons)
    ax_color = 'lightgoldenrodyellow'
    ax_radio = plt.axes([0.35, 0.05, 0.3, 0.15], facecolor=ax_color)
    radio = RadioButtons(ax_radio, keys)

    def change_view(label):
        # 更新 Sinogram
        im_sinogram.set_data(sinogram_dict[label])
        # 重新归一化显示范围，以便看清不同幅度的滤波数据
        data_s = sinogram_dict[label]
        im_sinogram.set_clim(data_s.min(), data_s.max())
        title_sinogram.set_text(f'Sinogram ({label})')

        # 更新 Reconstruction
        im_recon.set_data(recon_dict[label])
        data_r = recon_dict[label]
        im_recon.set_clim(data_r.min(), data_r.max())
        title_recon.set_text(f'Reconstruction ({label})')

        fig.canvas.draw_idle()

    radio.on_clicked(change_view)
    
    print("交互式界面已启动。请使用下方的按钮切换不同的滤波模式。")
    plt.show()


if __name__ == "__main__":
    configs = {
        "N": 180,
        "N_d": 180,
        "fbp_reconstruction_result_path": "./image_result/fbp_reconstruction_result/fbp_reconstruction_result.png"
    }

    import os
    os.makedirs(os.path.dirname(configs["fbp_reconstruction_result_path"]), exist_ok=True)

    theta = np.arange(0, 180, 1)
    I = shepp_logan(configs["N"])

    print("Generating projections...")
    P_raw = forward_projection(theta, configs["N"], configs["N_d"])

    print("Reconstructing with R-L filter...")
    recon_ram_lak, P_ram_lak = fbp_reconstruction(P_raw, theta, configs["N"], filter_type='R-L')

    print("Reconstructing with S-L filter...")
    recon_shepp_logan, P_shepp_logan = fbp_reconstruction(P_raw, theta, configs["N"], filter_type='S-L')

    # 准备数据字典
    sinogram_data = {
        'Raw Projection': P_raw,
        'R-L Filtered': P_ram_lak,
        'S-L Filtered': P_shepp_logan
    }

    recon_data = {
        'Raw Projection': np.zeros_like(recon_ram_lak), # Raw 对应的重建这里暂无（或者可以是直接反投影），这里简单置零或放一个占位
        'R-L Filtered': recon_ram_lak,
        'S-L Filtered': recon_shepp_logan
    }
    
    # 为了让 'Raw Projection' 也有意义，我们可以计算一个未滤波反投影
    print("Calculating unfiltered backprojection for comparison...")
    recon_unfiltered = backprojection(P_raw, theta, configs["N"])
    recon_data['Raw Projection'] = recon_unfiltered

    # 启动交互式展示
    interactive_display(I, sinogram_data, recon_data)
