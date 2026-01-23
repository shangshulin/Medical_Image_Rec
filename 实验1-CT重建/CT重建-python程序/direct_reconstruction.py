import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from scipy import ndimage
from scipy.signal import convolve

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
    P = (P - P_min) / (P_max - P_min)
    return P

def irandon_transform(image, steps):
    # 定义用于存储重建后的图像的数组
    channels = len(image[0])
    origin = np.zeros((steps, channels, steps))
    for i in range(steps):
        projectionValue = image[:, i]
        projectionValueExpandDim = np.expand_dims(projectionValue, axis=0)
        projectionValueRepeat = projectionValueExpandDim.repeat(channels, axis=0)
        origin[i] = ndimage.rotate(projectionValueRepeat, i * 180 / steps, reshape=False).astype(np.float64)
    iradon = np.sum(origin, axis=0)
    return iradon
if __name__ == "__main__":
    configs = {
        "N": 180, # 图像大小
        "N_d": 180,  # 投影器数量
        "direct_reconstruction_result_path": "./image_result/direct_reconstruction_result/direct_reconstruction_result.png"
    }

    theta = np.arange(0, 180, 1)
    I = shepp_logan(configs["N"])  # Replace with your phantom generation code

    # N_d = 2 * np.ceil(np.linalg.norm(np.array(I.shape) - np.floor((np.array(I.shape) - 1) / 2) - 1)) - 4
    P = forward_projection(theta, configs["N"], int(configs["N_d"]))

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(I, cmap='gray')
    plt.title('180x180 Head Phantom')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(P, cmap='gray')
    plt.colorbar()
    plt.title('180° Parallel Beam Projection')
    iradon_result = irandon_transform(P, P.shape[0])
    plt.subplot(1, 3, 3)
    plt.imshow(iradon_result, cmap='gray')
    plt.title('BP Reconstruction')
    plt.savefig(configs["direct_reconstruction_result_path"])
    plt.show()