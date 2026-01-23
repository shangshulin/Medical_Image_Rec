import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from scipy import ndimage
from scipy.signal import convolve


def forward_projection(theta_proj, N, N_d):
    """
    获取探测器数据（正弦图）
    :param theta_proj: 保存所有投影角度的数组
    :param N: 图像尺寸
    :param N_d: 投影器数量
    :return: P: 投影结果，大小为 (投影器数量, 投影角度数量)
    """
    # 10 个椭圆
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
    rho = shep[:, 0] # 椭圆的密度（衰减系数）
    ae = 0.5 * N * shep[:, 1] # 椭圆 x 轴半轴长度
    be = 0.5 * N * shep[:, 2] # 椭圆 y 轴半轴长度
    xe = 0.5 * N * shep[:, 3] # 椭圆中心 x 坐标
    ye = 0.5 * N * shep[:, 4] # 椭圆中心 y 坐标
    alpha = shep[:, 5] # 椭圆旋转角度
    # 角度转弧度
    alpha = alpha * np.pi / 180
    theta_proj = theta_proj * np.pi / 180
    # 探测器采样坐标：N_d个采样点，中心在0，左右对称
    TT = np.arange(-(N_d - 1) / 2, (N_d - 1) / 2 + 1)

    for k1 in range(theta_num):
        P_theta = np.zeros(int(N_d)) # 存储当前角度下，所有探测器通道的投影值
        for k2 in range(len(xe)): # 遍历所有椭圆
            # 计算椭圆在投影角度θ下的等效半轴平方和
            a = (ae[k2] * np.cos(theta_proj[k1] - alpha[k2])) ** 2 + (be[k2] * np.sin(theta_proj[k1] - alpha[k2])) ** 2
            # 计算射线与椭圆的相交条件
            temp = a - (TT - xe[k2] * np.cos(theta_proj[k1]) - ye[k2] * np.sin(theta_proj[k1])) ** 2
            # 找到与当前椭圆相交的射线（temp>0表示有交点）
            ind = temp > 0
            # 计算射线在椭圆内的路径长度，并乘以密度rho，累加到投影值
            P_theta[ind] += rho[k2] * (2 * ae[k2] * be[k2] * np.sqrt(temp[ind])) / a
        # 将当前角度的投影值存入投影矩阵
        P[:, k1] = P_theta
    # 将投影值归一化到 [0,1]
    P_min = np.min(P)
    P_max = np.max(P)
    P = (P - P_min) / (P_max - P_min)
    return P


N = 180
theta = np.arange(0, 180, 1)
I = shepp_logan(N)  # Replace with your phantom generation code

# N_d：每个投影角度（如 0°、1°、2°...179°）下，探测器阵列包含的采样点数量
# N_d = 2 * np.ceil(np.linalg.norm(np.array(I.shape) - np.floor((np.array(I.shape) - 1) / 2) - 1)) - 4
N_d = N
P = forward_projection(theta, N, int(N_d))

plt.figure()
plt.imshow(I, cmap='gray')
plt.title('180x180 Head Phantom')
plt.figure()
plt.imshow(P, cmap='gray')
plt.colorbar()
plt.title('180° Parallel Beam Projection')
# plt.show()

def IRandonTransform00(image, steps):
    # 定义用于存储重建后的图像的数组
    channels = len(image[0])
    origin = np.zeros((steps, channels, steps))
    for i in range(steps):
        # 传入的图像中的每一列都对应于一个角度的投影值
        # 这里用的图像是上篇博文里得到的Radon变换后的图像裁剪后得到的
        projectionValue = image[:, i]
        # 这里利用维度扩展和重复投影值数组来模拟反向均匀回抹过程
        projectionValueExpandDim = np.expand_dims(projectionValue, axis=0)
        projectionValueRepeat = projectionValueExpandDim.repeat(channels, axis=0)
        origin[i] = ndimage.rotate(projectionValueRepeat, i * 180 / steps, reshape=False).astype(np.float64)
    # 各个投影角度的投影值已经都保存在origin数组中，只需要将它们相加即可
    iradon = np.sum(origin, axis=0)
    return iradon


# 读取图片
# image = cv2.imread('straightLine.png', cv2.IMREAD_GRAYSCALE)

iradon00 = IRandonTransform00(P, P.shape[0])
plt.figure()
plt.imshow(iradon00, cmap='gray')
plt.title('BP Reconstruction')


# 两种滤波器的实现
def RLFilter(N, d):
    filterRL = np.zeros((N,))
    for i in range(N):
        filterRL[i] = - 1.0 / np.power((i - N / 2) * np.pi * d, 2.0)
        if np.mod(i - N / 2, 2) == 0:
            filterRL[i] = 0
    filterRL[int(N / 2)] = 1 / (4 * np.power(d, 2.0))
    return filterRL


def SLFilter(N, d):
    filterSL = np.zeros((N,))
    for i in range(N):
        # filterSL[i] = - 2 / (np.power(np.pi, 2.0) * np.power(d, 2.0) * (np.power((4 * (i - N / 2)), 2.0) - 1))
        filterSL[i] = - 2 / (np.pi ** 2.0 * d ** 2.0 * (4 * (i - N / 2) ** 2.0 - 1))
    return filterSL

def IRandonTransform01(image, steps):
    # 定义用于存储重建后的图像的数组
    channels = len(image[0])
    origin = np.zeros((steps, channels, channels))
    # filter = RLFilter(channels, 1)
    filter = RLFilter(channels, 1)
    for i in range(steps):
        projectionValue = image[:, i]
        projectionValueFiltered = convolve(filter, projectionValue, "same")
        projectionValueExpandDim = np.expand_dims(projectionValueFiltered, axis=0)
        projectionValueRepeat = projectionValueExpandDim.repeat(channels, axis=0)
        origin[i] = ndimage.rotate(projectionValueRepeat, i * 180 / steps, reshape=False).astype(np.float64)
    iradon = np.sum(origin, axis=0)
    return iradon


def IRandonTransform02(image, steps):
    # 定义用于存储重建后的图像的数组
    channels = len(image[0])
    origin = np.zeros((steps, channels, channels))
    # filter = RLFilter(channels, 1)
    filter = SLFilter(channels, 1)
    for i in range(steps):
        projectionValue = image[:, i]
        projectionValueFiltered = convolve(filter, projectionValue, "same")
        projectionValueExpandDim = np.expand_dims(projectionValueFiltered, axis=0)
        projectionValueRepeat = projectionValueExpandDim.repeat(channels, axis=0)
        origin[i] = ndimage.rotate(projectionValueRepeat, i * 180 / steps, reshape=False).astype(np.float64)
    iradon = np.sum(origin, axis=0)
    return iradon


# 读取图片
# image = cv2.imread('straightLine.png', cv2.IMREAD_GRAYSCALE)

iradon01 = IRandonTransform01(P, len(P[0]))
iradon02 = IRandonTransform02(P, len(P[0]))
# 绘制原始图像和对应的sinogram图
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(iradon01, cmap='gray')
plt.title('RL FBP Reconstruction')
plt.subplot(1, 2, 2)
plt.imshow(iradon02, cmap='gray')
plt.title('SL FBP Reconstruction')
plt.show()