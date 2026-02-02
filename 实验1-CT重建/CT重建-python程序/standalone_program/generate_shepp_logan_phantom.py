import numpy as np
import matplotlib.pyplot as plt


def shepp_logan_phantom(size=256):
    # 椭圆参数: [x0, y0, a, b, angle_deg, rho]
    ellipses = [
        [0.0, 0.0, 0.92, 0.69, 90, 2.0],           # 主椭圆（头部）
        [0.0, -0.0184, 0.874, 0.6624, 90, -0.98],  # 颅骨
        [0.22, 0.0, 0.31, 0.11, 72, -0.02],        # 左脑室
        [-0.22, 0.0, 0.41, 0.16, 108, -0.02],      # 右脑室
        [0.0, 0.35, 0.25, 0.21, 90, 0.01],         # 脑干
        [0.0, 0.1, 0.046, 0.046, 0, 0.01],         # 小椭圆
        [0.0, -0.1, 0.046, 0.046, 0, 0.01],        # 小椭圆
        [-0.08, -0.605, 0.046, 0.023, 0, 0.01],    # 眼球
        [0.0, -0.605, 0.023, 0.023, 0, 0.01],      # 眼球
        [0.06, -0.605, 0.046, 0.023, 90, 0.01]     # 眼球
    ]

    # 创建归一化坐标网格 [-1, 1]
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]

    phantom = np.zeros((size, size), dtype=np.float32)

    for item in ellipses:
        x0, y0, a, b, angle_deg, gray_val = item
        angle = np.deg2rad(angle_deg)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # 平移
        x_shift = x - x0
        y_shift = y - y0

        # 逆旋转
        xr = cos_a * x_shift + sin_a * y_shift
        yr = -sin_a * x_shift + cos_a * y_shift

        # 椭圆方程
        ellipse_mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0

        # 设置灰度值
        phantom[ellipse_mask] += gray_val

    return phantom


if __name__ == "__main__":
    configs = {
        "image_size": 256, # 图像大小
        "plot_coordinate_system": False, # 是否绘制坐标系
        "alpha": 1, # 图像透明度
        "image_result_path": './image_result/shepp_logan_image/shepp_logan_phantom.png' # shepp-logan 图像保存路径
    }

    image_size = configs["image_size"]
    phantom = shepp_logan_phantom(image_size)
    plt.figure(figsize=(8, 8))
    plt.imshow(phantom, cmap='gray', extent=[-1, 1, -1, 1], vmin=0.95, vmax=1.25, origin='lower', alpha=configs["alpha"])
    plt.axis(configs["plot_coordinate_system"])
    plt.savefig(configs["image_result_path"], bbox_inches='tight', pad_inches=0)
    plt.show()