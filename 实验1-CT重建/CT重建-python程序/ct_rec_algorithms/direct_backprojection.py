import numpy as np
import warnings

warnings.filterwarnings('ignore')

# -------------------------- 算法默认参数（与主程序配置匹配） --------------------------
DEFAULT_RECON_PARAMS = {
    "filter_type": "shepp-logan",  # 主程序默认配置的滤波类型
    "sigma": 0.8,  # 主程序默认配置的sigma值
    "interpolation": "linear"  # 主程序默认配置的插值方式
}


# -------------------------- 核心工具函数 --------------------------
def _generate_filter_kernel(detector_num, filter_type="shepp-logan", sigma=0.8):
    """
    生成CT重建滤波核（匹配主程序参数）
    :param detector_num: 探测器数量（投影数据列数）
    :param filter_type: 滤波类型：ram-lak/shepp-logan/butterworth
    :param sigma: 滤波核平滑参数
    :return: 滤波核数组，shape=(detector_num,)
    """
    # 生成频率轴
    freq = np.linspace(-0.5, 0.5, detector_num, dtype=np.float32)
    kernel = np.zeros_like(freq)

    # 不同滤波核实现
    if filter_type == "ram-lak":
        # 基础Ram-Lak滤波核（斜坡滤波）
        kernel = np.abs(freq)
    elif filter_type == "shepp-logan":
        # Shepp-Logan滤波核（主程序默认）
        kernel = np.abs(freq) * np.sinc(2 * sigma * freq)
    elif filter_type == "butterworth":
        # 巴特沃斯滤波核
        kernel = np.abs(freq) / np.sqrt(1 + (freq / sigma) ** 4)
    else:
        raise ValueError(f"不支持的滤波类型：{filter_type}，仅支持ram-lak/shepp-logan/butterworth")

    # 归一化滤波核
    kernel = kernel / np.sum(kernel) if np.sum(kernel) != 0 else kernel
    return kernel


def _interpolate_projection(x_coords, projection_data, method="linear"):
    """
    投影数据插值（匹配主程序的插值配置）
    :param x_coords: 待插值的坐标数组
    :param projection_data: 单角度投影数据，shape=(detector_num,)
    :param method: 插值方式：linear/nearest
    :return: 插值后的投影值数组
    """
    detector_num = len(projection_data)
    # 探测器坐标归一化到[-detector_num/2, detector_num/2)
    detector_coords = np.linspace(-detector_num / 2, detector_num / 2, detector_num, endpoint=False)

    if method == "linear":
        # 线性插值（主程序默认）
        interpolated_vals = np.interp(
            x_coords, detector_coords, projection_data, left=0, right=0
        )
    elif method == "nearest":
        # 最近邻插值
        idx = np.clip(
            np.round(x_coords + detector_num / 2).astype(int),
            0, detector_num - 1
        )
        interpolated_vals = projection_data[idx]
    else:
        raise ValueError(f"不支持的插值方式：{method}，仅支持linear/nearest")

    return interpolated_vals


# -------------------------- 核心重建函数 --------------------------
def direct_backprojection_recon(sinogram, angles, params=None):
    """
    CT直接反投影重建核心算法
    :param sinogram: 投影数据，shape=(num_angles, num_detectors)（角度数×探测器数）
    :param angles: 投影角度，shape=(num_angles,)（弧度）
    :param params: 重建参数字典，覆盖默认参数
    :return: 重建图像，shape=(img_size, img_size)
    """
    # 1. 参数合并与校验
    recon_params = DEFAULT_RECON_PARAMS.copy()
    if params and isinstance(params, dict):
        recon_params.update(params)

    # 基础维度校验（匹配主程序的数据格式）
    if len(sinogram.shape) != 2:
        raise ValueError(f"投影数据维度错误！应为2维(角度数×探测器数)，当前：{sinogram.shape}")
    if len(angles.shape) != 1 or len(angles) != sinogram.shape[0]:
        raise ValueError(f"投影角度数量与投影数据行数不匹配！角度数：{len(angles)}，投影数据行数：{sinogram.shape[0]}")

    num_angles, num_detectors = sinogram.shape
    img_size = num_detectors  # 重建图像尺寸与探测器数量一致（主程序默认逻辑）

    # 2. 投影数据滤波（FBP核心步骤）
    filter_kernel = _generate_filter_kernel(num_detectors,
                                            recon_params["filter_type"],
                                            recon_params["sigma"])
    # 对每个角度的投影数据做卷积滤波
    filtered_sinogram = np.apply_along_axis(
        lambda x: np.convolve(x, filter_kernel, mode="same"),
        axis=1,
        arr=sinogram
    ).astype(np.float32)

    # 3. 初始化重建图像
    recon_img = np.zeros((img_size, img_size), dtype=np.float32)
    # 生成图像坐标网格（中心为原点）
    x_grid, y_grid = np.meshgrid(
        np.arange(img_size) - img_size / 2,
        np.arange(img_size) - img_size / 2,
        indexing="xy"
    )

    # 4. 逐角度反投影
    for angle_idx, angle in enumerate(angles):
        # 4.1 计算当前角度的投影方向
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        # 计算每个像素在当前投影角度下的探测器坐标
        proj_coords = x_grid * cos_theta + y_grid * sin_theta

        # 4.2 插值获取投影值（匹配当前像素）
        interpolated_vals = _interpolate_projection(
            proj_coords,
            filtered_sinogram[angle_idx],
            method=recon_params["interpolation"]
        )

        # 4.3 反投影累加
        recon_img += interpolated_vals

    # 5. 归一化（消除角度数和弧度的影响）
    recon_img = recon_img * (np.pi / (2 * num_angles))
    # 灰度值归一化到[0, 255]（适配图像显示）
    recon_img = (recon_img - np.min(recon_img)) / (np.max(recon_img) - np.min(recon_img) + 1e-8) * 255
    recon_img = recon_img.astype(np.uint8)

    return recon_img


# -------------------------- 主程序调用接口（关键：与主程序完全匹配） --------------------------
def run_direct_reconstruction(sinogram, angles, params=None):
    """
    主程序调用的统一接口（固定函数名+返回格式）
    :param sinogram: 投影数据（numpy数组）
    :param angles: 投影角度（numpy数组，弧度）
    :param params: 重建参数字典（可选）
    :return: 重建图像, 执行状态(str: success/failed), 错误信息(str)
    """
    try:
        # 调用核心重建函数
        recon_result = direct_backprojection_recon(sinogram, angles, params)
        return recon_result, "success", ""
    except Exception as e:
        # 捕获所有异常，返回错误信息（主程序可弹窗提示）
        return None, "failed", str(e)


# -------------------------- 测试代码（验证接口可用性） --------------------------
if __name__ == "__main__":
    # 模拟主程序传递的测试数据
    test_num_angles = 180
    test_num_detectors = 256
    # 生成测试投影数据
    test_sinogram = np.random.rand(test_num_angles, test_num_detectors).astype(np.float32) * 100
    # 生成测试角度（0~π弧度，匹配主程序的模拟扫描逻辑）
    test_angles = np.linspace(0, np.pi, test_num_angles, endpoint=False, dtype=np.float32)
    # 自定义参数（与主程序配置一致）
    test_params = {
        "filter_type": "shepp-logan",
        "sigma": 0.8,
        "interpolation": "linear"
    }

    # 调用接口测试
    recon_img, status, msg = run_direct_reconstruction(test_sinogram, test_angles, test_params)
    if status == "success":
        print(f"重建成功！重建图像尺寸：{recon_img.shape}")
        print(f"重建图像灰度范围：[{np.min(recon_img)}, {np.max(recon_img)}]")
    else:
        print(f"重建失败：{msg}")