import numpy as np
import warnings

warnings.filterwarnings('ignore')

# -------------------------- 算法默认参数 --------------------------
DEFAULT_RECON_PARAMS = {
    "interpolation": "linear"  # 仅保留插值方式
}


# -------------------------- 核心工具函数 --------------------------
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
    CT纯直接反投影重建算法（无滤波）
    :param sinogram: 投影数据，shape=(num_angles, num_detectors)（角度数×探测器数）
    :param angles: 投影角度，shape=(num_angles,)（弧度）
    :param params: 重建参数字典，覆盖默认参数
    :return: 重建图像，shape=(img_size, img_size)
    """
    # 1. 参数合并与校验
    recon_params = DEFAULT_RECON_PARAMS.copy()
    if params and isinstance(params, dict):
        # 过滤掉可能传入的滤波相关参数，避免报错
        valid_params = {k: v for k, v in params.items() if k in recon_params.keys()}
        recon_params.update(valid_params)

    # 基础维度校验（保留原逻辑）
    if len(sinogram.shape) != 2:
        raise ValueError(f"投影数据维度错误！应为2维(角度数×探测器数)，当前：{sinogram.shape}")
    if len(angles.shape) != 1 or len(angles) != sinogram.shape[0]:
        raise ValueError(f"投影角度数量与投影数据行数不匹配！角度数：{len(angles)}，投影数据行数：{sinogram.shape[0]}")

    num_angles, num_detectors = sinogram.shape
    img_size = num_detectors  # 重建图像尺寸与探测器数量一致

    # 2. 移除「投影数据滤波」步骤（直接使用原始投影数据）
    raw_sinogram = sinogram.astype(np.float32)

    # 3. 初始化重建图像
    recon_img = np.zeros((img_size, img_size), dtype=np.float32)
    # 生成图像坐标网格（中心为原点）
    x_grid, y_grid = np.meshgrid(
        np.arange(img_size) - img_size / 2,
        np.arange(img_size) - img_size / 2,
        indexing="xy"
    )

    # 4. 逐角度反投影（核心逻辑不变，仅使用原始投影数据）
    for angle_idx, angle in enumerate(angles):
        # 4.1 计算当前角度的投影方向
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        # 计算每个像素在当前投影角度下的探测器坐标
        proj_coords = x_grid * cos_theta + y_grid * sin_theta

        # 4.2 插值获取投影值（匹配当前像素）
        interpolated_vals = _interpolate_projection(
            proj_coords,
            raw_sinogram[angle_idx],  # 使用原始投影数据（无滤波）
            method=recon_params["interpolation"]
        )

        # 4.3 反投影累加
        recon_img += interpolated_vals

    # 5. 归一化（保留原逻辑，消除角度数和弧度影响+适配显示）
    recon_img = recon_img * (np.pi / (2 * num_angles))
    # 灰度值归一化到[0, 255]
    recon_img = (recon_img - np.min(recon_img)) / (np.max(recon_img) - np.min(recon_img) + 1e-8) * 255
    recon_img = recon_img.astype(np.uint8)

    return recon_img


# -------------------------- 主程序调用接口（保持兼容） --------------------------
def run_direct_reconstruction(sinogram, angles, params=None):
    """
    主程序调用的统一接口（固定函数名+返回格式）
    :param sinogram: 投影数据（numpy数组）
    :param angles: 投影角度（numpy数组，弧度）
    :param params: 重建参数字典（可选，仅插值参数有效）
    :return: 重建图像, 执行状态(str: success/failed), 错误信息(str)
    """
    try:
        # 调用核心重建函数
        recon_result = direct_backprojection_recon(sinogram, angles, params)
        return recon_result, "success", ""
    except Exception as e:
        # 捕获所有异常，返回错误信息
        return None, "failed", str(e)


# -------------------------- 测试代码（验证纯反投影重建） --------------------------
if __name__ == "__main__":
    # 模拟主程序传递的测试数据
    test_num_angles = 180
    test_num_detectors = 256
    # 生成测试投影数据
    test_sinogram = np.random.rand(test_num_angles, test_num_detectors).astype(np.float32) * 100
    # 生成测试角度（0~π弧度）
    test_angles = np.linspace(0, np.pi, test_num_angles, endpoint=False, dtype=np.float32)
    # 自定义参数（仅插值方式有效，滤波参数会被过滤）
    test_params = {
        "interpolation": "linear"  # 仅保留插值参数
    }

    # 调用接口测试
    recon_img, status, msg = run_direct_reconstruction(test_sinogram, test_angles, test_params)
    if status == "success":
        print(f"纯直接反投影重建成功！重建图像尺寸：{recon_img.shape}")
        print(f"重建图像灰度范围：[{np.min(recon_img)}, {np.max(recon_img)}]")
    else:
        print(f"重建失败：{msg}")