# backprojection_filter.py - 反投影滤波算法
import numpy as np

def backprojection_filter(raw_data):
    """反投影滤波算法实现"""
    # 替换为你的真实算法代码，以下是模拟逻辑
    reconstructed = np.fliplr(raw_data)
    return reconstructed

if __name__ == "__main__":
    test_data = np.random.rand(256, 256)
    res = backprojection_filter(test_data)
    print(f"反投影滤波测试完成，结果尺寸：{res.shape}")