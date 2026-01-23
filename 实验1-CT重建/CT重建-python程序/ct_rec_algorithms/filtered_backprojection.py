# filtered_backprojection.py - 滤波反投影算法
import numpy as np

def filtered_backprojection(raw_data):
    """滤波反投影算法实现"""
    # 替换为你的真实算法代码，以下是模拟逻辑
    reconstructed = np.rot90(raw_data)
    return reconstructed

if __name__ == "__main__":
    test_data = np.random.rand(256, 256)
    res = filtered_backprojection(test_data)
    print(f"滤波反投影测试完成，结果尺寸：{res.shape}")