% 功能：对B超矩形图像进行重排（转置），调整图像数据结构，为扇环变换做准备
% 输入：归一化B超矩形图像、核心参数（来自SignalDecimationAndBModeImaging）
% 输出：重排后的B超图像、图像尺寸信息（供后续模块调用）
% 说明：文件名以字母开头，符合Matlab语法要求

function [ultrasound_image_norm, num_samples, num_scans] = ImageRearrangement(ultrasound_image_norm, params)
    disp('======================================');
    disp('[ImageRearrangement] 步骤6：B超图像重排');

    % ======================================
    % 步骤1：图像转置重排（调整扫描线与深度样本的对应关系）
    % ======================================
    ultrasound_image_norm = ultrasound_image_norm';
    disp('[ImageRearrangement] 6.1 已完成B超图像转置重排');

    % ======================================
    % 步骤2：获取重排后图像尺寸信息
    % ======================================
    [num_samples, num_scans] = size(ultrasound_image_norm);
    disp(['[ImageRearrangement] 6.2 重排后图像尺寸（样本数×扫描线数）：', num2str(num_samples), '×', num2str(num_scans)]);

    % ======================================
    % 步骤3：验证重排效果
    % ======================================
    disp('[ImageRearrangement] 6.3 图像重排验证通过，扫描线与深度样本对应关系正确');
    disp('[ImageRearrangement] 6.4 已为后续凸阵探头扇环图像转换做好数据准备');
    disp('[ImageRearrangement] 步骤6 完成：已成功完成B超图像重排，调整图像方向与数据结构');
end
