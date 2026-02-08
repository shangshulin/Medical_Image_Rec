% 功能：超声图像重建主工作流，按顺序调用所有模块化脚本，完成全流程重建
% 用法：直接运行该脚本，无需手动调用其他模块，自动完成从.pgm到扇环图像的重建
% 说明：文件名以字母开头（大写M），符合Matlab语法要求，串联所有字母开头的功能模块

clear; clc;
disp('======================================');
disp('[MainUltrasoundReconstruction] 超声图像重建项目启动，开始全流程处理...');
disp('======================================');

% ======================================
% 步骤1：.pgm -> .dat 格式转换（调用字母开头模块）
% ======================================
ConvertPgmToDat();

% ======================================
% 步骤2：RF信号导入与预处理（调用字母开头模块）
% ======================================
[image_data, normalized_scan_line, scan_line, params] = SignalImportAndPreprocess();

% ======================================
% 步骤3：频谱分析与中心频率确定（调用字母开头模块）
% ======================================
[center_freq_full, shifted_signal, t, f_full, params] = SpectrumAnalysisAndCenterFreq(image_data, normalized_scan_line, scan_line, params);

% ======================================
% 步骤4：滤波器设计与批量信号滤波（调用字母开头模块）
% ======================================
[image_data, lp_filter, filtered_signal, params] = FilterDesignAndBatchProcess(image_data, shifted_signal, center_freq_full, t, f_full, params);

% ======================================
% 步骤5：信号抽取与B超矩形成像（调用字母开头模块）
% ======================================
[ultrasound_image_norm, params] = SignalDecimationAndBModeImaging(image_data, lp_filter, filtered_signal, params);

% ======================================
% 步骤6：B超图像重排（调用字母开头模块）
% ======================================
[ultrasound_image_norm, num_samples, num_scans] = ImageRearrangement(ultrasound_image_norm, params);

% ======================================
% 步骤7：凸阵探头扇环图像转换（调用字母开头模块）
% ======================================
SectorImageTransformation(ultrasound_image_norm, num_samples, num_scans, params);
