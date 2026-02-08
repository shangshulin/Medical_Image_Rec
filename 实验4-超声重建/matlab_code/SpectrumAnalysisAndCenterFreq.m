% 功能：对预处理后的RF信号进行FFT频谱分析，确定信号中心频率，完成基带移频
% 输入：二维RF数据矩阵、归一化单条扫描线、核心参数（来自SignalImportAndPreprocess）
% 输出：中心频率、移频后的基带信号、频谱分析图形（供后续模块调用）
% 说明：文件名以字母开头，符合Matlab语法要求

function [center_freq_full, shifted_signal, t, f_full, params] = SpectrumAnalysisAndCenterFreq(image_data, normalized_scan_line, scan_line, params)
    disp('======================================');
    disp('[SpectrumAnalysisAndCenterFreq] 步骤3：观察信号频谱，确定中心频率');

    % ======================================
    % 步骤1：配置图形窗口参数，绘制时域与频域分析图
    % ======================================
    screen_size = get(0, 'ScreenSize');  % 获取屏幕分辨率
    fig1_width = 900;
    fig1_height = 600;
    fig1_left = (screen_size(3) - fig1_width) / 2;
    fig1_bottom = (screen_size(4) - fig1_height) / 2;
    
    % 创建时域+频域分析图形窗口
    figure('Name', '[SpectrumAnalysisAndCenterFreq] 超声扫描线时域+频域分析结果', ...
        'Position', [fig1_left, fig1_bottom, fig1_width, fig1_height], ...
        'WindowStyle', 'normal');
    disp('[SpectrumAnalysisAndCenterFreq] 3.1 已创建时域与频域分析图形窗口');

    % ======================================
    % 步骤2：绘制归一化时域波形
    % ======================================
    subplot(2, 1, 1); 
    plot(normalized_scan_line, 'LineWidth', 1);
    title(['[SpectrumAnalysisAndCenterFreq] 时域显示：第', num2str(params.target_scan_line), '条扫描线（归一化到0-1）']);
    xlabel(['样本点序号（总数量：', num2str(params.num_points_per_line), '）']);
    ylabel('归一化幅度');
    grid on; grid minor;
    ylim([-0.1, 1.1]);
    disp('[SpectrumAnalysisAndCenterFreq] 3.2 已绘制目标扫描线归一化时域波形');

    % ======================================
    % 步骤3：FFT变换，转换到频域并提取正频率部分
    % ======================================
    N = length(scan_line);  % 信号长度（单条扫描线的样本点数量）
    fft_result = fft(scan_line);  % 快速傅里叶变换
    half_N = floor(N/2);  % 取半长（对称频谱，仅分析正频率部分）
    fft_result_half = fft_result(1:half_N);  % 提取正频率部分频谱
    f = (0:half_N-1) * params.Fs / N;  % 计算正频率轴坐标
    magnitude = abs(fft_result_half) / N;  % 计算归一化幅度谱

    % ======================================
    % 步骤4：绘制频谱幅度谱并标记中心频率
    % ======================================
    subplot(2, 1, 2);  
    plot(f, magnitude, 'LineWidth', 1, 'Color', [0, 0.5, 0.8]);
    hold on;
    
    % 寻找正频率部分的中心频率（最大幅度对应频率）
    [max_magnitude, idx] = max(magnitude);
    center_frequency = f(idx);
    
    % 标记中心频率
    plot(center_frequency, max_magnitude, 'ro', 'MarkerSize', 8, 'DisplayName', '中心频率');
    xline(center_frequency, 'r--', 'LineWidth', 1);
    yline(max_magnitude, 'r--', 'LineWidth', 1);
    hold off;
    
    % 频域图标注
    title(['[SpectrumAnalysisAndCenterFreq] 频域显示：第', num2str(params.target_scan_line), '条扫描线幅度谱']);
    xlabel(sprintf('频率（Hz）| （对应MHz：%.2f）', center_frequency/1e6));
    ylabel('归一化幅度');
    grid on; grid minor;
    legend('幅度谱', '中心频率', 'Location', 'best');
    disp('[SpectrumAnalysisAndCenterFreq] 3.3 已绘制频谱幅度谱并标记正频率部分中心频率');

    % ======================================
    % 步骤5：全频谱分析（包含正负频率），验证中心频率
    % ======================================
    f_full = (-N/2:N/2-1) * (params.Fs / N);  % 完整频率轴（包含正负频率）
    t = (0:N-1)/params.Fs;  % 时间轴（与采样频率匹配）
    scan_line_fft = fftshift(fft(scan_line));  % FFT移位，零频率居中
    [~, max_idx_full] = max(abs(scan_line_fft));
    center_freq_full = f_full(max_idx_full);  % 全频谱中心频率（最终用于移频）
    
    % 输出频域分析结果
    disp(['[SpectrumAnalysisAndCenterFreq] 3.4 正频率部分中心频率（MHz）：', num2str(round(center_frequency/1e6, 4))]);
    disp(['[SpectrumAnalysisAndCenterFreq] 3.5 全频谱验证中心频率（MHz）：', num2str(center_freq_full/1e6, '%.2f')]);

    % ======================================
    % 步骤6：绘制信号处理过程对比图（原始→移频→滤波前）
    % ======================================
    fig2_width = 800;
    fig2_height = 900;
    fig2_left = (screen_size(3) - fig2_width) / 2;
    fig2_bottom = (screen_size(4) - fig2_height) / 2 - 50;
    
    figure('Name', '[SpectrumAnalysisAndCenterFreq] 信号处理过程对比', ...
        'Position', [fig2_left, fig2_bottom, fig2_width, fig2_height], ...
        'WindowStyle', 'normal');
    
    % 子图1：原始信号时域
    subplot(4, 1, 1);
    plot(t*1e6, scan_line);
    title('[SpectrumAnalysisAndCenterFreq] 原始信号（时域）');
    xlabel('时间 (μs)');
    ylabel('幅度');
    grid on; grid minor;
    
    % 子图2：原始信号频谱
    subplot(4, 1, 2);
    plot(f_full/1e6, abs(scan_line_fft));
    title(sprintf('[SpectrumAnalysisAndCenterFreq] 原始信号频谱 (中心频率: %.2f MHz)', center_freq_full/1e6));
    xlabel('频率 (MHz)');
    ylabel('幅度');
    grid on; grid minor;

    % ======================================
    % 步骤7：基带移频处理（将中心频率移至零频率附近）
    % ======================================
    shifted_signal = scan_line .* exp(-1i*2*pi*center_freq_full*t);  % 复指数移频
    shifted_fft = fftshift(fft(shifted_signal));
    
    % 子图3：移频后基带信号频谱
    subplot(4, 1, 3);
    plot(f_full/1e6, abs(shifted_fft));
    title('[SpectrumAnalysisAndCenterFreq] 移频后的频谱（基带信号）');
    xlabel('频率 (MHz)');
    ylabel('幅度');
    grid on; grid minor;
    
    % 预留子图4（供滤波模块填充）
    subplot(4, 1, 4);
    title('[FilterDesignAndBatchProcess] 低通滤波后的频谱（待填充）');
    xlabel('频率 (MHz)');
    ylabel('幅度');
    grid on; grid minor;
    
    % 整体标题
    sgtitle('[SpectrumAnalysisAndCenterFreq-FilterDesignAndBatchProcess] 信号处理过程对比', 'FontSize', 12);
    
    disp('[SpectrumAnalysisAndCenterFreq] 3.6 已完成基带移频处理，信号中心频率移至零频率附近');
    disp('[SpectrumAnalysisAndCenterFreq] 步骤3 完成：已成功观察信号频谱并确定中心频率，完成基带移频处理');
end
