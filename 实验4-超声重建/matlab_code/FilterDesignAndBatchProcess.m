% 功能：设计FIR低通滤波器，对移频后的基带信号进行滤波，批量处理所有扫描线
% 输入：二维RF数据矩阵、基带移频信号、中心频率、时间轴/频率轴（来自SpectrumAnalysisAndCenterFreq）
% 输出：批量滤波后的RF数据矩阵、FIR滤波器、滤波后的信号（供后续模块调用）
% 说明：文件名以字母开头，符合Matlab语法要求

function [image_data, lp_filter, filtered_signal, params] = FilterDesignAndBatchProcess(image_data, shifted_signal, center_freq_full, t, f_full, params)
    disp('======================================');
    disp('[FilterDesignAndBatchProcess] 步骤4：设计滤波器并完成批量信号滤波');

    % ======================================
    % 步骤1：配置FIR低通滤波器参数
    % ======================================
    params.f_cutoff_stop = params.Fs / 8;  % 阻带截止频率（5MHz，与采样频率匹配）
    params.f_cutoff_pass = 0.455 * params.f_cutoff_stop;  % 通带截止频率（经验值，保证滤波效果）
    disp(['[FilterDesignAndBatchProcess] 4.1 滤波器参数配置完成：通带截止频率（Hz）=', num2str(params.f_cutoff_pass)]);
    disp(['[FilterDesignAndBatchProcess] 4.1 滤波器参数配置完成：阻带截止频率（Hz）=', num2str(params.f_cutoff_stop)]);

    % ======================================
    % 步骤2：设计FIR低通滤波器（需Signal Processing Toolbox）
    % ======================================
    lp_filter = designfilt('lowpassfir', ...
        'PassbandFrequency', params.f_cutoff_pass, ...
        'StopbandFrequency', params.f_cutoff_stop, ...
        'PassbandRipple', 0.5, ...
        'StopbandAttenuation', 60, ...
        'SampleRate', params.Fs);
    disp('[FilterDesignAndBatchProcess] 4.2 FIR低通滤波器设计完成（阻带衰减60dB，通带波纹0.5）');

    % ======================================
    % 步骤3：对单条基带信号进行低通滤波
    % ======================================
    filtered_signal = filter(lp_filter, real(shifted_signal));  % 取实部，滤除虚部噪声
    filtered_fft = fftshift(fft(filtered_signal));
    
    % 填充信号处理对比图的子图4
    % 步骤1：根据窗口名称查找已有窗口句柄
    fig_handle = findobj('Name', '[SpectrumAnalysisAndCenterFreq] 信号处理过程对比');
    % 步骤2：判断窗口是否存在，存在则激活，不存在则提示（避免报错）
    if ~isempty(fig_handle)
        figure(fig_handle);  % 传入窗口句柄，激活已有窗口
    else
        warning('[FilterDesignAndBatchProcess] 警告：未找到指定名称的图形窗口，无法更新滤波频谱');
    end
    subplot(4, 1, 4);
    plot(f_full/1e6, abs(filtered_fft));
    title(sprintf('[FilterDesignAndBatchProcess] 低通滤波后的频谱 (截止频率: %.2f MHz)', params.f_cutoff_pass/1e6));
    xlabel('频率 (MHz)');
    ylabel('幅度');
    grid on; grid minor;
    disp('[FilterDesignAndBatchProcess] 4.3 单条扫描线基带信号滤波完成，频谱已更新');

    % ======================================
    % 步骤4：批量处理所有扫描线（移频+滤波）
    % ======================================
    processed_image = zeros(size(image_data));  % 预分配内存，提升处理速度
    disp('[FilterDesignAndBatchProcess] 4.4 开始批量处理所有扫描线，进度如下：');
    
    for i = 1:params.num_lines
        current_line = image_data(i, :);
        
        % 移频处理（与单条信号一致）
        shifted_line = current_line .* exp(-1i*2*pi*center_freq_full*t);
        
        % 低通滤波（取实部）
        filtered_line = filter(lp_filter, real(shifted_line));
        
        % 存储处理后的数据
        processed_image(i, :) = filtered_line;
        
        % 输出处理进度
        if mod(i, 100) == 0
            disp(['[FilterDesignAndBatchProcess] 已处理 ', num2str(i), '/', num2str(params.num_lines), ' 条扫描线']);
        end
    end
    
    % 更新图像数据为批量滤波后的结果
    image_data = processed_image;
    disp('[FilterDesignAndBatchProcess] 4.5 所有扫描线批量滤波处理完成');
    disp('[FilterDesignAndBatchProcess] 步骤4 完成：已成功设计并使用FIR低通滤波器完成超声信号滤波处理');
end
