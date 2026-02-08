% 功能：对滤波后的信号进行下采样抽取，完成图像增强与B超矩形成像
% 输入：批量滤波后的RF数据矩阵、FIR滤波器、核心参数（来自FilterDesignAndBatchProcess）
% 输出：归一化B超矩形图像、抽取因子、成像参数（供后续模块调用）
% 说明：文件名以字母开头，符合Matlab语法要求

function [ultrasound_image_norm, params] = SignalDecimationAndBModeImaging(image_data, lp_filter, filtered_signal, params)
    disp('======================================');
    disp('[SignalDecimationAndBModeImaging] 步骤5：信号抽取与B超矩形成像');

    % ======================================
    % 步骤1：配置抽取参数并完成下采样抽取
    % ======================================
    params.D = 4;  % 抽取因子（下采样倍数，可调整）
    filtered_scan_line = filtered_signal;
    extracted_scan_line = downsample(filtered_scan_line, params.D);  % 下采样抽取
    disp(['[SignalDecimationAndBModeImaging] 5.1 信号抽取完成：抽取因子D=', num2str(params.D)]);

    % ======================================
    % 步骤2：计算抽取后信号参数并绘制频谱图
    % ======================================
    N_extracted = length(extracted_scan_line);
    params.Fs_extracted = params.Fs / params.D;  % 抽取后采样频率
    f_extracted = (-N_extracted/2:N_extracted/2-1) * (params.Fs_extracted / N_extracted);
    extracted_scan_line_fft = fftshift(fft(extracted_scan_line));
    
    % 绘制抽取后信号频谱图
    screen_size = get(0, 'ScreenSize');
    fig3_width = 800;
    fig3_height = 500;
    fig3_left = (screen_size(3) - fig3_width) / 2;
    fig3_bottom = (screen_size(4) - fig3_height) / 2 - 150;
    
    figure('Name', '[SignalDecimationAndBModeImaging] 抽取后信号频谱分析', ...
        'Position', [fig3_left, fig3_bottom, fig3_width, fig3_height], ...
        'WindowStyle', 'normal');
    plot(f_extracted/1e6, abs(extracted_scan_line_fft), 'LineWidth', 1, 'Color', [0.8, 0.3, 0]);
    title(sprintf('[SignalDecimationAndBModeImaging] 抽取后信号的频谱（D=%d，采样频率=%.2f MHz）', params.D, params.Fs_extracted/1e6));
    xlabel('频率 (MHz)');
    ylabel('幅度');
    grid on; grid minor;
    disp('[SignalDecimationAndBModeImaging] 5.2 抽取后信号频谱图绘制完成');

    % ======================================
    % 步骤3：初始化B超成像矩阵并完成批量抽取
    % ======================================
    params.extracted_num_points_per_line = ceil(params.num_points_per_line / params.D);
    ultrasound_image = zeros(params.num_lines, params.extracted_num_points_per_line);
    disp('[SignalDecimationAndBModeImaging] 5.3 开始批量抽取所有扫描线，构建B超成像矩阵，进度如下：');
    
    for i = 1:params.num_lines
        current_scan_line = image_data(i, :);
        filtered_line = filter(lp_filter, current_scan_line);
        extracted_line = downsample(filtered_line, params.D);
        
        % 填充0以保证矩阵尺寸统一
        if length(extracted_line) < params.extracted_num_points_per_line
            extracted_line(end+1:params.extracted_num_points_per_line) = 0;
        end
        
        ultrasound_image(i, :) = extracted_line;
        
        % 输出成像进度
        if mod(i, 100) == 0
            disp(['[SignalDecimationAndBModeImaging] 成像进度：', num2str(i), '/', num2str(params.num_lines), ' 条扫描线']);
        end
    end

    % ======================================
    % 步骤4：图像增强处理（对数变换→动态范围压缩→百分位数截断→gamma校正）
    % ======================================
    % 4.1 对数变换：压缩动态范围，增强暗部细节
    log_compressed_image = 20 * log10(abs(ultrasound_image) + 1);
    
    % 4.2 动态范围压缩：剔除极端值
    dynamic_range = 50;  % 动态范围（50dB）
    max_dB = max(log_compressed_image(:));
    log_compressed_image = max(log_compressed_image, max_dB - dynamic_range);
    
    % 4.3 百分位数截断：优化对比度
    percentile_low = 5;
    percentile_high = 95;
    min_dB = prctile(log_compressed_image(:), percentile_low);
    max_dB_trunc = prctile(log_compressed_image(:), percentile_high);
    
    % 4.4 归一化：映射到[0,1]区间
    ultrasound_image_norm = (log_compressed_image - min_dB) / (max_dB_trunc - min_dB);
    
    % 4.5 gamma校正：提亮暗部
    params.gamma = 0.7;
    ultrasound_image_norm = ultrasound_image_norm .^ params.gamma;
    
    % 4.6 限制极值：避免显示异常
    ultrasound_image_norm = min(max(ultrasound_image_norm, 0), 1);
    disp('[SignalDecimationAndBModeImaging] 5.4 图像增强处理完成（对数变换+动态范围压缩+gamma校正）');

    % ======================================
    % 步骤5：绘制B超矩形图像
    % ======================================
    fig4_width = 900;
    fig4_height = 700;
    fig4_left = (screen_size(3) - fig4_width) / 2;
    fig4_bottom = (screen_size(4) - fig4_height) / 2 - 200;
    
    figure('Name', '[SignalDecimationAndBModeImaging] 优化灰度后的B超成像结果（矩形）', ...
        'Position', [fig4_left, fig4_bottom, fig4_width, fig4_height], ...
        'WindowStyle', 'normal');
    imagesc(ultrasound_image_norm);
    colormap(gray);
    axis image;
    colorbar;
    title(sprintf('[SignalDecimationAndBModeImaging] 优化灰度后的B超成像结果（矩形，抽取因子D=%d）', params.D));
    xlabel(['扫描线序号（总数量：', num2str(params.num_lines), '）']);
    ylabel(['抽取后的样本点（总数量：', num2str(params.extracted_num_points_per_line), '）']);
    set(gca, 'CLim', [0 1]);
    disp('[SignalDecimationAndBModeImaging] 5.5 B超矩形图像绘制完成');
    disp('[SignalDecimationAndBModeImaging] 步骤5 完成：已成功完成信号抽取与灰度B超成像，得到矩形B超图像');
end
