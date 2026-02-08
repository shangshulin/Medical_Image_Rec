% 功能：从.dat二进制文件导入超声RF信号，完成二进制->十进制转换、数据重塑与归一化
% 输入：wht_liver_dat/ 下的.dat文件
% 输出：工作区中的二维RF数据矩阵、归一化单条扫描线数据（供后续模块调用）
% 说明：文件名以字母开头，符合Matlab语法要求

function [image_data, normalized_scan_line, scan_line, params] = SignalImportAndPreprocess()
    % ======================================
    % 步骤1：配置核心参数（与格式转换模块保持一致）
    % ======================================
    params.file_path = './wht_liver_dat/RF_B_00000.dat';  % 输入.dat文件路径
    params.num_points_per_line = 3716;  % 每行点数（单条扫描线的样本点数量）
    params.num_lines = 760;  % 扫描线总数（图像行数）
    params.data_type = 'int32';  % 数据类型（4字节/数据点）
    params.Fs = 40e6;  % 采样频率（40MHz，超声RF信号采样频率）
    params.target_scan_line = 1;  % 待分析的扫描线序号（1~760）
    params.expected_size = params.num_points_per_line * params.num_lines;  % 预期数据总长度

    disp('======================================');
    disp('[SignalImportAndPreprocess] 步骤1：超声RF信号的获取与导入');
    disp('[SignalImportAndPreprocess] 1.1 配置完成，开始读取.dat二进制文件...');

    % ======================================
    % 步骤2：打开并读取.dat二进制文件
    % ======================================
    fid = fopen(params.file_path, 'rb');  % 以二进制只读模式打开文件
    if fid == -1
        error('[SignalImportAndPreprocess] 错误：无法打开文件：%s，请检查文件路径是否正确', params.file_path);
    end
    
    % 读取二进制数据并转换为十进制数值
    data = fread(fid, params.data_type);
    fclose(fid);  % 关闭文件
    disp(['[SignalImportAndPreprocess] 1.2 已成功读取.dat文件，原始数据长度：', num2str(length(data))]);

    % ======================================
    % 步骤3：验证数据长度有效性
    % ======================================
    disp('[SignalImportAndPreprocess] 步骤2：二进制文件->十进制数据转换与验证');
    disp(['[SignalImportAndPreprocess] 2.1 预期数据长度（十进制数据点个数）：', num2str(params.expected_size)]);
    disp(['[SignalImportAndPreprocess] 2.2 实际读取长度（十进制数据点个数）：', num2str(length(data))]);
    
    if length(data) ~= params.expected_size
        error('[SignalImportAndPreprocess] 错误：读取的数据大小与预期不一致！实际：%d，预期：%d', length(data), params.expected_size);
    else
        disp('[SignalImportAndPreprocess] 2.3 数据长度验证通过，二进制数据已完整转换为十进制数值');
    end

    % ======================================
    % 步骤4：将一维数据重塑为二维矩阵（匹配超声扫描结构）
    % ======================================
    % 先reshape为[单条扫描线点数, 扫描线总数]，再转置得到[扫描线总数, 单条扫描线点数]
    image_data = reshape(data, [params.num_points_per_line, params.num_lines])';
    disp(['[SignalImportAndPreprocess] 2.4 二维十进制矩阵最终尺寸（扫描线数×单条线样本点数）：', num2str(size(image_data))]);

    % ======================================
    % 步骤5：提取目标扫描线并进行归一化预处理
    % ======================================
    if params.target_scan_line < 1 || params.target_scan_line > params.num_lines
        error('[SignalImportAndPreprocess] 错误：扫描线序号超出范围！请选择1~%d之间的数值', params.num_lines);
    end
    
    % 提取目标扫描线
    scan_line = image_data(params.target_scan_line, :);
    disp(['[SignalImportAndPreprocess] 2.5 已提取第', num2str(params.target_scan_line), '条扫描线作为分析目标']);
    
    % 归一化处理（映射到[0,1]区间，便于时域波形观察）
    scan_line_min = min(scan_line);
    scan_line_max = max(scan_line);
    if scan_line_max == scan_line_min
        normalized_scan_line = zeros(size(scan_line));
        warning('[SignalImportAndPreprocess] 警告：所选扫描线数据无波动，归一化结果为全零矩阵');
    else
        normalized_scan_line = (scan_line - scan_line_min) / (scan_line_max - scan_line_min);
    end
    
    disp('[SignalImportAndPreprocess] 2.6 目标扫描线已完成归一化预处理');
    disp('[SignalImportAndPreprocess] 步骤2 完成：二进制.dat文件已成功转换为十进制数据并完成格式整理');
end
