% 功能：将存储超声RF原始数据（肝脏）的.pgm文件批量转换为.int32格式二进制.dat文件
% 输入：./raw_data/RF/2025_12_25__14_50_56/ 下的.pgm文件
% 输出：wht_liver_dat/ 下的对应.dat文件（int32格式，去除544字节文件头）
% 说明：文件名以字母开头，符合Matlab语法要求

function ConvertPgmToDat()
    % ======================================
    % 配置参数（可根据实际数据调整）
    % ======================================
    input_dir = './raw_data/RF/2025_12_25__14_50_56/';  % 输入.pgm文件所在目录
    output_dir = 'wht_liver_dat/';  % 输出.dat文件的目标文件夹
    header_size = 544;  % 固定文件头大小（544字节，需与.pgm文件格式匹配）
    expected_size = 3716 * 760;  % 期望的数据大小（单条扫描线3716点，共760条扫描线）

    % ======================================
    % 步骤1：创建输出文件夹（若不存在则自动创建）
    % ======================================
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);  % 创建目标文件夹，避免写入文件时因文件夹不存在报错
        disp(['[ConvertPgmToDat] 已创建输出文件夹：', output_dir]);
    end

    % ======================================
    % 步骤2：获取输入目录下所有.pgm格式文件的列表
    % ======================================
    pgm_file_list = dir(fullfile(input_dir, '*.pgm'));  % 查找目录下所有.pgm文件
    if isempty(pgm_file_list)
        error('[ConvertPgmToDat] 错误：在目录 %s 中未找到任何.pgm格式文件！', input_dir);
    end
    disp(['[ConvertPgmToDat] 共找到 ', num2str(length(pgm_file_list)), ' 个.pgm文件，开始批量处理...']);

    % ======================================
    % 步骤3：循环遍历所有.pgm文件，逐一处理并保存
    % ======================================
    for i = 1:length(pgm_file_list)
        % 提取单个.pgm文件的完整路径
        pgm_filename = pgm_file_list(i).name;  % 获取文件名（含后缀）
        pgm_full_path = fullfile(input_dir, pgm_filename);  % 拼接完整文件路径
        
        % 构造对应的输出.dat文件完整路径
        dat_filename = strrep(pgm_filename, '.pgm', '.dat');  % 替换文件后缀：.pgm -> .dat
        dat_full_path = fullfile(output_dir, dat_filename);  % 拼接输出文件的完整路径

        % ======================================
        % 步骤4：读取.pgm文件（跳过文件头，读取有效RF数据）
        % ======================================
        fid = fopen(pgm_full_path, 'r');  % 以只读方式打开.pgm文件
        if fid == -1
            warning('[ConvertPgmToDat] 警告：跳过无法打开的文件：%s', pgm_full_path);
            continue;
        end
        
        % 读取并跳过文件头（544字节，char格式）
        fread(fid, header_size, 'char');
        
        % 读取有效RF数据（int32格式，4字节/数据点，对应超声RF信号格式）
        rf_data = fread(fid, 'int32');
        
        % 关闭.pgm文件
        fclose(fid);

        % ======================================
        % 步骤5：验证数据大小（确保数据完整性）
        % ======================================
        if length(rf_data) ~= expected_size
            warning('[ConvertPgmToDat] 警告：跳过数据大小不符的文件：%s（实际长度：%d，预期长度：%d）', ...
                pgm_filename, length(rf_data), expected_size);
            continue;
        end

        % ======================================
        % 步骤6：写入.dat文件（二进制格式，保留int32数据类型）
        % ======================================
        fid_out = fopen(dat_full_path, 'wb');  % 以二进制写入模式打开.dat文件
        if fid_out == -1
            warning('[ConvertPgmToDat] 警告：跳过无法创建的输出文件：%s', dat_full_path);
            continue;
        end
        
        % 写入int32格式数据（与后续信号处理格式保持一致）
        fwrite(fid_out, rf_data, 'int32');
        
        % 关闭.dat文件
        fclose(fid_out);

        % 输出当前文件处理完成提示
        disp(['[ConvertPgmToDat] 已成功处理并保存：', dat_full_path]);
    end

    % ======================================
    % 批量处理完成提示
    % ======================================
    disp('======================================');
    % 修正后代码
    disp(sprintf('[ConvertPgmToDat] 所有.pgm文件批量处理完成！结果已保存至：%s', output_dir));

end
