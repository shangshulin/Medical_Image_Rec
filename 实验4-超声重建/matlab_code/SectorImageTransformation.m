% 功能：将重排后的矩形B超图像转换为凸阵探头扇环图像（极坐标→笛卡尔坐标）
% 输入：重排后的B超图像、图像尺寸信息、核心参数（来自ImageRearrangement）
% 输出：扇环B超图像、保存所有处理结果
% 说明：文件名以字母开头，符合Matlab语法要求

function SectorImageTransformation(ultrasound_image_norm, num_samples, num_scans, params)
    disp('======================================');
    disp('[SectorImageTransformation] 步骤7：图像坐标变换成扇形图（凸阵探头）');

    % ======================================
    % 步骤1：配置凸阵探头扇环参数
    % ======================================
    params.scan_angle = 65;  % 扫描角度（65度，符合凸阵探头常规参数）
    params.radius_start = 50;  % 起始半径（50像素，避开探头盲区）
    params.radius_end = 800;  % 结束半径（800像素，最大成像深度）
    disp(['[SectorImageTransformation] 7.1 凸阵探头参数配置完成：扫描角度=', num2str(params.scan_angle), '度']);
    disp(['[SectorImageTransformation] 7.1 扇环图像参数配置完成：起始半径=', num2str(params.radius_start), '像素，结束半径=', num2str(params.radius_end), '像素']);

    % ======================================
    % 步骤2：创建扇环坐标网格（极坐标）
    % ======================================
    theta = linspace(-params.scan_angle/2, params.scan_angle/2, num_scans);  % 角度网格
    r = linspace(params.radius_start, params.radius_end, num_samples);       % 半径网格
    [THETA, R] = meshgrid(theta, r);  % 生成二维极坐标网格
    disp('[SectorImageTransformation] 7.2 已创建扇环极坐标网格');

    % ======================================
    % 步骤3：极坐标→笛卡尔坐标转换（直接处理角度值，无需弧度转换）
    % ======================================
    X = R .* sind(THETA);
    Y = R .* cosd(THETA);
    disp('[SectorImageTransformation] 7.3 已完成极坐标→笛卡尔坐标转换');

    % ======================================
    % 步骤4：绘制凸阵探头扇环图像
    % ======================================
    screen_size = get(0, 'ScreenSize');
    fig5_width = 800;
    fig5_height = 800;
    fig5_left = (screen_size(3) - fig5_width) / 2;
    fig5_bottom = (screen_size(4) - fig5_height) / 2 - 250;
    
    figure('Name', '[SectorImageTransformation] 凸阵探头B超扇环图像', ...
        'Position', [fig5_left, fig5_bottom, fig5_width, fig5_height], ...
        'WindowStyle', 'normal');
    
    h = pcolor(X, Y, ultrasound_image_norm);
    shading interp;  % 插值着色，提升图像平滑度
    axis equal;      % 保持纵横比，避免扇环变形
    axis ij;         % 翻转Y轴，符合B超视觉习惯
    
    % 设置显示范围
    max_range = max(params.radius_end * sind(params.scan_angle/2), params.radius_end);
    xlim([-max_range max_range]);
    ylim([0 params.radius_end]);
    
    % 配置颜色与标注
    colormap(gray);
    colorbar;
    title('[SectorImageTransformation] 凸阵探头B超扇环图像');
    xlabel('横向距离 (像素)');
    ylabel('深度 (像素)');
    disp('[SectorImageTransformation] 7.4 扇环图像基础绘制完成');

    % ======================================
    % 步骤5：添加扇环边界线，提升可读性
    % ======================================
    hold on;
    theta_line = linspace(-params.scan_angle/2, params.scan_angle/2, 100);
    
    % 内弧
    x_inner = params.radius_start * sind(theta_line);
    y_inner = params.radius_start * cosd(theta_line);
    plot(x_inner, y_inner, 'r--', 'LineWidth', 1);
    
    % 外弧
    x_outer = params.radius_end * sind(theta_line);
    y_outer = params.radius_end * cosd(theta_line);
    plot(x_outer, y_outer, 'r--', 'LineWidth', 1);
    
    % 左右边界线
    plot([params.radius_end * sind(-params.scan_angle/2), 0], ...
         [params.radius_end * cosd(-params.scan_angle/2), 0], 'r--', 'LineWidth', 1);
    plot([params.radius_end * sind(params.scan_angle/2), 0], ...
         [params.radius_end * cosd(params.scan_angle/2), 0], 'r--', 'LineWidth', 1);
    hold off;
    
    % 优化显示效果
    grid on;
    box on;
    disp('[SectorImageTransformation] 7.5 扇环边界线添加完成，图像可读性提升');

    % ======================================
    % 步骤6：保存所有处理结果
    % ======================================
    % 修正后：仅保存当前函数可访问的变量（输入参数+内部创建的变量）
    save('ultrasound_complete_processing_with_bmode_and_sector.mat', ...
        'ultrasound_image_norm', 'params', ...
        'X', 'Y', 'THETA', 'R');

    disp('[SectorImageTransformation] 7.6 所有处理结果已保存至：ultrasound_complete_processing_with_bmode_and_sector.mat');

    % ======================================
    % 流程完成提示
    % ======================================
    disp('======================================');
    disp('[SectorImageTransformation] 步骤7 完成：已成功将矩形B超图像转换为凸阵探头扇环图像');
    disp('[整体流程] 所有步骤全部完成！已成功完成从超声RF信号导入到凸阵探头扇环图像生成的全流程');
    disp('[整体流程] 生成的图像窗口包含：时域频域分析图、信号处理对比图、抽取后频谱图、矩形B超图、扇环B超图');
end
