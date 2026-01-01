% 3D打印机转角误差数据生成脚本
% 生成理想路径与实际路径的偏差数据集，包括圆角和尖锐转角

fprintf('开始生成3D打印转角误差数据...\n');

% 参数设置
num_samples = 5000;  % 数据样本数量
max_velocity = 150;  % 最大速度 (mm/s)
min_radius = 0.5;    % 最小转角半径 (mm) - 现在包括尖锐转角
max_radius = 50;     % 最大转角半径 (mm)
nozzle_weight = 10;  % 喷头重量 (g)

% 初始化数据存储
data = zeros(num_samples, 6);  % [速度, 转角半径, 喷头重量, 支撑梁刚度, 理想x, 理想y]
displacement_vectors = zeros(num_samples, 2);  % [x方向偏差, y方向偏差]

% 生成随机参数
velocities = rand(num_samples, 1) * max_velocity;
% 包含尖锐转角：生成更小的半径值，包括接近0的值（代表尖锐转角）
radius_selector = rand(num_samples, 1);
sharp_corner_ratio = 0.3; % 30%的样本是尖锐转角

radii = zeros(num_samples, 1);
% 对于尖锐转角，使用更小的半径值
radii(radius_selector <= sharp_corner_ratio) = 0.5 + 4.5 * rand(sum(radius_selector <= sharp_corner_ratio), 1);
% 对于圆角，使用较大的半径值
radii(radius_selector > sharp_corner_ratio) = 5 + 45 * rand(sum(radius_selector > sharp_corner_ratio), 1);

stiffness_values = 0.5 + rand(num_samples, 1) * 1.5;  % 支撑梁刚度系数

for i = 1:num_samples
    % 当前参数
    v = velocities(i);
    r = radii(i);
    k = stiffness_values(i);
    
    % 生成理想路径点
    % 对于尖锐转角和圆角使用不同的角度处理
    if r <= 5  % 尖锐转角
        angle = pi/2;  % 90度转角
        ideal_x = r * cos(angle/2);  % 简化模型，实际应用中可能需要更复杂的几何计算
        ideal_y = r * sin(angle/2);
    else  % 圆角
        angle = pi/4;  % 圆角角度
        ideal_x = r * cos(angle);
        ideal_y = r * sin(angle);
    end
    
    % 计算实际偏差（基于物理模型的简化模拟）
    % 偏差与速度平方、喷头重量成正比，与支撑梁刚度成反比
    % 尖锐转角通常会有更大的偏差
    base_deviation = (v^2 * nozzle_weight) / (k * 1000);
    
    % 尖锐转角的偏差更大
    if r <= 5
        deviation_factor = 1.5;  % 尖锐转角偏差放大因子
    else
        deviation_factor = 1.0;  % 圆角偏差正常
    end
    
    deviation_x = base_deviation * deviation_factor * cos(angle + pi/4);
    deviation_y = base_deviation * deviation_factor * sin(angle + pi/4);
    
    % 添加随机噪声模拟其他因素
    noise_factor = 0.1;
    deviation_x = deviation_x + noise_factor * randn();
    deviation_y = deviation_y + noise_factor * randn();
    
    % 存储数据
    data(i, :) = [v, r, nozzle_weight, k, ideal_x, ideal_y];
    displacement_vectors(i, :) = [deviation_x, deviation_y];
    
    if mod(i, 1000) == 0
        fprintf('已生成 %d/%d 个样本\n', i, num_samples);
    end
end

% 合并输入特征和输出标签
dataset = [data, displacement_vectors];

% 确保数据目录存在
data_dir = '../data';
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
    fprintf('创建目录: %s\n', data_dir);
end

% 保存数据到CSV文件
csv_filename = fullfile(data_dir, 'printer_displacement_data.csv');
fprintf('保存数据到 %s\n', csv_filename);

% 检查文件是否能成功打开
fid = fopen(csv_filename, 'w');
if fid == -1
    error('无法打开文件 %s 进行写入', csv_filename);
end

% 写入CSV文件
fprintf(fid, 'velocity,radius,weight,stiffness,ideal_x,ideal_y,displacement_x,displacement_y\n');
for i = 1:size(dataset, 1)
    fprintf(fid, '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n', dataset(i, :));
end

% 关闭文件
fclose(fid);

fprintf('数据生成完成！共生成 %d 个样本\n', num_samples);
fprintf('数据已保存至 %s\n', csv_filename);

% 输出一些统计信息
fprintf('位移向量统计:\n');
fprintf('X方向偏差: 平均值=%.4f, 标准差=%.4f\n', mean(displacement_vectors(:,1)), std(displacement_vectors(:,1)));
fprintf('Y方向偏差: 平均值=%.4f, 标准差=%.4f\n', mean(displacement_vectors(:,2)), std(displacement_vectors(:,2)));

% 输出尖锐转角和圆角的统计信息
sharp_corner_indices = radii <= 5;
fprintf('尖锐转角样本数: %d (%.2f%%)\n', sum(sharp_corner_indices), sum(sharp_corner_indices)/num_samples*100);
fprintf('圆角样本数: %d (%.2f%%)\n', sum(~sharp_corner_indices), sum(~sharp_corner_indices)/num_samples*100);

fprintf('尖锐转角位移向量统计:\n');
fprintf('X方向偏差: 平均值=%.4f, 标准差=%.4f\n', ...
    mean(displacement_vectors(sharp_corner_indices,1)), std(displacement_vectors(sharp_corner_indices,1)));
fprintf('Y方向偏差: 平均值=%.4f, 标准差=%.4f\n', ...
    mean(displacement_vectors(sharp_corner_indices,2)), std(displacement_vectors(sharp_corner_indices,2)));

fprintf('圆角位移向量统计:\n');
fprintf('X方向偏差: 平均值=%.4f, 标准差=%.4f\n', ...
    mean(displacement_vectors(~sharp_corner_indices,1)), std(displacement_vectors(~sharp_corner_indices,1)));
fprintf('Y方向偏差: 平均值=%.4f, 标准差=%.4f\n', ...
    mean(displacement_vectors(~sharp_corner_indices,2)), std(displacement_vectors(~sharp_corner_indices,2)));