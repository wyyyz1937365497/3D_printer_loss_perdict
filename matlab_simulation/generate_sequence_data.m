% 3D打印机转角误差序列数据生成脚本
% 生成包含转角前后序列的路径数据，用于训练序列模型

fprintf('开始生成3D打印转角误差序列数据...\n');

% 参数设置
num_sequences = 1000;  % 序列数量
sequence_length = 10;  % 每个序列的长度
max_velocity = 150;    % 最大速度 (mm/s)
min_radius = 0.5;      % 最小转角半径 (mm)
max_radius = 50;       % 最大转角半径 (mm)
nozzle_weight = 10;    % 喷头重量 (g)

% 初始化数据存储
sequences = zeros(num_sequences, sequence_length, 4);  % [速度, 转角半径, 喷头重量, 支撑梁刚度]
displacement_sequences = zeros(num_sequences, sequence_length, 2);  % [x方向偏差, y方向偏差]

% 生成随机参数
velocity_sequences = rand(num_sequences, sequence_length) * max_velocity;

% 生成包含尖锐转角的序列
sharp_corner_ratio = 0.3; % 30%的序列包含尖锐转角

for seq_idx = 1:num_sequences
    % 随机选择是否包含尖锐转角
    has_sharp_corner = (rand() < sharp_corner_ratio);
    
    for i = 1:sequence_length
        v = velocity_sequences(seq_idx, i);
        
        % 为序列中的点生成参数
        if has_sharp_corner
            % 在序列中间部分增加尖锐转角的可能性
            if i == ceil(sequence_length/2) && rand() < 0.8
                r = 0.5 + 4.5 * rand();  % 尖锐转角半径 (0.5-5mm)
            else
                r = 5 + 45 * rand();     % 圆角半径 (5-50mm)
            end
        else
            % 随机生成半径
            r = 0.5 + 49.5 * rand();   % 半径范围 (0.5-50mm)
        end
        
        % 支撑梁刚度系数，保持在合理范围内
        k = 0.5 + rand() * 1.5;
        
        % 计算当前点的特征
        sequences(seq_idx, i, :) = [v, r, nozzle_weight, k];
        
        % 计算实际偏差（基于物理模型的简化模拟）
        % 偏差与速度平方、喷头重量成正比，与支撑梁刚度成反比
        % 尖锐转角通常会有更大的偏差
        base_deviation = (v^2 * nozzle_weight) / (k * 1000);
        
        % 尖锐转角的偏差更大
        if r <= 5
            deviation_factor = 1.5 + rand() * 0.5;  % 尖锐转角偏差放大因子
        else
            deviation_factor = 1.0;  % 圆角偏差正常
        end
        
        % 根据转角类型计算偏差方向
        if r <= 5
            angle = pi/2 + (rand() - 0.5) * pi/4;  % 尖锐转角，添加一些随机方向变化
        else
            angle = pi/4 + (rand() - 0.5) * pi/2;  % 圆角，添加一些随机方向变化
        end
        
        deviation_x = base_deviation * deviation_factor * cos(angle + pi/4);
        deviation_y = base_deviation * deviation_factor * sin(angle + pi/4);
        
        % 添加随机噪声模拟其他因素
        noise_factor = 0.1;
        deviation_x = deviation_x + noise_factor * randn();
        deviation_y = deviation_y + noise_factor * randn();
        
        % 存储偏差
        displacement_sequences(seq_idx, i, :) = [deviation_x, deviation_y];
        
        if mod(seq_idx, 100) == 0
            fprintf('已生成 %d/%d 个序列\n', seq_idx, num_sequences);
        end
    end
end

% 合并序列数据
% 重塑数据以便保存
seq_data_reshaped = reshape(sequences, num_sequences * sequence_length, 4);
disp_data_reshaped = reshape(displacement_sequences, num_sequences * sequence_length, 2);

% 创建完整的数据集
dataset = [seq_data_reshaped, disp_data_reshaped];

% 确保数据目录存在
data_dir = '../data';
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
    fprintf('创建目录: %s\n', data_dir);
end

% 保存数据到CSV文件
csv_filename = fullfile(data_dir, 'printer_sequence_displacement_data.csv');
fprintf('保存序列数据到 %s\n', csv_filename);

% 检查文件是否能成功打开
fid = fopen(csv_filename, 'w');
if fid == -1
    error('无法打开文件 %s 进行写入', csv_filename);
end

% 写入CSV文件
fprintf(fid, 'velocity,radius,weight,stiffness,displacement_x,displacement_y,sequence_id,step_in_sequence\n');
for seq_idx = 1:num_sequences
    for step_idx = 1:sequence_length
        fprintf(fid, '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d\n', ...
            sequences(seq_idx, step_idx, 1), ...
            sequences(seq_idx, step_idx, 2), ...
            sequences(seq_idx, step_idx, 3), ...
            sequences(seq_idx, step_idx, 4), ...
            displacement_sequences(seq_idx, step_idx, 1), ...
            displacement_sequences(seq_idx, step_idx, 2), ...
            seq_idx, ...
            step_idx);
    end
end

% 关闭文件
fclose(fid);

fprintf('序列数据生成完成！共生成 %d 个序列，每个序列 %d 步\n', num_sequences, sequence_length);
fprintf('数据已保存至 %s\n', csv_filename);

% 输出一些统计信息
fprintf('位移向量统计:\n');
% 先计算X方向的统计
x_displacements = displacement_sequences(:, :, 1);
y_displacements = displacement_sequences(:, :, 2);

fprintf('X方向偏差: 平均值=%.4f, 标准差=%.4f\n', mean(x_displacements, 'all'), std(x_displacements, [], 'all'));
fprintf('Y方向偏差: 平均值=%.4f, 标准差=%.4f\n', mean(y_displacements, 'all'), std(y_displacements, [], 'all'));

% 计算尖锐转角和圆角的统计信息
sharp_corner_mask = squeeze(sequences(:, :, 2)) <= 5;
total_points = num_sequences * sequence_length;
fprintf('尖锐转角样本数: %d (%.2f%%)\n', sum(sharp_corner_mask(:)), sum(sharp_corner_mask(:))/total_points*100);
fprintf('圆角样本数: %d (%.2f%%)\n', sum(~sharp_corner_mask(:)), sum(~sharp_corner_mask(:))/total_points*100);

% 重塑数组以进行统计分析
sequences_reshaped = reshape(sequences, num_sequences * sequence_length, 4);
displacements_reshaped = reshape(displacement_sequences, num_sequences * sequence_length, 2);
sharp_corner_mask_reshaped = reshape(sharp_corner_mask, num_sequences * sequence_length, 1);

% 计算尖锐转角统计
sharp_corner_indices = find(sharp_corner_mask_reshaped);
if ~isempty(sharp_corner_indices)
    fprintf('尖锐转角位移向量统计:\n');
    fprintf('X方向偏差: 平均值=%.4f, 标准差=%.4f\n', ...
        mean(displacements_reshaped(sharp_corner_indices, 1)), std(displacements_reshaped(sharp_corner_indices, 1)));
    fprintf('Y方向偏差: 平均值=%.4f, 标准差=%.4f\n', ...
        mean(displacements_reshaped(sharp_corner_indices, 2)), std(displacements_reshaped(sharp_corner_indices, 2)));
else
    fprintf('尖锐转角位移向量统计: 无尖锐转角样本\n');
end

% 计算圆角统计
round_corner_indices = find(~sharp_corner_mask_reshaped);
if ~isempty(round_corner_indices)
    fprintf('圆角位移向量统计:\n');
    fprintf('X方向偏差: 平均值=%.4f, 标准差=%.4f\n', ...
        mean(displacements_reshaped(round_corner_indices, 1)), std(displacements_reshaped(round_corner_indices, 1)));
    fprintf('Y方向偏差: 平均值=%.4f, 标准差=%.4f\n', ...
        mean(displacements_reshaped(round_corner_indices, 2)), std(displacements_reshaped(round_corner_indices, 2)));
else
    fprintf('圆角位移向量统计: 无圆角样本\n');
end