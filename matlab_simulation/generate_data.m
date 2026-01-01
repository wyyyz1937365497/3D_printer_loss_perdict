% 3D打印机转角误差数据生成脚本
% 生成包含尖锐转角和圆角的路径数据，用于训练误差预测模型
% 修改：更加强调当前点速度方向和下一点位置对误差的影响

fprintf('开始生成3D打印转角误差数据...\n');

% 参数设置
num_samples = 5000;  % 样本数量
max_velocity = 150;  % 最大速度 (mm/s)
min_radius = 0.5;    % 最小转角半径 (mm)
max_radius = 50;     % 最大转角半径 (mm)
nozzle_weight = 10;  % 喷头重量 (g)

% 初始化数据存储
velocities = zeros(num_samples, 1);      % 速度
radii = zeros(num_samples, 1);           % 转角半径
weights = zeros(num_samples, 1);         % 喷头重量
stiffnesses = zeros(num_samples, 1);     % 支撑梁刚度
ideal_positions_x = zeros(num_samples, 1);  % 理想x坐标
ideal_positions_y = zeros(num_samples, 1);  % 理想y坐标
displacements_x = zeros(num_samples, 1);    % x方向偏差
displacements_y = zeros(num_samples, 1);    % y方向偏差

% 生成随机参数
velocities = rand(num_samples, 1) * max_velocity;
radii = min_radius + rand(num_samples, 1) * (max_radius - min_radius);
weights = nozzle_weight * ones(num_samples, 1);
% 支撑梁刚度系数，保持在合理范围内
stiffnesses = 0.5 + rand(num_samples, 1) * 1.5;
% 理想位置在合理范围内
ideal_positions_x = rand(num_samples, 1) * 100;
ideal_positions_y = rand(num_samples, 1) * 100;

% 生成路径点序列以体现速度方向和下一点位置的影响
path_length = round(sqrt(num_samples));  % 估算路径长度

% 生成一个更复杂的路径，包含尖锐转角和圆角
fprintf('生成复杂路径...\n');
path_idx = 1;

% 生成一个矩形路径
rect_side_length = 20;
for i = 1:20
    if path_idx > num_samples
        break
    end
    % 底边
    ideal_positions_x(path_idx) = i * rect_side_length / 20;
    ideal_positions_y(path_idx) = 0;
    path_idx = path_idx + 1;
end

for i = 1:20
    if path_idx > num_samples
        break
    end
    % 右边
    ideal_positions_x(path_idx) = rect_side_length;
    ideal_positions_y(path_idx) = i * rect_side_length / 20;
    path_idx = path_idx + 1;
end

for i = 1:20
    if path_idx > num_samples
        break
    end
    % 顶边
    ideal_positions_x(path_idx) = rect_side_length - i * rect_side_length / 20;
    ideal_positions_y(path_idx) = rect_side_length;
    path_idx = path_idx + 1;
end

for i = 1:20
    if path_idx > num_samples
        break
    end
    % 左边
    ideal_positions_x(path_idx) = 0;
    ideal_positions_y(path_idx) = rect_side_length - i * rect_side_length / 20;
    path_idx = path_idx + 1;
end

% 生成一个圆形路径
circle_center_x = 30;
circle_center_y = 10;
circle_radius = 8;
for i = 1:30
    if path_idx > num_samples
        break
    end
    angle = 2 * pi * i / 30;
    ideal_positions_x(path_idx) = circle_center_x + circle_radius * cos(angle);
    ideal_positions_y(path_idx) = circle_center_y + circle_radius * sin(angle);
    path_idx = path_idx + 1;
end

% 补充剩余点，使其形成连续路径
for i = path_idx:num_samples
    if i > 1
        % 基于前一个点生成下一个点，形成连续路径
        prev_x = ideal_positions_x(i-1);
        prev_y = ideal_positions_y(i-1);
        % 随机选择一个方向和距离
        angle = rand() * 2 * pi;
        distance = 0.5 + rand() * 2;  % 0.5到2.5之间的距离
        ideal_positions_x(i) = prev_x + distance * cos(angle);
        ideal_positions_y(i) = prev_y + distance * sin(angle);
    else
        ideal_positions_x(i) = rand() * 50;
        ideal_positions_y(i) = rand() * 50;
    end
end

% 计算速度方向和下一点位置的影响
fprintf('计算速度方向和下一点位置对误差的影响...\n');
for i = 1:num_samples
    v = velocities(i);
    r = radii(i);
    k = stiffnesses(i);
    
    % 获取当前点和下一点的位置
    current_x = ideal_positions_x(i);
    current_y = ideal_positions_y(i);
    
    % 获取下一点位置（循环到第一个点）
    next_idx = mod(i, num_samples) + 1;
    next_x = ideal_positions_x(next_idx);
    next_y = ideal_positions_y(next_idx);
    
    % 计算当前点到下一点的方向向量
    direction_x = next_x - current_x;
    direction_y = next_y - current_y;
    direction_norm = sqrt(direction_x^2 + direction_y^2);
    
    % 归一化方向向量
    if direction_norm > 1e-6
        direction_x = direction_x / direction_norm;
        direction_y = direction_y / direction_norm;
    else
        direction_x = 1;
        direction_y = 0;
    end
    
    % 计算当前点的速度方向（如果可用）
    if i > 1
        prev_x = ideal_positions_x(i-1);
        prev_y = ideal_positions_y(i-1);
        vel_direction_x = current_x - prev_x;
        vel_direction_y = current_y - prev_y;
        vel_direction_norm = sqrt(vel_direction_x^2 + vel_direction_y^2);
        
        if vel_direction_norm > 1e-6
            vel_direction_x = vel_direction_x / vel_direction_norm;
            vel_direction_y = vel_direction_y / vel_direction_norm;
        else
            vel_direction_x = direction_x;
            vel_direction_y = direction_y;
        end
    else
        vel_direction_x = direction_x;
        vel_direction_y = direction_y;
    end
    
    % 计算实际偏差（基于物理模型的简化模拟）
    % 偏差与速度平方、喷头重量成正比，与支撑梁刚度成反比
    % 考虑速度方向和下一点位置的影响
    base_deviation = (v^2 * nozzle_weight) / (k * 1000);
    
    % 尖锐转角的偏差更大（半径小于5mm为尖锐转角）
    if r <= 5
        deviation_factor = 1.5;  % 尖锐转角偏差放大因子
    else
        deviation_factor = 1.0;  % 圆角偏差正常
    end
    
    % 考虑速度方向与目标方向之间的夹角影响（更加显著）
    angle_diff = acos(max(-1, min(1, vel_direction_x * direction_x + vel_direction_y * direction_y)));
    direction_influence = 1 + 2.0 * abs(sin(angle_diff));  % 夹角越大影响越大，权重增加
    
    % 基于速度方向和下一点位置的偏差方向
    deviation_x = base_deviation * deviation_factor * direction_influence * vel_direction_x;
    deviation_y = base_deviation * deviation_factor * direction_influence * vel_direction_y;
    
    % 添加基于下一点位置的额外影响（更加显著）
    distance_to_next = sqrt((next_x - current_x)^2 + (next_y - current_y)^2);
    if distance_to_next > 0.1  % 避免非常近的点
        % 更强的动态影响
        pos_influence = 0.8 * (v / max_velocity) * (distance_to_next / 50);  % 距离越远影响越大
        deviation_x = deviation_x + pos_influence * (next_x - current_x) / distance_to_next;
        deviation_y = deviation_y + pos_influence * (next_y - current_y) / distance_to_next;
    end
    
    % 添加动态路径曲率的影响
    if i > 1 && i < num_samples
        prev_x = ideal_positions_x(i-1);
        prev_y = ideal_positions_y(i-1);
        next_x = ideal_positions_x(i+1);
        next_y = ideal_positions_y(i+1);
        
        % 计算曲率（三点之间的角度变化）
        vec1_x = current_x - prev_x;
        vec1_y = current_y - prev_y;
        vec2_x = next_x - current_x;
        vec2_y = next_y - current_y;
        
        norm1 = sqrt(vec1_x^2 + vec1_y^2);
        norm2 = sqrt(vec2_x^2 + vec2_y^2);
        
        if norm1 > 1e-6 && norm2 > 1e-6
            dot_product = (vec1_x * vec2_x + vec1_y * vec2_y) / (norm1 * norm2);
            angle_change = acos(max(-1, min(1, dot_product)));
            
            % 曲率影响
            curvature_influence = 1.5 * (angle_change / pi);  % 最大180度变化
            deviation_x = deviation_x + curvature_influence * vel_direction_x * (v / max_velocity);
            deviation_y = deviation_y + curvature_influence * vel_direction_y * (v / max_velocity);
        end
    end
    
    % 添加随机噪声模拟其他因素
    noise_factor = 0.1;
    deviation_x = deviation_x + noise_factor * randn();
    deviation_y = deviation_y + noise_factor * randn();
    
    % 存储偏差
    displacements_x(i) = deviation_x;
    displacements_y(i) = deviation_y;
    
    if mod(i, 1000) == 0
        fprintf('已处理 %d/%d 个样本\n', i, num_samples);
    end
end

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
for i = 1:num_samples
    fprintf(fid, '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n', ...
        velocities(i), radii(i), weights(i), stiffnesses(i), ...
        ideal_positions_x(i), ideal_positions_y(i), ...
        displacements_x(i), displacements_y(i));
end

% 关闭文件
fclose(fid);

fprintf('数据生成完成！共生成 %d 个样本\n', num_samples);
fprintf('数据已保存至 %s\n', csv_filename);

% 输出一些统计信息
fprintf('位移向量统计:\n');
fprintf('X方向偏差: 平均值=%.4f, 标准差=%.4f\n', mean(displacements_x), std(displacements_x));
fprintf('Y方向偏差: 平均值=%.4f, 标准差=%.4f\n', mean(displacements_y), std(displacements_y));

% 输出尖锐转角和圆角的统计信息
sharp_corner_mask = radii <= 5;
fprintf('尖锐转角样本数: %d (%.2f%%)\n', sum(sharp_corner_mask), sum(sharp_corner_mask)/num_samples*100);
fprintf('圆角样本数: %d (%.2f%%)\n', sum(~sharp_corner_mask), sum(~sharp_corner_mask)/num_samples*100);

fprintf('尖锐转角位移向量统计:\n');
fprintf('X方向偏差: 平均值=%.4f, 标准差=%.4f\n', ...
    mean(displacements_x(sharp_corner_mask)), std(displacements_x(sharp_corner_mask)));
fprintf('Y方向偏差: 平均值=%.4f, 标准差=%.4f\n', ...
    mean(displacements_y(sharp_corner_mask)), std(displacements_y(sharp_corner_mask)));

fprintf('圆角位移向量统计:\n');
fprintf('X方向偏差: 平均值=%.4f, 标准差=%.4f\n', ...
    mean(displacements_x(~sharp_corner_mask)), std(displacements_x(~sharp_corner_mask)));
fprintf('Y方向偏差: 平均值=%.4f, 标准差=%.4f\n', ...
    mean(displacements_y(~sharp_corner_mask)), std(displacements_y(~sharp_corner_mask)));