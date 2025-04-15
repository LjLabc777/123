clear;
clc;

measured_values_path = 'measured_values_all_dc.mat';
state_values_path = 'state_values_all_dc.mat';

num = 70080;


measured_data = load(measured_values_path);
state_data = load(state_values_path);

disp(fieldnames(measured_data));
measured_values_all = measured_data.measured_values_all;

disp(fieldnames(measured_values_all));
bus_Pinj = measured_values_all.bus_Pinj;
branch_Pinj = measured_values_all.branch_Pinj;

disp(fieldnames(state_data));
state_values_all = state_data.state_values_all;

disp(fieldnames(state_values_all));

bus_Va = state_values_all.bus_Va;
bus_Va_rad = bus_Va * pi / 180;

P_real = [bus_Pinj; branch_Pinj]';
state_values = bus_Va';
P_lc = P_real + n_std * randn(num, 137);
lc_data = P_lc;

P_zaosheng = P_lc - P_real;
P_zaosheng_fc = var(P_zaosheng, 1);
W = zeros(137, 137);

for i = 1:137
    W(i, i) = 1 / P_zaosheng_fc(i);
end

mpc = loadcase('case57');
[Bbus, Bf, Pbusinj, Pfinj] = makeBdc(mpc);
Bf_matrix = full(Bf);
Bbus_matrix = full(Bbus);
H1 = [Bbus_matrix; Bf_matrix];
H = H1(:, 2:end);

Z = P_lc'./100;
Z_zhuanzhi = Z';
A = H' * W * H;
x = (inv(A) * H' * W * Z );
x1 = [zeros(1, size(x, 2)); x];
z123 = H * x;
cancha = Z - H * x;

std_dev1 = ones(size(cancha)) * 100;

normalized_cancha = abs(cancha ./ std_dev1);

max_normalized_cancha = max(normalized_cancha);

xx = x';

index1 = randperm(70080);
i = 1;
for j = 1:30300
    index2(i) = index1(j);
    i = i + 1;
end

attack = sort(index2);
attack_index1 = unique(attack);
index = attack_index1';
attack_num = length(attack_index1);
lc_data = P_lc;

attack_index = [];

for m = 1:length(attack)
    i = attack(m);
    for k = 1:15
        attack_index = [attack_index; i];
        i = i + 1;
    end
end

zn = [];
xn = [];
Z1 = Z';
for i = 1:attack_num
    z_nor = Z1(index(i), :);
    zn = [zn; z_nor];
    x_nor = xx(index(i), :);
    xn = [xn; x_nor];
end

label1 = zeros(57, attack_num);

for i = 1:attack_num
    k_num = randi([2, 5]);
    selected_nodes = randperm(57, k_num);
    for j = 1:k_num
        label1(selected_nodes(j), i) = 1;
    end
end
label1_no_ref = label1(2:end, :);

desired_L2_norm_a = 0.08;

c0 = label1_no_ref .* xn';

a0 = H * c0;

current_L2_norms_a = vecnorm(a0, 2, 1);

scaling_factors = desired_L2_norm_a ./ current_L2_norms_a;

scaling_factors(current_L2_norms_a == 0) = 0;

c = c0 .* scaling_factors;
c_du = c * (180 / pi);
xn_du = xn * (180 / pi);

a = H * c;

new_L2_norms_a = vecnorm(a, 2, 1);

disp('Adjusted matrix a L2 norms for each column:');
disp(new_L2_norms_a);

rows_to_protect = [1,4, 7, 11, 21, 22, 24, 26, 34, 36, 37, 39, 40, 45, 46, 48];

a(rows_to_protect, :) = 0;

za_label = a; 
za_label(za_label ~= 0) = 1; 
za_label(rows_to_protect, :) = 0;

za = zn + a';

za_full = Z';

for i = 1:attack_num
    za_full(index(i), :) = za(i, :);
end

xa = A \ H' * W * za_full';
xa_du = xa * (180 / pi);

Z_est = H * xa;

residuals = za_full' - Z_est;

std_dev2 = ones(size(residuals)) * 100;

normalized_residuals = abs(residuals ./ std_dev2);

max_normalized_residual = max(normalized_residuals);

C_label = zeros(57, 70080);
Z_label = zeros(137, 70080);

sigle_label = zeros(1, 70080);

for s = 1:attack_num
    Z1(index(s), :) = za(s, :);
    x(:, index(s)) = xa(:, s);
    C_label(:, index(s)) = label1(:, s);
    Z_label(:, index(s)) = za_label(:, s);
    sigle_label(:, index(s)) = 1;
end

measurements = Z1;
measurements_label = Z_label';
measurements2 = measurements(1:70080, :);

sig_label = sigle_label';
sig_label2 = sig_label(1:70080, :);

x_sta = x';
x_sta_label = C_label';

X = measurements2;
Y_binary = sig_label2;
Y_multi = Z_label';

save('combined_data.mat', 'X', 'Y_binary', 'Y_multi');
