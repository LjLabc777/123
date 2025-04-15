clear;
clc;

file_path = 'standardized_data_combined.xlsx';
standardized_data = readtable(file_path);
standardized_data = table2array(standardized_data);

T = size(standardized_data, 1) * size(standardized_data, 2);

batch_size = 1000;
num_batches = ceil(T / batch_size);

mpc = case57;
num_buses = size(mpc.bus, 1);
num_branches = size(mpc.branch, 1);

measured_values_all = struct('bus_Pinj', zeros(num_buses, T), ...
                             'branch_Pinj', zeros(num_branches, T));

state_values_all = struct('bus_Va', zeros(num_buses, T));

standardized_data_flattened = reshape(standardized_data', 1, T);

for batch = 1:num_batches
    start_idx = (batch - 1) * batch_size + 1;
    end_idx = min(batch * batch_size, T);

    fprintf('Starting batch %d, time steps %d to %d...\n', batch, start_idx, end_idx);

    try
        current_batch_data = standardized_data_flattened(start_idx:end_idx);

        [measured_values_batch, state_values_batch] = run_time_series_simulation_dc(current_batch_data, end_idx - start_idx + 1);

        measured_values_all.bus_Pinj(:, start_idx:end_idx) = measured_values_batch.bus_Pinj;
        measured_values_all.branch_Pinj(:, start_idx:end_idx) = measured_values_batch.branch_Pinj;
        state_values_all.bus_Va(:, start_idx:end_idx) = state_values_batch.bus_Va;

        fprintf('Batch %d completed and results saved.\n', batch);
    catch ME
        fprintf('Error in batch %d: %s\n', batch, ME.message);
    end
end

save('measured_values_all_dc.mat', 'measured_values_all');
save('state_values_all_dc.mat', 'state_values_all');

fprintf('All batches completed. Results saved.\n');

function [measured_values, state_values] = run_time_series_simulation_dc(standardized_data, T)
    mpc = case57;

    load_bus_indices = find(mpc.bus(:, 3) > 0);

    num_load_buses = length(load_bus_indices);

    original_Pd = mpc.bus(load_bus_indices, 3);
    original_Pg = mpc.gen(:, 2);

    Pmin = mpc.gen(:, 10);
    Pmax = mpc.gen(:, 9);

    q = 0.1;
    sigma_s = 0.03;

    num_branches = size(mpc.branch, 1);
    num_buses = size(mpc.bus, 1);

    measured_values = struct('bus_Pinj', zeros(num_buses, T), ...
                             'branch_Pinj', zeros(num_branches, T));
                             
    state_values = struct('bus_Va', zeros(num_buses, T));

    for t = 1:T
        try
            mean_xi_Pd = 1 + q * standardized_data(t);
            xi_Pd = normrnd(mean_xi_Pd, sigma_s, [num_load_buses, 1]);

            mpc.bus(load_bus_indices, 3) = original_Pd .* xi_Pd;

            xi_Pg = max(normrnd(1, sigma_s, [size(mpc.gen, 1), 1]), 0.1);

            Pg_adjusted = original_Pg .* xi_Pg;
            mpc.gen(:, 2) = min(max(Pg_adjusted, Pmin), Pmax);

            results = rundcpf(mpc);

            bus_Pg = zeros(num_buses, 1);
            bus_Pg(mpc.gen(:, 1)) = results.gen(:, 2);
            measured_values.bus_Pinj(:, t) = bus_Pg - results.bus(:, 3);
            measured_values.branch_Pinj(:, t) = results.branch(:, 14);
            state_values.bus_Va(:, t) = results.bus(:, 9);

        catch ME
            fprintf('Error at time step %d: %s\n', t, ME.message);
            continue;
        end
    end
end
