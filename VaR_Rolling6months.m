%% ============================
% Q1.2
% ============================

clear; close all; clc;
format short;

%% ============================
%  Load Data and Define Market
% ============================

filename = 'stock_price.csv'; 

img_directory = 'Images/'; % Directory for saving figures (must exist)
txt_directory  = "Results/"; % Directory for saving results 

%Check if imgdirectory and txt_directory  exist, otherwise create them
get_check_directory(img_directory)
get_check_directory(txt_directory)

% Load dataset
dataset = readtable(filename, 'MissingRule', 'omitrow');
ColLabels = dataset.Properties.VariableNames;
Tickers = ColLabels(2:end); % Extract tickers (6)
HistPrices = dataset{:, 2:end}; % Historical prices
HistDates = dataset{:, 1}; % Historical dates

[NObs, NAsset] = size(HistPrices);


% Compute Asset Log-Returns
LogRet = log(HistPrices(2:end, :) ./ HistPrices(1:end-1, :));
Dates = HistDates(2:end);

% Assign Portfolio Composition: Equally Weighted
w_eq = ones(NAsset, 1) / NAsset;
PortRet = LogRet * w_eq;
nobs = length(PortRet);


%% ============================
% 6months rolling, starting on 1st July 2014
% ============================

window = 120; %assumption 20 trading days per month, 6x20 = 120days
startDate = datetime('01-Jul-2014', 'InputFormat', 'dd-MMM-yyyy');

startIndex = find(Dates >= startDate, 1);
rolling6m = zeros(length(Dates) - startIndex + 1, 10);

% Confidence Intervals
ConfLevel = [0.9, 0.99]; 

for j = startIndex:length(Dates); % j starts from 1st June
    data6m = PortRet(j-window+1:j);
    mean6m = mean(data6m);
    sigma6m = std(data6m);
    
   
    % Parametric gaussian
    [VaR_p_90, ES_p_90] = get_VaR('G', data6m, ConfLevel(1,1));
    [VaR_p_99, ES_p_99] = get_VaR('G', data6m, ConfLevel(1,2));
    [VaR_np_90, ES_np_90] = get_VaR('NP', data6m, ConfLevel(1,1));
    [VaR_np_99, ES_np_90] = get_VaR('NP', data6m, ConfLevel(1,2));
    [VaR_boot_90, ES_boot_90] = get_VaR('BOOT', data6m, ConfLevel(1,1));
    [VaR_boot_99, ES_boot_99] = get_VaR('BOOT', data6m, ConfLevel(1,2));
    [VaR_mc_90, ES_mc_90] = get_VaR('MC', data6m, ConfLevel(1,1));
    [VaR_mc_99, ES_mc_99] = get_VaR('MC', data6m, ConfLevel(1,2));
    [VaR_ewma_90, ES_ewma_90] = get_VaR('EWMA', data6m, ConfLevel(1,1));
    [VaR_ewma_99, ES_ewma_99] = get_VaR('EWMA', data6m, ConfLevel(1,2));

    % For 90% CI
    rolling6m(j-startIndex+1, 1) = VaR_p_90;
    rolling6m(j-startIndex+1, 2) = VaR_p_99;
    rolling6m(j-startIndex+1, 3) = VaR_np_90;
    rolling6m(j-startIndex+1, 4) = VaR_np_99;
    rolling6m(j-startIndex+1, 5) = VaR_boot_90;
    rolling6m(j-startIndex+1, 6) = VaR_boot_99;
    rolling6m(j-startIndex+1, 7) = VaR_mc_90;
    rolling6m(j-startIndex+1, 8) = VaR_mc_99;
    rolling6m(j-startIndex+1, 9) = VaR_ewma_90;
    rolling6m(j-startIndex+1, 10) = VaR_ewma_99;

end


%% ============================
%  Plot Rolling 6-Month VaR Estimates 
% ============================

% 90% Confidence Level
figure;
hold on;
plot(Dates(startIndex:end), rolling6m(:,1), 'm', 'LineWidth', 1.5); % VaR Parametric 90%
plot(Dates(startIndex:end), rolling6m(:,3), 'b', 'LineWidth', 1.5); % VaR Non-Parametric 90%
plot(Dates(startIndex:end), rolling6m(:,5), 'g--', 'LineWidth', 1.2); % VaR Bootstrap 90%
plot(Dates(startIndex:end), rolling6m(:,7), 'k--', 'LineWidth', 1.2); % VaR Monte Carlo 90%
plot(Dates(startIndex:end), rolling6m(:,9), 'c--', 'LineWidth', 1.2); % VaR EWMA 90%

xlabel('Date');
ylabel('Value at Risk (VaR)');
title('Rolling 6-Month VaR (90% Confidence Level)');
legend({'Gaussian', 'Historical Simulation', 'Bootstrap', 'Monte Carlo', 'EWMA'}, 'Location', 'best');
datetick('x', 'yyyy', 'keeplimits'); % Format x-axis as years
grid on;
hold off;

saveas(gcf, fullfile(img_directory, 'Q1_2_Rolling6m_VaR_90_5.png'));
disp('Rolling 6-month VaR (90%) plot saved.');

figure;
hold on;
plot(Dates(startIndex:end), rolling6m(:,1), 'm', 'LineWidth', 2); % VaR Parametric 90%
plot(Dates(startIndex:end), rolling6m(:,3), 'b', 'LineWidth', 2); % VaR Non-Parametric 90%
plot(Dates(startIndex:end), rolling6m(:,5), 'g--', 'LineWidth', 1.2); % VaR Bootstrap 90%
plot(Dates(startIndex:end), rolling6m(:,7), 'k--', 'LineWidth', 1.2); % VaR Monte Carlo 90%

xlabel('Date');
ylabel('Value at Risk (VaR)');
title('Rolling 6-Month VaR (90% Confidence Level)');
legend({'Gaussian', 'Historical Simulation', 'Bootstrap', 'Monte Carlo'}, 'Location', 'best');
datetick('x', 'yyyy', 'keeplimits'); % Format x-axis as years
grid on;
hold off;


% 99% Confidence Level
figure;
hold on;
plot(Dates(startIndex:end), rolling6m(:,2), 'm', 'LineWidth', 1.5); % VaR Parametric 99%
plot(Dates(startIndex:end), rolling6m(:,4), 'b', 'LineWidth', 1.5); % VaR Non-Parametric 99%
plot(Dates(startIndex:end), rolling6m(:,6), 'g--', 'LineWidth', 1.2); % VaR Bootstrap 99%
plot(Dates(startIndex:end), rolling6m(:,8), 'k--', 'LineWidth', 1.2); % VaR Monte Carlo 99%
plot(Dates(startIndex:end), rolling6m(:,10), 'c--', 'LineWidth', 1.2); % VaR EWMA 90%

xlabel('Date');
ylabel('Value at Risk (VaR)');
title('Rolling 6-Month VaR (99% Confidence Level)');
legend({'Gaussian', 'Historical Simulation', 'Bootstrap', 'Monte Carlo', 'EWMA'}, 'Location', 'best');
datetick('x', 'yyyy', 'keeplimits'); % Format x-axis as years
grid on;
hold off;

figure;
hold on;
plot(Dates(startIndex:end), rolling6m(:,2), 'm', 'LineWidth', 2); % VaR Parametric 99%
plot(Dates(startIndex:end), rolling6m(:,4), 'b', 'LineWidth', 2); % VaR Non-Parametric 99%
plot(Dates(startIndex:end), rolling6m(:,6), 'g--', 'LineWidth', 1.2); % VaR Bootstrap 99%
plot(Dates(startIndex:end), rolling6m(:,8), 'k--', 'LineWidth', 1.2); % VaR Monte Carlo 99%

xlabel('Date');
ylabel('Value at Risk (VaR)');
title('Rolling 6-Month VaR (99% Confidence Level)');
legend({'Gaussian', 'Historical Simulation', 'Bootstrap', 'Monte Carlo'}, 'Location', 'best');
datetick('x', 'yyyy', 'keeplimits'); % Format x-axis as years
grid on;
hold off;


saveas(gcf, fullfile(img_directory, 'Q1_2_Rolling6m_VaR_99_5.png'));
disp('Rolling 6-month VaR (99%) plot saved.');

%% ============================
%  Q1.3 Compute VaR Violations
% ============================

% Initialize violation counts
VaR_violations = zeros(4,2); % 4 models x 2 confidence levels (90% & 99%)
total_observations = length(rolling6m); % Total number of VaR forecasts


for model_idx = 1:5 % 4 models: Parametric, Non-Parametric, Bootstrap, Monte Carlo
    for conf_idx = 1:2 % 90% and 99%
        % Get VaR estimates
        VaR_estimate = rolling6m(:, (model_idx - 1) * 2 + conf_idx);
        VaR_violations(model_idx, conf_idx) = sum(PortRet(startIndex:end) < -VaR_estimate);
        
    end
end


VaR_violation_pct = (VaR_violations / total_observations) * 100;

% Display results
Models = {'Gaussian', 'Historical Simulation', 'Bootstrap', 'Monte Carlo', 'EWMA'};
ConfLevels = {'90%', '99%'};

ViolationTable = array2table(VaR_violations, 'RowNames', Models, 'VariableNames', ConfLevels);
ViolationPctTable = array2table(VaR_violation_pct, 'RowNames', Models, 'VariableNames', ConfLevels);

disp('Number of VaR Violations:');
disp(ViolationTable);

disp('Percentage of VaR Violations:');
disp(ViolationPctTable);

% 
% txtfilename = txt_directory + "Q1_3_VaR_Violations.txt";
% log_to_file("# ========================================================", txtfilename);
% log_to_file(strjoin(["Rolling 6m Portfolio VaR Violations"], ''), txtfilename);
% log_to_file("# ========================================================", txtfilename);
% 
% % number of violations
% log_to_file("Number of VaR Violations:", txtfilename);
% log_to_file(strjoin(["Model", "90%", "99%"], '   '), txtfilename);
% log_to_file("--------------------------------------------------------", txtfilename);
% 
% for i = 1:length(Models)
%     log_to_file(strjoin([Models{i}, num2str(VaR_violations(i,1)), num2str(VaR_violations(i,2))], '   '), txtfilename);
% end
% 
% % Log percentage of violations
% log_to_file("", txtfilename);
% log_to_file("Percentage of VaR Violations:", txtfilename);
% log_to_file(strjoin(["Model", "90%", "99%"], '   '), txtfilename);
% log_to_file("--------------------------------------------------------", txtfilename);
% 
% for i = 1:length(Models)
%     log_to_file(strjoin([Models{i}, sprintf('%.2f', VaR_violation_pct(i,1)), sprintf('%.2f', VaR_violation_pct(i,2))], '   '), txtfilename);
% end
% 


%% ============================
%  Q1.4 Kupiec Test 
% ============================

expected_violations = total_observations * (1 - ConfLevel);

% initialization, 4 models x 2 conf levels
LR_uc = zeros(5,2);
p_values = zeros(5,2);
reject_H0 = zeros(5,2);

chi2_critical = chi2inv(0.95, 1); 

for model_idx = 1:5 % 5 models: Parametric, Non-Parametric, Bootstrap, Monte Carlo, EWMA
    for conf_idx = 1:2 % 90% and 99% 
        VaR_estimate = rolling6m(:, (model_idx - 1) * 2 + conf_idx);
        VaR_violations(model_idx, conf_idx) = sum(PortRet(startIndex:end) < -VaR_estimate);
        [LR_uc(model_idx, conf_idx), p_values(model_idx, conf_idx)] = get_LRuc(1 - ConfLevel(conf_idx), VaR_violations(model_idx, conf_idx), total_observations);
        reject_H0(model_idx, conf_idx) = LR_uc(model_idx, conf_idx) > chi2_critical;
    end
end

KupiecTable = array2table(LR_uc, 'RowNames', Models, 'VariableNames', {'90%', '99%'});
PValueTable = array2table(p_values, 'RowNames', Models, 'VariableNames', {'90%', '99%'});
RejectH0Table = array2table(reject_H0, 'RowNames', Models, 'VariableNames', {'90%', '99%'});

disp('Kupiec Test (LR_uc) Results:');
disp(KupiecTable);

disp('Kupiec Test p-values:');
disp(PValueTable);

disp('H0 Rejection Decision (1 = Reject, 0 = Accept):');
disp(RejectH0Table);


%% ============================
%  Q1.4 Kupiec and Christoffersen Tests
% ============================

% Initialize matrices to store test results
LR_uc = zeros(5,2); % Kupiec Unconditional Coverage Test
p_values_uc = zeros(5,2);
reject_H0_uc = zeros(5,2);

LR_cc = zeros(5,2); % Christoffersen Conditional Coverage Test
p_values_cc = zeros(5,2);
reject_H0_cc = zeros(5,2);

LR_combined = zeros(5,2); % Combined Conditional Coverage Test
p_values_combined = zeros(5,2);
reject_H0_combined = zeros(5,2);

% Initialization
n00 = zeros(5,2); 
n01 = zeros(5,2); 
n10 = zeros(5,2); 
n11 = zeros(5,2);

total_observations = length(rolling6m); % Number of VaR predictions

for model_idx = 1:5 % 5 models: Gaussian, Historical, Bootstrap, Monte Carlo, EWMA
    for conf_idx = 1:2 % 90% and 99% confidence levels
        % Get the VaR estimates for the model at the given confidence level
        VaR_estimate = rolling6m(:, (model_idx - 1) * 2 + conf_idx);
        violations = PortRet(startIndex:end) < -VaR_estimate; % 1 if VaR is violated, 0 otherwise
        num_violations = sum(violations);

        % Kupiec Unconditional Coverage Test
        [LR_uc(model_idx, conf_idx), p_values_uc(model_idx, conf_idx)] = get_LRuc(1 - ConfLevel(conf_idx), num_violations, total_observations);
        reject_H0_uc(model_idx, conf_idx) = LR_uc(model_idx, conf_idx) > chi2inv(0.95, 1);

        % Compute transition counts for Christoffersen Test
        for t = 2:length(violations)
            if violations(t-1) == 0 && violations(t) == 0
                n00(model_idx, conf_idx) = n00(model_idx, conf_idx) + 1;
            elseif violations(t-1) == 0 && violations(t) == 1
                n01(model_idx, conf_idx) = n01(model_idx, conf_idx) + 1;
            elseif violations(t-1) == 1 && violations(t) == 0
                n10(model_idx, conf_idx) = n10(model_idx, conf_idx) + 1;
            elseif violations(t-1) == 1 && violations(t) == 1
                n11(model_idx, conf_idx) = n11(model_idx, conf_idx) + 1;
            end
        end

        % Christoffersen Conditional Coverage Test
        [LR_cc(model_idx, conf_idx), p_values_cc(model_idx, conf_idx)] = get_LRind(n00(model_idx, conf_idx), n01(model_idx, conf_idx), n10(model_idx, conf_idx), n11(model_idx, conf_idx));
        reject_H0_cc(model_idx, conf_idx) = LR_cc(model_idx, conf_idx) > chi2inv(0.95, 1);

        % Combined Conditional Coverage Test (Kupiec + Christoffersen)
        LR_combined(model_idx, conf_idx) = LR_uc(model_idx, conf_idx) + LR_cc(model_idx, conf_idx);
        p_values_combined(model_idx, conf_idx) = 1 - chi2cdf(LR_combined(model_idx, conf_idx), 2);
        reject_H0_combined(model_idx, conf_idx) = LR_combined(model_idx, conf_idx) > chi2inv(0.95, 2);
    end
end


%% ============================
%  Display Test Results
% ============================

% Kupiec Test Results
KupiecTable = array2table(LR_uc, 'RowNames', Models, 'VariableNames', {'90%', '99%'});
PValueTable_UC = array2table(p_values_uc, 'RowNames', Models, 'VariableNames', {'90%', '99%'});
RejectTable_UC = array2table(reject_H0_uc, 'RowNames', Models, 'VariableNames', {'90%', '99%'});

disp('Kupiec Unconditional Coverage Test (LR_uc) Results:');
disp(KupiecTable);
disp('Kupiec Test p-values:');
disp(PValueTable_UC);
disp('H0 Rejection Decision (1 = Reject, 0 = Accept):');
disp(RejectTable_UC);

% Christoffersen Conditional Coverage Test Results
CondCoverageTable = array2table(LR_cc, 'RowNames', Models, 'VariableNames', {'90%', '99%'});
PValueTable_CC = array2table(p_values_cc, 'RowNames', Models, 'VariableNames', {'90%', '99%'});
RejectTable_CC = array2table(reject_H0_cc, 'RowNames', Models, 'VariableNames', {'90%', '99%'});

disp('Christoffersen Conditional Coverage Test (LR_cc) Results:');
disp(CondCoverageTable);
disp('Conditional Coverage Test p-values:');
disp(PValueTable_CC);
disp('H0 Rejection Decision (1 = Reject, 0 = Accept):');
disp(RejectTable_CC);

% Combined Test Results
CombinedTable = array2table(LR_combined, 'RowNames', Models, 'VariableNames', {'90%', '99%'});
PValueTable_Combined = array2table(p_values_combined, 'RowNames', Models, 'VariableNames', {'90%', '99%'});
RejectTable_Combined = array2table(reject_H0_combined, 'RowNames', Models, 'VariableNames', {'90%', '99%'});

disp('Combined Conditional Coverage Test (Kupiec + Christoffersen) Results:');
disp(CombinedTable);
disp('Combined Test p-values:');
disp(PValueTable_Combined);
disp('H0 Rejection Decision (1 = Reject, 0 = Accept):');
disp(RejectTable_Combined);

%% ============================
%  Q1.4 Distributional Tests
% ============================

% Initialize matrices for storing test results
PIT_Values = zeros(length(Dates) - startIndex + 1, 5, 2); % (Observations, Models, Confidence Levels)
KS_pvalues = zeros(5,2);  % Kolmogorov-Smirnov Test
Kuiper_pvalues = zeros(5,2);  % Kuiper's Test

for model_idx = 1:5  % 5 models: Gaussian, Historical, Bootstrap, Monte Carlo, EWMA
    for conf_idx = 1:2  % 90% and 99% confidence levels
        % Get the VaR estimates from rolling6m
        VaR_estimate = rolling6m(:, (model_idx - 1) * 2 + conf_idx);
        
        % Compute Probability Integral Transform (PIT) values
        PIT_Values(:, model_idx, conf_idx) = normcdf(PortRet(startIndex:end), -VaR_estimate, std(PortRet(startIndex:end)));
        
        % Fix numerical issues by clipping values
        PIT_Values(:, model_idx, conf_idx) = max(min(PIT_Values(:, model_idx, conf_idx), 0.999), 0.001);

        % Kolmogorov-Smirnov Test for Uniformity
        [~, KS_pvalues(model_idx, conf_idx)] = kstest(PIT_Values(:, model_idx, conf_idx), 'CDF', makedist('Uniform', 0, 1));
        
        % Kuiper’s Test for Uniformity
        Kuiper_pvalues(model_idx, conf_idx) = kuiper_test(PIT_Values(:, model_idx, conf_idx));
    end
end

%% ============================
%  Display Test Results
% ============================

% Kolmogorov-Smirnov Test Results
KSTable = array2table(KS_pvalues, 'RowNames', Models, 'VariableNames', {'90%', '99%'});
disp('Kolmogorov-Smirnov Test (P-Values for Uniformity):');
disp(KSTable);

% Kuiper's Test Results
KuiperTable = array2table(Kuiper_pvalues, 'RowNames', Models, 'VariableNames', {'90%', '99%'});
disp('Kuiper’s Test (P-Values for Uniformity):');
disp(KuiperTable);


for model_idx = 1:5;  % 5 models: Gaussian, Historical, Bootstrap, Monte Carlo, EWMA
    for conf_idx = 1:2;
    figure
    histogram(PIT_Values(:, model_idx, conf_idx), 20, 'Normalization', 'probability');
    title(['PIT Histogram - Model ', num2str(model_idx), ' at ', num2str(ConfLevel(conf_idx)*100), '% Confidence']);
    xlabel('PIT Values');
    ylabel('Frequency');
    grid on;
    end
end


%% ============================
% %  Save Results to File
% % ============================
% 
% txtfilename = txt_directory + "Q1_Distributional_Tests.txt";
% 
% log_to_file("# ========================================================", txtfilename);
% log_to_file("Distributional Tests for VaR Model Backtesting", txtfilename);
% log_to_file("# ========================================================", txtfilename);
% 
% % Kolmogorov-Smirnov Test
% log_to_file("Kolmogorov-Smirnov Test Results:", txtfilename);
% for i = 1:length(Models)
%     log_to_file(strjoin([Models{i}, num2str(KS_pvalues(i,1)), num2str(KS_pvalues(i,2))], '   '), txtfilename);
% end
% 
% % Kuiper’s Test
% log_to_file("Kuiper’s Test Results:", txtfilename);
% for i = 1:length(Models)
%     log_to_file(strjoin([Models{i}, num2str(Kuiper_pvalues(i,1)), num2str(Kuiper_pvalues(i,2))], '   '), txtfilename);
% end
% 
% % Berkowitz Normality Test
% log_to_file("Berkowitz Normality Test Results:", txtfilename);
% for i = 1:length(Models)
%     log_to_file(strjoin([Models{i}, num2str(Berkowitz_pvalues(i,1)), num2str(Berkowitz_pvalues(i,2))], '   '), txtfilename);
% end
% 
% log_to_file("# ========================================================", txtfilename);
% 
% disp('Distributional test results saved.');

