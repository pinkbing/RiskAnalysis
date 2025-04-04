clear; close all; clc;

%**************************************************%
%%%%%%%% BOTTOM-UP APPROACH TO PORTFOLIO RISK %%%%%%
%%%%%%%% BOOTSTRAP SIMULATION N-PERIODS     %%%%%%%%%
%**************************************************%

%% ============================
%  Load Data and Define Market
% ============================
imgDir = 'Images/'; % Directory for saving figures

% Ensure directories exist
if ~exist(imgDir, 'dir'), mkdir(imgDir); end

% Compute Asset Log-Returns for a selected asset
logRet = readmatrix('log_return.csv'); % Log returns
T = size(logRet, 1); % Number of time periods
[NObs, NAsset] = size(logRet); % Number of observations and assets

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Bootstrap Estimates for N-Days VaR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ndays = 50; % VaR horizon (in days)
Nb = 1000; % Number of bootstrap samples
alpha = 0.99; % Confidence level

% Preallocate arrays for bootstrap results
ProbNdays = zeros(Nb, Ndays); 

% Repeat Bootstrap simulations
for i = 1:Nb
    % Simulate T x Ndays cumulative returns
    U = randi(T, T, Ndays); % Random indices for bootstrapping
    simLogRetT = cumsum(logRet(U), 2); % Cumulative returns for each horizon
    
    for j = 1:Ndays
        ProbNdays(i,j) = sum(simLogRetT(:,j)<-0.05) / NObs;
    end 
end

% Bootstrap estimates and confidence intervals
ProbBNdays = mean(ProbNdays); % Mean probability for each horizon


% Create a table for results
Bootstrap = table((1:Ndays)', ProbBNdays', ...
    'VariableNames', {'Horizon', 'Return Probability < -0.05'});
disp(Bootstrap); % Display table

% Save results to a text file
writetable(Bootstrap, 'bootstrap_table.txt', 'Delimiter', 'tab');

%% ============================
%  Plot Bootstrap Results
% ============================
% Plot Result over the horizon
h1 = figure('Color', [1 1 1]);
plot(1:Ndays, ProbBNdays, 'g*', 'LineWidth', 1.5); % VaR estimates
hold on;

xlabel('Horizon (days)', 'Interpreter', 'latex');
title('Bootstrap Estimates of 50-days Return Probability Less Than -0.05', 'Interpreter', 'latex');
legend('Probability', ...
    'Location', 'best', 'Interpreter', 'latex');
saveas(h1, fullfile(imgDir,'bootstrap_ndays.png'));


% % MONTE CARLO AND THEORETICAL

% Load log returns from CSV file
log_returns = readmatrix('log_return.csv');

% Estimate mean (mu) and standard deviation (sigma)
mu_hat = mean(log_returns);  % Mean daily return
sigma_hat = std(log_returns); % Standard deviation of daily returns

% Define parameters
Ndays = 50;  % Horizon from 1 to 50 days
NObs = length(log_returns); % Number of observations
Nsim = 10000;  % Number of simulations
loss_threshold = -0.05; % -5% loss threshold

% Initialize probability arrays
prob_theoretical = zeros(1, Ndays);
prob_monte_carlo = zeros(1, Ndays);

% Compute probabilities for each horizon
for H = 1:Ndays
    % Theoretical probability using Gaussian CDF
    mu_H = H * mu_hat;
    sigma_H = sqrt(H) * sigma_hat;
    prob_theoretical(H) = normcdf(loss_threshold, mu_H, sigma_H);
    
    % Monte Carlo Simulation: Generate random samples from N(mu, sigma) -
    % 10,000 simulations
    simulated_returns = sum(normrnd(mu_H, sigma_H, [Nsim, 1]), 2);
    prob_monte_carlo(H) = mean(simulated_returns < loss_threshold);
    
end

% Plot results
h2 = figure;
plot(1:Ndays, prob_theoretical, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'Color', 'r'); hold on;
plot(1:Ndays, prob_monte_carlo, '--s', 'LineWidth', 2, 'MarkerSize', 6, 'Color', 'b');
xlabel('Horizon (Days)');
ylabel('Probability of Losing More than 5%');
title('Comparison of Theoretical & Monte Carlo Probabilities');
legend('Theoretical (Gaussian CDF)', 'Monte Carlo Simulation' );
grid on;
hold off;
saveas(h2, fullfile(imgDir,'theoretical_vs_montecarlo.png'));

