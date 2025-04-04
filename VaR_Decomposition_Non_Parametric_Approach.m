% ============================
% RISK ANALYSIS Q2
%
% NON-PARAMETRIC APPROACH: HISTORICAL SIMULATION
% ============================

%%
clear; close all; clc;
%% ============================
% Load Data and Calculate Log Return
% ============================

filename = 'stock_price.csv'; 

img_directory = 'Images/'; % Directory for saving figures (must exist)
txt_directory  = "Results/"; % Directory for saving results 

%Check if imgdirectory and txt_directory  exist, otherwise create them
get_check_directory(img_directory);
get_check_directory(txt_directory);

% Load dataset
dataset = readtable(filename, 'MissingRule', 'omitrow');
ColLabels = dataset.Properties.VariableNames;
Tickers = ColLabels(2:end); % Extract tickers (6)
HistPrices = dataset{:, 2:end}; % Historical prices
HistDates = dataset{:, 1}; % Historical dates

[NObs, NAsset] = size(HistPrices);

split_idx = floor(NObs/2); %divide equally

% Split dataset
first_half = HistPrices(1:split_idx, :);
second_half = HistPrices(split_idx + 1:end, :);

%% Compute Component VaR - Non-Parametric Approach

% Compute Asset Log-Returns
LogRet1 = log(first_half(2:end, :) ./ first_half(1:end-1, :));
Dates1 = HistDates(1:split_idx);

% Confidence Level
alpha = 0.95;

%% Equally weighted Portfolio
w_eq = ones(NAsset, 1) / NAsset;

% Portfolio Return
PortRet1 = LogRet1*w_eq;

% Estimate Mean Vector and Covariance Matrix
MeanV = mean(LogRet1)';
Sigma = cov(LogRet1);

% % Compute portfolio variance
% sg2p = w_eq' * Sigma * w_eq;

% Compute portfolio VaR 
[VaR_hs_eq, ES_hs_eq] = get_riskmeasures('NP', PortRet1, alpha); % Non-parametric VaR and ES

% Compute Marginal VaR (MVaR)
eps = 0.1 * VaR_hs_eq; % Error margin for VaR interval
lowerBound = -VaR_hs_eq - eps; % Lower bound for VaR interval
upperBound = -VaR_hs_eq + eps; % Upper bound for VaR interval
pos = find((PortRet1 >= lowerBound) & (PortRet1 <= upperBound)); % Positions within VaR interval

% Extract simulations that satisfy the condition
condReturns = LogRet1(pos, :); % Conditional returns

% Compute MVaR: take the conditional mean
MVaR_hs_eq = -mean(condReturns)'; % Marginal VaR, transpose to colum vector

% Compute Component VaR
CVaR_hs_eq = w_eq .* MVaR_hs_eq; % Component VaR

% Compute Component VaR in percentage
CVaR_hs_eq_p = CVaR_hs_eq / sum(CVaR_hs_eq); % Normalized Component VaR
checkVaR = [sum(CVaR_hs_eq), VaR_hs_eq]; % Check consistency

% Plot CVaR contributions
threshold = 0.05; % Threshold for identifying hot spots
X = categorical(Tickers); % Ticker names for plotting

h1 = figure('Color', [1 1 1]);
bar(X, CVaR_hs_eq); % Bar plot of CVaR contributions
hold on;
bar(X, CVaR_hs_eq .* (CVaR_hs_eq > threshold), 'r'); % Highlight contributions above threshold
yline(threshold, 'r', 'LineWidth', 1.5); % Threshold line
xlabel('Stock', 'Interpreter', 'latex');
ylabel('CVaR\%', 'Interpreter', 'latex');
title('Identifying Hot Spots (CVaR)', 'Interpreter', 'latex');

% Create table for Component VaR results
T_Component_hs = table(w_eq, MVaR_hs_eq, CVaR_hs_eq, CVaR_hs_eq_p);
T_Component_hs.Properties.VariableNames = {'Weights', 'MVaR', 'CVaR', 'CVaR%'};
T_Component_hs.Properties.RowNames = Tickers;

h = figure()
StockNames = categorical(Tickers);
bar(StockNames, CVaR_hs_eq_p);
title('Equally Weighted Portfolio CVaR(\%) - Historical Simulation', 'Interpreter', 'latex'); 
xlabel('Stocks', 'Interpreter', 'latex'); 
ylabel('\%', 'Interpreter', 'latex');


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Handle functions %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sg2p = @(x, Sigma) x' * Sigma * x;
VaR_hs = @(x, LogRet) -quantile(LogRet*x, 1 - alpha);
MVaR_hs = @(x, VaR_hs, LogRet) ...
    -mean(LogRet((LogRet * x >= -VaR_hs - 0.1*VaR_hs) & ...
                 (LogRet * x <= -VaR_hs + 0.1*VaR_hs), :))';
CVaR_hs = @(x, VaR_hs, LogRet) x .* MVaR_hs(x, VaR_hs, LogRet);
Conv_hs = @(w, VaR_hs, LogRet) ...
    sqrt(w.^2 .* var(LogRet, 0, 1) - (w' .* MVaR_hs(w, VaR_hs, LogRet)).^2);
Divers = @(x, Sigma) x'*diag(Sigma).^0.5/sqrt(sg2p(x, Sigma));

%% 1. Equally Weighted Portfolio
Conv_hs_eq = Conv_hs(w_eq, VaR_hs_eq, LogRet1);

%% 2. Risk Parity Portfolio
x0 = w_eq;
w_hs_rp = fmincon(@(x) std(CVaR_hs(x, VaR_hs(x, LogRet1), LogRet1)), x0, [], [], ones(1, NAsset), 1, zeros(NAsset, 1), ones(NAsset, 1));

VaR_HS_rp = -quantile(LogRet1*w_hs_rp, 1-alpha);
eps = 0.1 * VaR_HS_rp; % Error margin for VaR interval
lowerBound_rp = -VaR_HS_rp - eps; % Lower bound for VaR interval
upperBound_rp = -VaR_HS_rp + eps; % Upper bound for VaR interval
PortRet_rp = LogRet1*w_hs_rp;
pos_rp = find((PortRet_rp >= lowerBound_rp) & (PortRet_rp <= upperBound_rp)); % Positions within VaR interval

% Extract simulations that satisfy the condition
condReturns_rp = LogRet1(pos_rp, :); % Conditional returns

% Compute MVaR: take the conditional mean
MVaR_HS_rp = -mean(condReturns_rp)'; % Marginal VaR, transpose to colum vector
CVaR_HS_rp = w_hs_rp .* MVaR_HS_rp;
CVaR_HS_rp_p = CVaR_HS_rp / sum(CVaR_HS_rp);

% Compute Component VaR

sg2_hs_rp = sg2p(w_hs_rp, Sigma);
VaR_hs_rp = VaR_hs(w_hs_rp, LogRet1);
MVaR_hs_rp = MVaR_hs(w_hs_rp, VaR_hs_rp, LogRet1);
CVaR_hs_rp = CVaR_hs(w_hs_rp, VaR_hs_rp, LogRet1);
CVaR_hs_rp_p = CVaR_hs_rp / sum(CVaR_hs_rp);
Conv_hs_rp = Conv_hs(w_hs_rp, VaR_hs_eq, LogRet1);


%% 3. Maximum Diversification Portfolio

x0 = w_eq;
w_md = fmincon(@(x) -Divers(x, Sigma), x0, [], [], ones(1, NAsset), 1, zeros(NAsset,1), ones(NAsset,1));
sg2md = sg2p(w_md, Sigma);
MVaR_hs_md = MVaR_hs(w_md, VaR_hs_eq, LogRet1);
CVaR_hs_md = CVaR_hs(w_md, VaR_hs_eq, LogRet1);
CVaR_hs_md_p = CVaR_hs_md / sum(CVaR_hs_md);
Conv_hs_md = Conv_hs(w_md, VaR_hs_eq, LogRet1);
Corr_hs_md = Sigma*w_md./(diag(Sigma)*sg2md).^0.5;

%% Tables for the report
T_w = table(w_eq, w_hs_rp, w_md);
T_w.Properties.VariableNames = {'EQ', 'RP', 'MaxDiv'};
T_w.Properties.RowNames = Tickers;

T_MRisk = table(MVaR_hs_eq, MVaR_HS_rp, MVaR_hs_md);
T_MRisk.Properties.VariableNames = {'EQ', 'RP', 'MaxDiv'};
T_MRisk.Properties.RowNames = Tickers;

T_CRisk = table(CVaR_hs_eq_p, CVaR_hs_rp_p, CVaR_hs_md_p);
T_CRisk.Properties.VariableNames = {'EQ', 'RP', 'MaxDiv'};
T_CRisk.Properties.RowNames = Tickers;

T_Conv = table(Conv_hs_eq, Conv_hs_rp, Conv_hs_md);
T_Conv.Properties.VariableNames = {'EQ', 'RP', 'MaxDiv'};
T_Conv.Properties.RowNames = Tickers;

% Plot portfolio weights
h = figure('Color', [1 1 1]);
StockNames = categorical(Tickers);
bar(StockNames, [w_eq, w_hs_rp, w_md]);
legend('Equally Weighted', 'Risk Parity', 'Max Div', 'interpreter', 'latex', 'location', 'northwest');
title('Portfolio weights (Non-Parametric Approach)', 'interpreter', 'latex');
print(h, '-dpng', fullfile(img_directory, 'Q2_HS_Weights_EQ_RP_MD'))

% Plot Marginal VaR
h = figure('Color', [1 1 1]);
bar(StockNames, [MVaR_hs_eq, MVaR_HS_rp, MVaR_hs_md]);
legend('Equally Weighted ', 'Risk Parity', 'Max Div', 'interpreter', 'latex', 'location', 'northwest');
title('Marginal VaR (Non-Parametric Approach)', 'interpreter', 'latex');
print(h, '-dpng', fullfile(img_directory, 'Q2_HS_MVaR_EQ_RP_MD'))

% Plot Component VaR%
h = figure('Color', [1 1 1]);
bar(StockNames, [CVaR_hs_eq_p, CVaR_HS_rp_p, CVaR_hs_md_p]);
legend('Equally Weighted', 'Risk Parity', 'Max Div', 'interpreter', 'latex', 'location', 'northwest');
title('Component VaR\% (Non-Parametric Approach)', 'interpreter', 'latex');
print(h, '-dpng', fullfile(img_directory, 'Q2_HS_CVaR_EQ_RP_MD'))


%% Performance Evaluation

LogRet2 = log(second_half(2:end, :) ./ second_half(1:end-1, :));

PortRet_hs_eq = LogRet2 * w_eq;
PortRet_hs_rp = LogRet2 * w_hs_rp;
PortRet_hs_md = LogRet2 * w_md;

% 1) Sharpe Ratio

SR_eq = mean(PortRet_hs_eq) / std(PortRet_hs_eq);
SR_rp = mean(PortRet_hs_rp) / std(PortRet_hs_rp);
SR_md = mean(PortRet_hs_md) / std(PortRet_hs_md);

% 2) Maximum Dradwdown

% Cumulative Return
cum_ret_hs_eq = exp(cumsum(PortRet_hs_eq));
cum_ret_hs_rp = exp(cumsum(PortRet_hs_rp));
cum_ret_hs_md = exp(cumsum(PortRet_hs_md));

% Maximum Drawdown
maxDD_eq = maxdrawdown(cum_ret_hs_eq);
maxDD_rp = maxdrawdown(cum_ret_hs_rp);
maxDD_md = maxdrawdown(cum_ret_hs_md);

% 3) VaR violations

% Non-Parametric approach
[VaR_hs_eq2, ES_hs_eq2] = get_riskmeasures('NP', PortRet_hs_eq, alpha); 
[VaR_hs_rp2, ES_hs_rp2] = get_riskmeasures('NP', PortRet_hs_rp, alpha);
[VaR_hs_md2, ES_hs_md2 ] = get_riskmeasures('NP', PortRet_hs_md, alpha);


violation_eq = sum(PortRet_hs_eq < -VaR_hs_eq2);
violation_rp = sum(PortRet_hs_rp < -VaR_hs_rp2);
violation_md = sum(PortRet_hs_md < -VaR_hs_md2);

T_Violation_g = table(violation_eq, violation_rp, violation_md);
T_Violation_g.Properties.VariableNames = {'E.Q.', 'R.P.', 'M.D'};
T_Violation_g.Properties.RowNames = "95\% VaR Violation";

% 4) ES

T_ES_hs = table(ES_hs_eq2, ES_hs_rp2, ES_hs_md2);
T_ES_hs.Properties.VariableNames = {'E.Q.', 'R.P.', 'M.D'};
T_ES_hs.Properties.RowNames = "Expected Shotfall";

% 5) Sortino Ratio

neg_ret_eq = PortRet_hs_eq(PortRet_hs_eq<0);
neg_ret_rp = PortRet_hs_rp(PortRet_hs_rp<0);
neg_ret_md = PortRet_hs_md(PortRet_hs_md<0);

downside_vol_eq = std(neg_ret_eq);
downside_vol_rp = std(neg_ret_rp);
downside_vol_md = std(neg_ret_md);

sortino_ratio_eq = mean(PortRet_hs_eq)/downside_vol_eq;
sortino_ratio_rp = mean(PortRet_hs_rp)/downside_vol_rp;
sortino_ratio_md = mean(PortRet_hs_md)/downside_vol_md;

T_Sortino_hs = table(sortino_ratio_eq, sortino_ratio_rp, sortino_ratio_md);
T_Sortino_hs.Properties.VariableNames = {'E.Q.', 'R.P.', 'M.D'};
T_Sortino_hs.Properties.RowNames = "Sortino Ratio";


% 6) Calmar Ratio
ann_return_hs_eq = mean(PortRet_hs_eq) * 252;
ann_return_hs_rp = mean(PortRet_hs_rp) * 252;
ann_return_hs_md = mean(PortRet_hs_md) * 252;

Calmar_hs_eq = ann_return_hs_eq / abs(maxDD_eq);
Calmar_hs_rp = ann_return_hs_rp / abs(maxDD_rp);
Calmar_hs_md = ann_return_hs_md / abs(maxDD_md);

T_Calmar_hs = table(Calmar_hs_eq, Calmar_hs_rp, Calmar_hs_md);
T_Calmar_hs.Properties.VariableNames = {'E.Q.', 'R.P.', 'M.D'};
T_Calmar_hs.Properties.RowNames = "Calmar Ratio";

%% ============================
% Portfolio Performance Summary Table
% ============================

% Define portfolio names
portfolios = {'Equally Weighted'; 'Risk Parity'; 'Max Diversification'};

% Create Performance Table (Ensure Portfolio is a Column, Not RowNames)
Performance_Table = table(...                                
    portfolios, ...                                 % Portfolio Names
    [SR_eq; SR_rp; SR_md], ...                     % Sharpe Ratios
    [maxDD_eq; maxDD_rp; maxDD_md], ...            % Maximum Drawdowns
    [violation_eq; violation_rp; violation_md], ...% VaR Violations
    [ES_hs_eq2; ES_hs_rp2; ES_hs_md2], ...         % Expected Shortfall
    [sortino_ratio_eq; sortino_ratio_rp; sortino_ratio_md], ...
    [Calmar_hs_eq; Calmar_hs_rp; Calmar_hs_md], ...
    'VariableNames', {'Portfolio', 'Sharpe_Ratio', 'Max_Drawdown', 'VaR_Violations','Expected_Shortfall','Sortino_Ratio', 'Calmar_Ratio'} ...
);

% Display the Summary Table
disp('Portfolio Performance Summary:');
disp(Performance_Table);

% === Update filename ===
txtfilename = txt_directory + "Q2_Non_Parametric_Approach.txt";

% === Write Table to File ===
log_to_file("# ========================================================", txtfilename);
log_to_file("Non-Parametric Approach", txtfilename);
log_to_file("# ========================================================", txtfilename);

log_to_file("--------------------------", txtfilename);
log_to_file("Portfolio Performance Summary", txtfilename);

% === Fix Table Header Formatting ===
table_header = sprintf('%-25s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
    'Portfolio', 'Sharpe Ratio', 'Max Drawdown', 'VaR Violations', 'Expected Shortfall', 'Sortino_Ratio','Calmar Ratio');
log_to_file(table_header, txtfilename);

% === Fix Row Formatting ===
for i = 1:height(Performance_Table)
    row_text = sprintf('%-25s %-12.5f %-12.5f %-4d %-12.5f %-12.5f %-12.5f\n', ...
        Performance_Table.Portfolio{i}, ...
        Performance_Table.Sharpe_Ratio(i), ...
        Performance_Table.Max_Drawdown(i), ...
        Performance_Table.VaR_Violations(i), ...
        Performance_Table.Expected_Shortfall(i), ...
        Performance_Table.Sortino_Ratio(i), ...
        Performance_Table.Calmar_Ratio(i));
    
    log_to_file(row_text, txtfilename);
end

log_to_file("# ========================================================", txtfilename);
