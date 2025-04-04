% ============================
% RISK ANALYSIS Q2
%
% PARAMETRIC APPROACH
% ============================

%%
clear; close all; clc;
format short;
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

%% Compute Component VaR - Parametric Approach
% 
% Compute Asset Log-Returns
LogRet1 = log(first_half(2:end, :) ./ first_half(1:end-1, :));
Dates1 = HistDates(1:split_idx);

% Confidence Level
alpha = 0.95;

%% Equally weighted Portfolio
w_eq = ones(NAsset, 1) / NAsset;

% Estimate Mean Vector and Covariance Matrix
mu = mean(LogRet1)';
Sigma = cov(LogRet1);

% Compute portfolio return = w * mu and variance = w' * cov_matrix * w
mu_p = w_eq'*mu;
var_p = w_eq' * Sigma * w_eq;
sigma_p = sqrt(var_p); % std dev

% Compute portfolio gaussian/parametric VaR 
z = norminv(1 - alpha,0,1);
VaR_eq = - z * sigma_p; % assume zero mean to avoid underestimating risk

% Compute Marginal VaR = -z * cov*w / sigma_p
MVaR_eq = -z* (Sigma*w_eq) / sigma_p; % assume zero mean to avoid underestimating risk

% Compute Component VaR = w * MVaR
CVaR_eq = w_eq .* MVaR_eq;
chk = [sum(CVaR_eq), VaR_eq]; % must be same value, sum of CVAR = portfolio VAR

% Compute Component VaR in percentage = CVaR/VaR
CVaR_eq_p = CVaR_eq/VaR_eq;

% Create table for Component VaR results
T_Component_g = table(w_eq, MVaR_eq, CVaR_eq, CVaR_eq_p);
T_Component_g.Properties.VariableNames = {'Weights', 'MVaR', 'CVaR', 'CVaR%'};
T_Component_g.Properties.RowNames = Tickers;

h = figure()
StockNames = categorical(Tickers);
bar(StockNames, CVaR_eq_p);
title('Equally Weighted Portfolio CVaR(\%) - Parametric Approach', 'Interpreter', 'latex'); 
xlabel('Stocks', 'Interpreter', 'latex'); 
ylabel('\%', 'Interpreter', 'latex');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Handle functions %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sg2p = @(x, Sigma) x' * Sigma * x;
MVaR = @(x, Sigma) -z*Sigma * x / sqrt(sg2p(x, Sigma));
CVaR = @(x, Sigma) x .* MVaR(x, Sigma);
Conv = @(x, Sigma) sqrt(x.^2 .* diag(Sigma) - (x .* CVaR(x, Sigma)).^2);
Divers = @(x, Sigma) x'*diag(Sigma).^0.5/sqrt(sg2p(x, Sigma));

%% 1. Equally Weighted Portfolio

Conv_eq = Conv(w_eq, Sigma);

%% 2. Risk Parity Portfolio

x0 = w_eq;
w_rp = fmincon(@(x) std(CVaR(x, Sigma)), x0, [], [], ones(1, NAsset), 1, zeros(NAsset, 1), ones(NAsset, 1));
sg2rp = sg2p(w_rp, Sigma);
VaR_rp = -z*sqrt(sg2rp);
MVaR_rp = MVaR(w_rp, Sigma);
CVaR_rp = CVaR(w_rp, Sigma);
CVaR_rp_p = CVaR_rp / sum(CVaR_rp);
Conv_rp = Conv(w_rp, Sigma);

%% 3. Maximum Diversification Portfolio

x0 = w_eq;
w_md = fmincon(@(x) -Divers(x, Sigma), x0, [], [], ones(1, NAsset), 1, zeros(NAsset,1), ones(NAsset,1));
sg2md = sg2p(w_md, Sigma);
VaR_md = -z*sqrt(sg2md);
MVaR_md = MVaR(w_md, Sigma);
CVaR_md = CVaR(w_md, Sigma);
CVaR_md_p = CVaR_md / sum(CVaR_md);
Conv_md = Conv(w_md, Sigma);
Corr_md = Sigma*w_md./(diag(Sigma)*sg2md).^0.5;

%% Tables for the report
T_w = table(w_eq, w_rp, w_md);
T_w.Properties.VariableNames = {'EQ', 'RP', 'MaxDiv'};
T_w.Properties.RowNames = Tickers;

T_MRisk = table(MVaR_eq, MVaR_rp, MVaR_md);
T_MRisk.Properties.VariableNames = {'EQ', 'RP', 'MaxDiv'};
T_MRisk.Properties.RowNames = Tickers;

T_CRisk = table(CVaR_eq_p, CVaR_rp_p, CVaR_md_p);
T_CRisk.Properties.VariableNames = {'EQ', 'RP', 'MaxDiv'};
T_CRisk.Properties.RowNames = Tickers;

T_Conv = table(Conv_eq, Conv_rp, Conv_md);
T_Conv.Properties.VariableNames = {'EQ', 'RP', 'MaxDiv'};
T_Conv.Properties.RowNames = Tickers;

% Plot portfolio weights
h = figure('Color', [1 1 1]);
StockNames = categorical(Tickers);
bar(StockNames, [w_eq, w_rp, w_md]);
legend('Equally Weighted', 'Risk Parity', 'Max Div', 'interpreter', 'latex', 'location', 'northwest');
title('Portfolio weights (Parametric Approach)', 'interpreter', 'latex');
print(h, '-dpng', fullfile(img_directory, 'Q2_Parametric_Weights_EQ_RP_MD'))

% Plot Marginal VaR
h = figure('Color', [1 1 1]);
bar(StockNames, [MVaR_eq, MVaR_rp, MVaR_md]);
legend('Equally Weighted', 'Risk Parity', 'Max Div', 'interpreter', 'latex', 'location', 'northwest');
title('Marginal VaR (Parametric Approach)', 'interpreter', 'latex');
print(h, '-dpng', fullfile(img_directory, 'Q2_Parametric_MVaR_EQ_RP_MD'))

% Plot Component VaR%
h = figure('Color', [1 1 1]);
bar(StockNames, [CVaR_eq_p, CVaR_rp_p, CVaR_md_p]);
legend('Equally Weighted', 'Risk Parity', 'Max Div', 'interpreter', 'latex', 'location', 'northwest');
title('Component VaR\% (Parametric Approach)', 'interpreter', 'latex');
print(h, '-dpng', fullfile(img_directory, 'Q2_Parametric_CVaR_EQ_RP_MD'))


%% Performance Evaluation

LogRet2 = log(second_half(2:end, :) ./ second_half(1:end-1, :));

PortRet_eq = LogRet2 * w_eq;
PortRet_rp = LogRet2 * w_rp;
PortRet_md = LogRet2 * w_md;

% 1) Sharpe Ratio

SR_eq = mean(PortRet_eq) / std(PortRet_eq);
SR_rp = mean(PortRet_rp) / std(PortRet_rp);
SR_md = mean(PortRet_md) / std(PortRet_md);

% 2) Maximum Dradwdown

% Cumulative Return
cum_ret_eq = exp(cumsum(PortRet_eq));
cum_ret_rp = exp(cumsum(PortRet_rp));
cum_ret_md = exp(cumsum(PortRet_md));

% Maximum Drawdown
maxDD_eq = maxdrawdown(cum_ret_eq);
maxDD_rp = maxdrawdown(cum_ret_rp);
maxDD_md = maxdrawdown(cum_ret_md);

% 3) VaR violations

% Parametric approach - VaR and ES
[VaR_eq2, ES_eq2] = get_riskmeasures('G', PortRet_eq, alpha); 
[VaR_rp2, ES_rp2] = get_riskmeasures('G', PortRet_rp, alpha);
[VaR_md2, ES_md2 ] = get_riskmeasures('G', PortRet_md, alpha);

violation_eq = sum(PortRet_eq < -VaR_eq2);
violation_rp = sum(PortRet_rp < -VaR_rp2);
violation_md = sum(PortRet_md < -VaR_md2);

T_Violation_g = table(violation_eq, violation_rp, violation_md);
T_Violation_g.Properties.VariableNames = {'E.Q.', 'R.P.', 'M.D'};
T_Violation_g.Properties.RowNames = "95% VaR Violation";

% 4) ES

T_ES_g = table(ES_eq2, ES_rp2, ES_md2);
T_ES_g.Properties.VariableNames = {'E.Q.', 'R.P.', 'M.D'};
T_ES_g.Properties.RowNames = "Expected Shotfall";


% 5) Sortino Ratio

neg_ret_eq = PortRet_eq(PortRet_eq<0);
neg_ret_rp = PortRet_rp(PortRet_rp<0);
neg_ret_md = PortRet_md(PortRet_md<0);

downside_vol_eq = std(neg_ret_eq);
downside_vol_rp = std(neg_ret_rp);
downside_vol_md = std(neg_ret_md);

sortino_ratio_eq = mean(PortRet_eq)/downside_vol_eq;
sortino_ratio_rp = mean(PortRet_rp)/downside_vol_rp;
sortino_ratio_md = mean(PortRet_md)/downside_vol_md;

T_Sortino_g = table(sortino_ratio_eq, sortino_ratio_rp, sortino_ratio_md);
T_Sortino_g.Properties.VariableNames = {'E.Q.', 'R.P.', 'M.D'};
T_Sortino_g.Properties.RowNames = "Sortino Ratio";



% 6) Calmar Ratio
ann_return_eq = mean(PortRet_eq) * 252;
ann_return_rp = mean(PortRet_rp) * 252;
ann_return_md = mean(PortRet_md) * 252;

Calmar_eq = ann_return_eq / abs(maxDD_eq);
Calmar_rp = ann_return_rp / abs(maxDD_rp);
Calmar_md = ann_return_md / abs(maxDD_md);

T_Calmar_g = table(Calmar_eq, Calmar_rp, Calmar_md);
T_Calmar_g.Properties.VariableNames = {'E.Q.', 'R.P.', 'M.D'};
T_Calmar_g.Properties.RowNames = "Calmar Ratio";
%% ============================
% Portfolio Performance Summary Table
% ============================

% Define portfolio names
portfolios = {'Equally Weighted'; 'Risk Parity'; 'Max Diversification'};

% Create Performance Table (Ensure Portfolio is a Column, Not RowNames)
Performance_Table = table(...
    portfolios, ...                                % Portfolio Names
    [SR_eq; SR_rp; SR_md], ...                     % Sharpe Ratios
    [maxDD_eq; maxDD_rp; maxDD_md], ...            % Maximum Drawdowns
    [violation_eq; violation_rp; violation_md], ...% VaR Violations
    [ES_eq2; ES_rp2; ES_md2], ...                  % Expected Shortfall (ES)
    [sortino_ratio_eq; sortino_ratio_rp; sortino_ratio_md], ...
    [Calmar_eq; Calmar_rp; Calmar_md], ...
    'VariableNames', {'Portfolio', 'Sharpe_Ratio', 'Max_Drawdown', 'VaR_Violations','Expected_Shortfall','Sortino_Ratio', 'Calmar_Ratio'} ...
);

% Display the Summary Table
disp('Portfolio Performance Summary:');
disp(Performance_Table);
% === Update filename ===
txtfilename = txt_directory + "Q2_Parametric_Approach.txt";

% === Write Table to File ===
log_to_file("# ========================================================", txtfilename);
log_to_file("Parametric Approach", txtfilename);
log_to_file("# ========================================================", txtfilename);

log_to_file("--------------------------", txtfilename);
log_to_file("Portfolio Performance Summary", txtfilename);

% === Manually format table content ===
table_header = sprintf('%-25s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
    'Portfolio', 'Sharpe Ratio', 'Max Drawdown', 'VaR Violations', 'Expected Shortfall', 'Sortino Ratio', 'Calmar Ratio');
log_to_file(table_header, txtfilename);

% Loop through each portfolio 
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
