function [VaR, ES] = get_riskmeasures(model, LogRet, ConfLevel)

%length(varargin)

switch model 
    case 'G'%use parametric Gaussian
        mu_ret = mean(LogRet);
        std_ret = std(LogRet);
        z = icdf('norm', 1-ConfLevel, 0, 1);
        VaR = -(mu_ret + z*std_ret);
        ES =  -(mu_ret - std_ret*normpdf(-VaR,0,1)/(1-ConfLevel)); 

    case 'NP'%use non parametric approach    
        VaR = -quantile(LogRet, 1-ConfLevel);
        ES =  -mean(LogRet((LogRet<=-VaR)));

    case 'BOOT'
        Nb = 500;
        nobs = length(LogRet);
        for i=1:Nb
            %pick a day at random
           U = randi(nobs, nobs, 1);        
            %pick the corresponding return: this is the bootstrap sample
           SimRet = LogRet(U, :);           
            %simulate  bootstrap VaR
          [VaRb(i), ESb(i)] = get_riskmeasures('NP', SimRet, ConfLevel);       
        end
        VaR = mean(VaRb);
        ES = mean(ESb);

    case 'STML'
        [phat, pci] =  mle(LogRet, 'distribution', 'tLocationScale');
        mu_ml  =  phat(1);
        sg_ml  =  phat(2);
        nu_ml  =  phat(3);
        VaR = -icdf('tLocationScale', 1-ConfLevel, mu_ml, sg_ml, nu_ml);
        ES = -quadgk(@(x) x.*pdf('tLocationScale', x, mu_ml, sg_ml, nu_ml), -inf, -VaR)/(1-ConfLevel);

    case 'STMM'
        mu_ret = mean(LogRet);
        std_ret = std(LogRet);
        kurt_ret = kurtosis(LogRet);        
        nobs = length(LogRet);
        mu_mm = mu_ret; 
        nu_mm = 4 + 6 / (kurt_ret - 3); % Degrees of freedom estimate
        sg_mm = sqrt(((nu_mm - 2) / nu_mm)) * std_ret;
        VaR = -icdf('tLocationScale', 1-ConfLevel, mu_mm, sg_mm, nu_mm);
        ES = -quadgk(@(x) x.*pdf('tLocationScale', x, mu_mm, sg_mm, nu_mm), -inf, -VaR)/(1-ConfLevel);
    
    case 'MC' % Monte Carlo Simulation
        num_simulations = 10000;
        SimRet = mean(LogRet) + std(LogRet) * randn(num_simulations, 1); % Generate simulated returns
        VaR = -quantile(SimRet, 1-ConfLevel);
        ES = -mean(SimRet(SimRet <= -VaR));
    
    case 'EWMA' % EWMA-Based VaR
        lambda = 0.94; % Smoothing parameter
        ewma0 = var(LogRet); % Initial variance
        ewmavar = get_ewma_var(LogRet, lambda, ewma0);
        ewma_vol = sqrt(ewmavar(end)); % Latest EWMA volatility estimate
        VaR = -norminv(1-ConfLevel) * ewma_vol;
        ES =  -(normpdf(-VaR,0,1)* ewma_vol/(1-ConfLevel));

    case 'ALL'
        [VaR(:, 1), ES(:,1)] = get_riskmeasures('G', LogRet, ConfLevel);
        [VaR(:, 2), ES(:,2)] = get_riskmeasures('NP', LogRet, ConfLevel);
        [VaR(:, 3), ES(:,3)] = get_riskmeasures('BOOT', LogRet, ConfLevel);
        [VaR(:, 4), ES(:,4)] = get_riskmeasures('STML', LogRet, ConfLevel);
        [VaR(:, 5), ES(:,5)] = get_riskmeasures('STMM', LogRet, ConfLevel);
        [VaR(:, 6), ES(:,6)] = get_riskmeasures('MC', LogRet, ConfLevel);
        [VaR(:, 7), ES(:,7)] = get_riskmeasures('EWMA', LogRet, ConfLevel);
end


