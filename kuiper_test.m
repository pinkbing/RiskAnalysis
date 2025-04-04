function pvalue = kuiper_test(PIT)
    % Sort PIT values
    PIT_sorted = sort(PIT);
    n = length(PIT_sorted);
    
    % Compute empirical CDF
    ECDF = (1:n)' / n;  % Ensure column vector
    
    % Compute Kuiper statistic
    D_plus = max(ECDF - PIT_sorted);
    D_minus = max(PIT_sorted - [0; ECDF(1:end-1)]); % Ensure matching dimensions
    
    % Compute Kuiper's V_n statistic
    Vn = D_plus + D_minus;
    
    % Approximate p-value using asymptotic formula
    lambda = (sqrt(n) + 0.155 + 0.24 / sqrt(n)) * Vn;
    pvalue = 2 * exp(-2 * lambda^2);
end