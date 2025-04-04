% function [LR , Pvalue ]= get_LRind (n00 ,n01 ,n10 , n11 )
% %
% % Likelihood ratio test statistic
% %
% %p under the independence assumption
% 
% p =( n01 + n11 )/( n01 + n11 + n00 + n10 )
% term1 =( n00 + n10 )* log (1 -p );
% term2 =( n01 + n11 )* log (p );
% %log -lik
% LR1 = -2*( term1 + term2 );
% %p under the dependence assumption
% pie0 = n01 /( n00 + n01 );
% pie1 = n11 /( n10 + n11 );
% term3 = n00 * log (1 - pie0 );
% term4 = n01 * log ( pie0 );
% term5 = n10 * log (1 - pie1 );
% term6 = n11 * log ( pie1 );
% LR2 = -2*( term3 + term4 + term5 + term6 );
% % Likelihood ratio test statistic
% LR =LR1 - LR2 ;
% %P-value
% Pvalue =1 - cdf ('chi2',LR ,1);
% end

% function [LR, Pvalue] = get_LRind(n00, n01, n10, n11)
%     % Avoid division by zero errors
%     if (n10 + n11 == 0) || (n00 + n01 == 0)
%         LR = 0;
%         Pvalue = 0;
%         return;
%     end
% 
%     % Unconditional probability
%     p = (n01 + n11) / (n00 + n01 + n10 + n11);
%     term1 = (n00 + n10) * log(1 - p);
%     term2 = (n01 + n11) * log(p);
%     LR1 = -2 * (term1 + term2);
% 
%     % Conditional probabilities
%     pie0 = n01 / (n00 + n01);
%     pie1 = n11 / (n10 + n11);
% 
%     % Ensure log(0) does not occur
%     if pie0 == 0 || pie1 == 0 || pie0 == 1 || pie1 == 1
%         LR = NaN;
%         Pvalue = NaN;
%         return;
%     end
% 
%     term3 = n00 * log(1 - pie0);
%     term4 = n01 * log(pie0);
%     term5 = n10 * log(1 - pie1);
%     term6 = n11 * log(pie1);
%     LR2 = -2 * (term3 + term4 + term5 + term6);
% 
%     % Compute likelihood ratio
%     LR = LR1 - LR2;
%     Pvalue = 1 - chi2cdf(LR, 1);
% end

function [LR, Pvalue] = get_LRind(n00, n01, n10, n11)
    % Avoid division by zero and undefined log(0) issues
    if (n10 + n11 == 0) || (n00 + n01 == 0)
        LR = 0;
        Pvalue = 1; % No need to test if there are no violations
        return;
    end

    % Unconditional probability
    total_transitions = n00 + n01 + n10 + n11;
    p = (n01 + n11) / total_transitions;

    % Ensure p is never exactly 0 or 1
    p = max(min(p, 1 - eps), eps);

    term1 = (n00 + n10) * log(1 - p);
    term2 = (n01 + n11) * log(p);
    LR1 = -2 * (term1 + term2);

    % Conditional probabilities
    pie0 = n01 / (n00 + n01);
    pie1 = n11 / (n10 + n11);

    % Ensure pie0 and pie1 are not exactly 0 or 1
    pie0 = max(min(pie0, 1 - eps), eps);
    pie1 = max(min(pie1, 1 - eps), eps);

    term3 = n00 * log(1 - pie0);
    term4 = n01 * log(pie0);
    term5 = n10 * log(1 - pie1);
    term6 = n11 * log(pie1);
    LR2 = -2 * (term3 + term4 + term5 + term6);

    % Compute likelihood ratio
    LR = LR1 - LR2;

    % Ensure LR is non-negative
    LR = max(LR, 0);

    % Compute p-value from chi-square distribution
    Pvalue = 1 - chi2cdf(LR, 1);
end