function [LR , Pvalue ]= get_LRuc (p , N , n)
%p=1 - alpha theoretical violation probability
%N number of violations
%n number of comparaisons , n>=N

phat =N/n;
term1 = -2 * ((n - N) * log(1 - p) + N * log(p));
term2 = -2 * ((n - N) * log(1 - phat) + N * log(phat));

if N ==0
LR = -2* n* log (1 -p );
elseif N == n
LR = -2* n* log (p );
else
LR = term1 - term2 ;
end
Pvalue =1 - cdf ('chi2',LR ,1);