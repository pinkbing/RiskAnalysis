function [ewmavar] = get_ewma_var(LogRet, lambda, ewma0)
%%INPUT
%LogRet : logreturn series
%lambda: smoothing parameter of the ewma model [0,1]
%ewma0: initial variance (a positive number)
%OUTPUT
%ewmavar: column vector containing the ewma variance series

ewmavar(1,1) = ewma0;

for j=1:length(LogRet)
    ewmavar(j+1,1) = lambda*ewmavar(j) + (1-lambda)*LogRet(j)^2;
end
