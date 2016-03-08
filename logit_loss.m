function [objv, grad] = logit_loss(Y, X, w, l2)
%LOGISTIC LOSS
% [objv, grad] = logit_loss(Y, X, w, l2)
% Y     : label
% X     : data
% w     : weight
% l2    : l2 regularizer
% objv  : the objective value
% grad  : the gradient

tau = Y .* (X * w);
objv = sum(log(1 + exp(-tau)));
tau = max(-100, min(100, tau));
grad = X' * (- Y ./ (1  + exp(tau))) + l2 * w;
