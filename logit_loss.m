function [objv, grad] = logit_loss(Y, X, w, l2)
Xw = X * w;
tau = Y .* (Xw);
objv = sum(log(1 + exp(-tau)));
tau = max(-100, min(100, tau));
grad = X' * (- Y ./ (1  + exp(tau))) + l2 * w;
