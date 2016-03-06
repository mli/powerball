function [objv, grad] = logit_loss(Y, X, w, l2, gamma, func)
Xw = X * w;
tau = Y .* (Xw);
objv = sum(log(1 + exp(-tau)));
tau = max(-100, min(100, tau));
grad = X' * (- Y ./ (1  + exp(tau))) + l2 * w;
% grad = min(100, max(-100, grad));

if func == 0; return; end

gg = abs(grad).^gamma;

if func == 1
  grad = sign(grad) .* gg;
elseif func == 2
  grad = tanh(sign(grad).*gg) .* gg;
elseif func == 3
  grad = (2./(1+exp(-sign(grad).*gg))-1) .* gg;
end
