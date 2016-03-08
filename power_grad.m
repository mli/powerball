function [objv, grad] = power_grad(loss, w, curr_iter, max_iter, gamma, func)


[objv, grad] = loss(w);
if func == 0; return; end

assert(length(gamma) == 2)
gamma = gamma(1) + (gamma(2) - gamma(1)) * curr_iter / max_iter;
gg = abs(grad).^gamma;

if func == 1
  grad = sign(grad) .* gg;
elseif func == 2
  grad = tanh(sign(grad).*gg) .* gg;
elseif func == 3
  grad = (2./(1+exp(-sign(grad).*gg))-1) .* gg;
end
