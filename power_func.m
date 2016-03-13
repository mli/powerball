function [objv, grad] = power_func(loss, w, curr_iter, max_iter, gamma)
%POWERBALL FUNCTION
% [objv, grad] = power_func(loss, w, curr_iter, max_iter, gamma, func)
% loss      : the loss function, such as logit_loss
% w         : the weight
% curr_iter : current iteration
% max_iter  : maximal iteration
% gamma     : [gamma_start, gamma_end]

[objv, grad, h] = loss(w);
% global grads
% grads = [grads grad];
assert(length(gamma) == 2)
gamma = gamma(1) + (gamma(2) - gamma(1)) * curr_iter / max_iter;

% grad = sign(grad) .* abs(grad).^gamma;
% grad = tanh(grad);
% grad = grad ./ (1 + abs(grad));

grad = grad ./ h;
