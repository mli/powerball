%% load data

load_data('ctr')
bt.lr = 1;
bt.rho = .2;
bt.nstep = 6;
m = 1;
max_iter = 100;
l2 = 1;

%%

% global grads
% grads = [];
w = @() randn(size(X,2),1)*.1;
res = [];
loss = @(w) logit_loss(Y, X, w, l2);
obj = @(w, k) power_func(loss, w, k, max_iter, [1,1]);
[objv] = lbfgs(obj, w(), m, max_iter, bt);


%%

x = [x; mean(abs(grads))];
y = [y; std((grads))];
