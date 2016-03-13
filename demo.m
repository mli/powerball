%% load data
[X, Y] = load_data('ctr');

%% hyper parameters
% line search decay
bt.rho = .2;
% lbfgs m
m = 5;
% l2 regularization, namely l2 * ||w||_2^2
l2 = 1;
% maximal number of iterations
max_iter = 100;

%% run
w = randn(size(X,2),1)*.1;
loss = @(w) logit_loss(Y, X, w, l2);

% standard lbfgs
obj1 = @(w, k) power_func(loss, w, k, max_iter, [1, 1]);
res1 = lbfgs(obj1, w(), m, max_iter, bt);

% adaptive gamma varying from .1 to .9
obj2 = @(w, k) power_func(loss, w, k, max_iter, [.1, .9]);
res2 = lbfgs(obj2, w(), m, max_iter, bt);

%% plot

clf
plot(1:max_iter, res1, '-ob');
hold on
plot(1:max_iter, res2, '-xr');
xlabel('iteration')
ylabel('objective')
legend('lbfgs', 'powerball')
