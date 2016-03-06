% n = 1e6; X = X(1:n, :); ix = sum(X); X = X(:,ix>10); Y = Y(1:n);

%% rcv1 + lbfgs
load data/rcv1
name = 'rcv1_lbfgs_5';
bt.lr = 1; bt.rho = .6; bt.nstep = 10;
m = 5; max_iter = 30;

%% rcv1 + gd

%% news20
load data/news20
name = 'news20_lbfgs_5';
bt.lr = 1; bt.rho = .6; bt.nstep = 10;
m = 5; max_iter = 30;

%% ctr lbfgs
load ctr
name = 'ctr_lbfgs_5';
bt.lr = 1; bt.rho = .6; bt.nstep = 10;
m = 5; max_iter = 100;

%% run
gammas = [1 .7 .4 .1];
res = search_gamma(X, Y, m, max_iter, 1, gammas, bt);

%% draw
figure(1)
clear opt;
opt.mk = {''};
muplot2([], res', opt)
legend(strcat('\gamma=', num2str(gammas')))
xlabel('iteration')
ylabel('objective')
savepdf(['fig/', name, '_objv'])

figure(2)
[~,i] = min(res);
clear opt
opt.ls = {''};
opt.mk = {'x'};
muplot2([], gammas(i), opt)
xlabel('iteration')
set(gca,'ytick', gammas(end:-1:1));
ylabel('\gamma')
ylim([0, 1]);
savepdf(['fig/', name, '_gamma'])
