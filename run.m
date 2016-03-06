% n = 1e5; X = X(1:n, :); ix = sum(X); X = X(:,ix>0); Y = Y(1:n);

%% rcv1 + lbfgs
load data/rcv1
name = 'rcv1_lbfgs_5';
bt.lr = 1; bt.rho = .6; bt.nstep = 10;
m = 5; max_iter = 30;

%% rcv1 + gd
load data/rcv1
name = 'rcv1_gd';
bt.lr = 1; bt.rho = .8; bt.nstep = 10;
m = 0; max_iter = 30;

%% news20 + lbfgs
load data/news20
name = 'news20_lbfgs_5';
bt.lr = 1; bt.rho = .6; bt.nstep = 10;
m = 5; max_iter = 30;

%% news20 + lbfgs
load data/news20
name = 'news20_gd';
bt.lr = .5; bt.rho = .6; bt.nstep = 10;
m = 0; max_iter = 200;

%% ctr lbfgs
load ctr
name = 'ctr_lbfgs_5';
bt.lr = 1; bt.rho = .2; bt.nstep = 6;
m = 5; max_iter = 100;

%% ctr gd
load ctr
name = 'ctr_gd';
bt.lr = .05; bt.rho = .5; bt.nstep = 10;
m = 0; max_iter = 200;

%% run
gammas = [1 .7 .4 .1];
res = search_gamma(X, Y, m, max_iter, 1, gammas, bt, 10);

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
