n = 1e6; X = X(1:n, :); ix = sum(X); X = X(:,ix>0); Y = Y(1:n);
%% gd
Y(Y<=0) = -1; Y(Y>0) = 1;

% linesearch
clear ls
% ls.lr = 5e-2;
ls.lr = 1;
ls.rho = .1;
ls.nstep = 5;

max_iter = 400;
l2 = 0;
m = 5;

func = [0 1 2 3];

oss = {};
res = [];
last = [];
for f = func
  fprintf('\n function = %d\n', f)
  if f == 0
    gammas = [1];
  else
    gammas = [.9:-.1:.1];
  end

  os = [];
  for gamma = gammas
    fprintf('\ngamma = %f\n', gamma)
    obj = @(w) logit_loss(Y, X, w, l2, gamma, f);
    objv = lbfgs(obj, zeros(size(X,2),1), m, max_iter, ls);
    os = [os; objv];
  end

  if f > 0
    last = [last os(:, end)];
  end
  oss{end+1} = os;
  [~,j]= min(mean(os(:, end-5:end)'));
  res = [res; os(j,:)];
end


%% draw
% figure
% muplot2(res');
% % legend(strcat('\gamma=', num2str(gammas')))
% legend(strcat('func=', num2str(func')))
% xlabel('iteration')
% ylabel('objective')
