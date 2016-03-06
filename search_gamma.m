function res = search_gamma(X, Y, m, max_iter, func, gamma, linesearch)
repeat = 10;
l2 = 0;

w = @() randn(size(X,2),1)*.1;
Y(Y<=0) = -1;
Y(Y>0) = 1;
res = [];
for g = gamma
  fprintf('gamma =  %f\n', g);
  obj = @(w) logit_loss(Y, X, w, l2, g, func);
  objv = zeros(repeat, max_iter);
  for r = 1 : repeat
    fprintf('repeat =  %d\n', r);
    objv(r,:) = lbfgs(obj, w(), m, max_iter, linesearch);
  end
  res = [res; mean(objv)];
end
