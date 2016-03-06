function res = search_gamma(X, Y, m, max_iter, func, gamma, linesearch, repeat)
l2 = 0;

w = @() randn(size(X,2),1)*.1;
Y(Y<=0) = -1;
Y(Y>0) = 1;
res = [];
for g = gamma
  obj = @(w) logit_loss(Y, X, w, l2, g, func);
  objv = zeros(repeat, max_iter);
  for r = 1 : repeat
    fprintf('gamma =  %f, repeat = %d\n', g, r);
    objv(r,:) = lbfgs(obj, w(), m, max_iter, linesearch);
  end
  res = [res; mean(objv, 1)];
end
