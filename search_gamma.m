function res = search_gamma(X, Y, l2, m, max_iter, func, gamma, linesearch, repeat)

w = @() randn(size(X,2),1)*.1;
Y(Y<=0) = -1; Y(Y>0) = 1;
res = [];
for g = gamma
  loss = @(w) logit_loss(Y, X, w, l2);
  obj = @(w, k) power_grad(loss, w, k, max_iter, [g,g], func);
  objv = zeros(repeat, max_iter);
  for r = 1 : repeat
    fprintf('gamma =  %f, repeat = %d\n', g, r);
    objv(r,:) = lbfgs(obj, w(), m, max_iter, linesearch);
  end
  res = [res; mean(objv, 1)];
end
