function [objvs] = lbfgs(obj, w, m, niter, linesearch)

default_ls = struct('lr', 1, 'c1', 1e-4, 'c2', .9, 'rho', .8, 'nstep', 8);

if ~exist('linesearch','var')
  ls = default_ls;
else
  ls = parse_options(linesearch, default_ls);
end

s = [];
y = [];
max_m = m;
[objv, g, auc] = obj(w);
objvs = [];
for k = 1 : niter
% two loop
  m = size(y, 2);
  p = - g;
  alpha = zeros(m,1);
  for i = min(max_m, m) : -1 : 1
    alpha(i) = (s(:,i)' * p ) / (s(:,i)' * y(:,i) + 1e-10);
    p = p - alpha(i) * y(:,i);
  end
  if m > 0
    eta = (s(:,m)'*y(:,m)) / (y(:,m)'*y(:,m) + 1e-10);
    p = eta * p;
  end
  for i = 1 : min(max_m, m)
    beta = (y(:,i)'*p) / (s(:,i)'*y(:,i));
    p = p + (alpha(i) - beta) * s(:,i);
  end
  p = min(max(p, -20), 20);

% back tracking
  alpha = ls.lr;
  gp = g'*p;
  fprintf('epoch %d, objv %f, gp %f, auc %f \n', k, objv, gp, auc);
  for j = 1 : ls.nstep
    [new_o, new_g, auc] = obj(w + alpha * p);
    new_gp = new_g' * p;
    fprintf('\talpha %f, new_objv %f, new_gp %f\n', alpha, new_o, new_gp);
    if (new_o <= objv + ls.c1 * alpha * gp) % && (new_gp >= ls.c2 * gp)
      break;
    end
    alpha = alpha * ls.rho;
  end

% update s and y
  if m >= max_m
    s = s(:,2:m);
    y = y(:,2:m);
  end

  if max_m > 0
    s = [s, alpha*p];
    y = [y, new_g - g];
  end

  w = w + alpha * p;
  g = new_g;
  objv = new_o;
  objvs = [objvs, objv];
end

function options = parse_options(options, defaultopt)
%options = parse_options(options, defaultopt)
% fill the missing options with values from the default one

f = fieldnames(defaultopt);
for i = 1 : length(f)
    if ~isfield(options,f{i}) || isempty(options.(f{i}))
        options.(f{i}) = defaultopt.(f{i});
    end
end