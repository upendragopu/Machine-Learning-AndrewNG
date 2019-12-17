function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


h = X*theta;
h = sigmoid(h);
a=0;
% for i = 1:m
%  J = J + (((y(i)) * (log(h(i)))) + ((1-y(i))*log(1-h(i))));
% endfor
% J = -(J/m);
%c =  log(h);
%c = y .* log(h);
% d =  1 - y;
%e = 1 - h;
%e = log((1-h));
%f = (1-y) .* (log((1-h)));
% J = J + sum(c) + sum(f);

    J = J + sum((y .* log(h))) + sum((1-y) .* (log((1-h))));
    J = -(J/m);
    temp = theta;
    temp(1) = 0;
    a = a + sum(temp .^ 2);
%for l = 2:rows(theta)
%    a = a + (theta(l)^2);
%endfor
    a = a * (lambda/(2*m));
    J = J + a;
% ---- Regularized Cost function ends --------------

% ---- Regularized Gradient function Begins --------------

h = X*theta;
h = sigmoid(h);

z = 0;
a = 0;
%for t = 1:rows(theta)     
% for i = 1:m
    
%    z = z + ((h(i) - y(i))*X(i,t));
    
% endfor

    z = (X'*(h - y)) / m;
temp = theta;
temp(1) = 0;
%  if t == 1 
%      a = 0;
%   else
%      a = theta(t)*lambda;    
%  endif
 
  a = (temp * lambda) / m;
 grad  = z + a;
 
 
% =============================================================

grad = grad(:);

end
