function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

% ---- Regularized Cost function Begins --------------
h = X*theta;
h = sigmoid(h)
a=0;
for i = 1:m
J = J + (((y(i)) * (log(h(i)))) + ((1-y(i))*log(1-h(i))));
endfor
J = -(J/m);

for l = 2:rows(theta)
    a = a + (theta(l)^2);
endfor
a = a * (lambda/(2*m));
J = J + a;
% ---- Regularized Cost function ends --------------

% ---- Regularized Gradient function Begins --------------

h = X*theta;
h = sigmoid(h)

z = 0;
a = 0;
for t = 1:rows(theta)     
 for i = 1:m
    
    z = z + ((h(i) - y(i))*X(i,t));
    
 endfor
  if t == 1 
      a = 0;
   else
      a = theta(t)*lambda;    
  endif
  z  = z + a;
 
 grad(t) = z/m;
 z = 0;



 
endfor


% ---- Regularized Gradient function Ends --------------
% =============================================================

end
