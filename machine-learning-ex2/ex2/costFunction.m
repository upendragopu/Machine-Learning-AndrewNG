function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

h = X*theta;
h = sigmoid(h)
for i = 1:m
J = J + (((y(i)) * (log(h(i)))) + ((1-y(i))*log(1-h(i))));

endfor
J = -(J/m);

h = X*theta;
h = sigmoid(h)

z = 0;
for t = 1:rows(grad)     
 for i = 1:m
    z = z + ((h(i) - y(i))*X(i,t));
  
 endfor
 grad(t) = z/m;
 z=0;


 
endfor

% =============================================================

end
