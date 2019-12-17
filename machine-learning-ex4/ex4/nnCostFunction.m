function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to the X data matrix

y_matrix = eye(num_labels)(y,:);
y = y_matrix;



%-------------------Vectorized Trail start-----------
X = [ones(m,1) X];
h = X*Theta1';
h = sigmoid(h);

h = [ones(m,1) h];

h2 = h*Theta2';
h2 = sigmoid(h2);

  ## J = J + sum((y .* log(h))) + sum((1-y) .* (log((1-h))));
  J = J + trace(y'* log(h2)) + trace((1-y)'* (log(1-h2)));
  J = -(J/m);
  
  t1=0;
  temp1=0;
  t2=0;
  temp2=0;
  
    temp = Theta1;
    temp(:,1) = 0;
    t1 = t1 + sum((temp .^ 2)(:));
    t1 = t1 * (lambda/(2*m));
        
    J = J + t1;

    temp = Theta2;
    temp(:,1) = 0;
    t2 = t2 + sum((temp .^ 2)(:));
    t2 = t2 * (lambda/(2*m));
    J = J + t2;
%-------------------Vectorized Trail end-----------    
##%-------------------Back propogation Starts-----------    
##%X = [ones(m,1) X];
##
##%tg = zeros(size(Theta1, Theta2));
##
##for t=1:m
##
##a1 = X(t,:);  
##%fprintf(['dimension of a1 is  %d x %d \n'], rows(a1), columns(a1));
##%a1 = [ones(rows(a1),1) a1];
##
##
##z2 = a1*Theta1';
##
##a2 = sigmoid(z2);
##%fprintf(['dimension of z2 is  %d x %d \n'], rows(z2), columns(z2));
##%fprintf(['dimension of a2 is  %d x %d \n'], rows(a2), columns(a2));
##
##a2 = [ones(rows(a2),1) a2];
##
##%fprintf(['dimension of a2 is  %d x %d \n'], rows(a2), columns(a2));
##
##
##z3 = a2*Theta2';
##a3 = sigmoid(z3);
##%a3 = [ones(rows(a3),1) a3];
##%fprintf('\n Size of z3 is %f x %f \n', size(z3,1),size(z3,2));
##yt = y(t,1);
##
##%fprintf('%d ', yt);
##
##%yt = y(1,:);
##sd3 = a3 - yt;
##%fprintf('%d %d', size(sd3));
##
##
##sd2 = (sd3*Theta2) .* (a2 .* (1-a2));
##sd2 = sd2(2:end);
##
##
##Theta1_grad = Theta1_grad + sd2'*a1;
##Theta2_grad = Theta2_grad + sd3'*a2;
##
##
##
##endfor
##
##Theta1_grad = Theta1_grad/m;
##Theta2_grad = Theta2_grad/m;
##
##
##%-------------------Back propogation end-----------

% -----Back Propogation optimized from Tutorial starts------

a1 = X;
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(rows(a2),1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
%a3 = [ones(rows(a3),1) a3];
d3 = a3 - y;

Theta2no1 = Theta2(:,2:end);
d2 = (d3*Theta2no1) .* sigmoidGradient(z2);

Theta1_grad = Theta1_grad + d2'*a1;
Theta2_grad = Theta2_grad + d3'*a2;

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;
% -----Back Propogation optimized from Tutorial ends ------

% -----Back Propogation with Gradients regularization from Tutorial starts------

Theta1(:,1) = 0
Theta2(:,1) = 0

regTheta1 = (lambda/m)*Theta1;
regTheta2 = (lambda/m)*Theta2;

Theta1_grad = Theta1_grad + regTheta1;
Theta2_grad = Theta2_grad + regTheta2;

% -----Back Propogation with Gradients regularization from Tutorial ends------
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
