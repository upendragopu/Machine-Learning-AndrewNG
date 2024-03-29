function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%C = [0.01 0.03 0.1 0.3 1 3 10 30];
%sigma = [0.01 0.03 0.1 0.3 1 3 10 30];

params = [100 100 100];

for C = [0.01 0.03 0.1 0.3 1 3 10 30]
  for sigma = [0.01 0.03 0.1 0.3 1 3 10 30] 
    x1 = [1 2 1]; x2 = [0 4 -1];
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions = svmPredict(model, Xval);
    cost = mean(double(predictions ~= yval));
    params = [params; [cost C sigma]];
    fprintf('\n\n\n\n C = %f and sigma = %f and Cost =  %f \n\n\n\n', C, sigma, cost);
  endfor
endfor



 [min_values,indices] = min(params,[],1)
  lowestcost = params(indices(1), 1);
  C = params(indices(1), 2);
  sigma = params(indices(1), 3);
  fprintf('\n\n\n\n Final C = %f and sigma = %f and Cost =  %f \n\n\n\n', C, sigma, lowestcost);

  
% =========================================================================

end
