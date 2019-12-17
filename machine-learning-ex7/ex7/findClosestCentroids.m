function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
%idx = zeros(size(X,1), 1);
idx = [];
% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%=======================================================
##temp =0;
##m = size(X,1);
##for i = 1:m
##  for j = 1:K
##  dist = (((X(i, 1) - centroids(j, 1))^2 + (X(i, 2) - centroids(j, 2))^2)
##  
##  if (j == 1)
##    idx(i) = 1;    
##  elseif (temp > dist)
##    idx(i) = j;
##  endif
##  temp = dist;  
##  endfor
##  #fprintf(' Centroid of %f point is %f \n', i, idx(i));
##endfor

%=======================================================
d = []; %size((X,1),K);
f = [];

for i = 1:K
  q = centroids(i,:);
  b = bsxfun(@minus, X,q);
  d = sum(b.^2,2);
  f = [f,d];
#  [minval, idx] = min(d,[],2);
  
endfor
   
[minval, idx] = min(f,[],2);

% =============================================================

end

