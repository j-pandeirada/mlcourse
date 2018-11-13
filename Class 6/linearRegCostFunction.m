function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Compute the cost and gradient of regularized linear regression for a particular choice of theta.
%

%create hypothesis function
h = X*theta;
%create error vector between data and hypothesis
err = h-y;
J = ((err'*err) + lambda*sum(theta(2:end).^2))/(2*m);

grad = ((1/m)*(h-y)'*X)';
grad(2:end)= grad(2:end) + (lambda/m).*theta(2:end);
% =========================================================================
grad = grad(:);
end