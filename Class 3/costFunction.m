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
% Compute the cost J of a particular choice of theta.
% Note: grad should have the same dimensions as theta
z = X*theta;
h = 1./(1+exp(-z)); %sigmoid function applied to our hypothesis

J = sum(-y.*log(h)-(1-y).*log(1-h))/m;
grad = ((h-y)'*X)./m;
% =============================================================
end