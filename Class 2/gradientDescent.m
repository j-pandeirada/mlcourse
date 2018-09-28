function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on parameter vector theta. 
    
    %create hypothesis function
    h = X*theta;
    %compute the derivative of cost function
    gradJ=(1/m)*(h-y)'*X;
    %update theta parameters
    theta=theta-alpha*gradJ';
    % ===========================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
