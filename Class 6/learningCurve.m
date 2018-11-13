function [error_train, error_val] =learningCurve(X, y, Xval, yval, lambda)
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. error_train(i) contains the training error for  i examples
%
% Number of training examples
m = size(X, 1);

% Inicialize the output vectors 
error_train = zeros(m, 1);
error_val   = zeros(m, 1);
%
% Note: The training error is evaluated on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%      The cross-validation error is evaluatde on
%       the _entire_ cross validation set (Xval and yval).
%
% Loop over the examples
  for i=1:m
    Xtrain=X(1:i, :);
    ytrain=y(1:i);
    [theta] = trainLinearReg(Xtrain, ytrain, lambda);

    error_train(i)=(Xtrain*theta-ytrain)'*(Xtrain*theta-ytrain)/(2*m);
    error_val(i)=(Xval*theta-yval)'*(Xval*theta-yval)/(2*m);

  end
