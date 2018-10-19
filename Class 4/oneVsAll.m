function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%classifiers in a matrix all_theta
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains multiple
%   (num_labels) logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1); %number of examples
n = size(X, 2); %number of features

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m,1) X];

% ====================== YOUR CODE HERE ======================
% You should complete the following code to train num_labels
% logistic regression classifiers with regularization parameter lambda. 

%For this assignment, we recommend using fmincg to optimize the cost function. 
%Use a for-loop (for c = 1:num_labels) to loop over the different classes.
% fmincg works similarly to fminunc, but is more efficient when
%dealing with large number of parameters.

  for c = 1:num_labels
% Set Initial theta
       initial_theta = zeros(n+1, 1); 
   
% Set options for fmincg(similar to the previous lab work with fminunc)
%choose for example 50 iterations
      options = optimset('GradObj', 'on', 'MaxIter', 250);
      
%Run fmincg to obtain the optimal theta (similar to the previous lab work with  fminunc) 
%Use the costFunctionReg.m function you completed for the previous lab work. 
% Hint: You can use y == c to obtain a binary vector y of 1's and 0's that
% assigns 1 only for examples that belongs to class c
    new_y = y==c;
    % Optimize
    [theta, J, exit_flag] = ...
        fmincg(@(t)(costFunctionReg(t, X, new_y, lambda)), initial_theta, options);

%Save the parameters of each binary classifier in one raw of matrix all_theta
     all_theta(c,:)= theta;
     
  end

% =========================================================================
end
