function p = predict(Theta1, Theta2, X)
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Number of examples
m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Complete the following code to make predictions using the trained NN. 
%You should set p to a vector containing labels between 1 to num_labels.

a1 = [ones(m,1) X];
z2 = Theta1*a1';
a2 = 1./(1+exp(-z2));a2=[ones(1,m);a2];
z3 = Theta2*a2;
a3 = 1./(1+exp(-z3));

[~,ind]=max(a3);
p=ind;
% =========================================================================
end
