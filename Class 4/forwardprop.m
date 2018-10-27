function nodes = forwardprop(Theta1, Theta2, X)
%   p = forwardprop(Theta1, Theta2, X) outputs structure with the values in the neurons
%   given an example from the dataset-> used for training the NN

% Number of examples
m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);

% ====================== Make forward prop calculations ======================

nodes.a1 = [ones(m,1) X];
z2 = Theta1*nodes.a1';
nodes.a2 = 1./(1+exp(-z2));nodes.a2=[ones(1,m);nodes.a2];
z3 = Theta2*nodes.a2;
nodes.a3 = 1./(1+exp(-z3));

nodes.a1 = nodes.a1';
% =========================================================================
end
