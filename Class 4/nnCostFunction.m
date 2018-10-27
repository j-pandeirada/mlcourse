function [J, grad] = nnCostFunction(thetaVec,X,y,sizes)
%compute cost function of the neural network without regularization 

%INPUTS:
%   - unrolled initialtheta vector
%   - size of each layer in the NN
%   - training data set -> y should already be one hot encoded!
%OUTPUTS:
%   - cost value
%   - gradient vector

%from thetavec get Theta1,Theta2 etc... -> weights of each layer
%get theta matrices from unrolled vector
theta.l1 = reshape(thetaVec(1:sizes.hidden_layer_size*(sizes.input_layer+1))...
                    ,sizes.hidden_layer_size,sizes.input_layer+1);

theta.l2 = reshape(thetaVec(sizes.hidden_layer_size*(sizes.input_layer+1)+1:end)...
                    ,sizes.num_labels,sizes.hidden_layer_size+1);

%number of examples
m = length(y);

%initialize gradient matrices
delta.l1 = zeros(size(theta.l1,1),size(theta.l1,2));
delta.l2 = zeros(size(theta.l2,1),size(theta.l2,2));

%use forward/back prop to compute gradients-> D1,D2, etc.. and cost J

%perform forward propagation
nodes = forwardprop(theta.l1,theta.l2,X);
h = nodes.a3;

%calculate cost -> without regularization
J = sum(sum(-y.*log(h)-(1-y).*log(1-h)))/m;

%perform backward propagation for each training example
for i=1:m
    %compute errors
    error.a3 = nodes.a3(:,i)-y(:,i);
    error.a2 = (theta.l2'*error.a3).*(nodes.a2(:,i)).*(1-nodes.a2(:,i));
    %compute gradients
    %detail:error vectors should not contain bias!
    delta.l1 = delta.l1 + error.a2(2:end)*nodes.a1(:,i)';
    delta.l2 = delta.l2 + error.a3*nodes.a2(:,i)';
end

%finalize gradient vector -> without regularization
D.l1 = delta.l1./m;
D.l2 = delta.l2./m;

%unroll D1,D2,etc... to get grad
grad = [D.l1(:);D.l2(:)];

end

