%% Initialization
clear ; close all; clc
tic
%% Setup the NN parameters
sizes.input_layer = 400;      % 20x20 Input Images of Digits
sizes.hidden_layer_size = 25; % 25 hidden units
sizes.num_labels = 10;        % 10 labels, from 1 to 10  (0 is mapped to label 10) 

input_layer_size = sizes.input_layer;
hidden_layer_size = sizes.hidden_layer_size;
num_labels = sizes.num_labels;
%% Load dataset and setup data structures
% Load Training Data
load ex3data1.mat
m = length(y);%number of examples

%Random theta structure initialization
INIT_EPSILON = 3;
theta.l1 = rand(hidden_layer_size,input_layer_size+1)*(2*INIT_EPSILON)-INIT_EPSILON;
theta.l2 = rand(num_labels,hidden_layer_size+1)*(2*INIT_EPSILON)-INIT_EPSILON;

%gradient structure initialization
delta.l1 = zeros(size(theta.l1,1),size(theta.l1,2));
delta.l2 = zeros(size(theta.l2,1),size(theta.l2,2));

%ONE-HOT ENCONDING Y
y_cod = onehotenconding(y,num_labels);

%unroll theta structure to get initial_theta
initial_theta = [theta.l1(:);theta.l2(:)];

%% Run minimization -> NN Learning
% Options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);

lambda = 1;

[nn_params, cost] = ...
	fmincg(@(t)(nnCostFunction(t, X, y_cod,sizes,lambda)), initial_theta, options);

%% Gather NN params and use it!
% get theta matrices from unrolled version
theta.l1 = reshape(nn_params(1:(hidden_layer_size*(input_layer_size+1))),...
                    hidden_layer_size,input_layer_size+1);
theta.l2 = reshape(nn_params(hidden_layer_size*(input_layer_size+1)+1:end),...
                   num_labels,hidden_layer_size+1);
toc
%Calculate training dataset accuracy -> compare mine to professor's               
load ex3weights.mat
prof_pred = predict(Theta1, Theta2, X);
mine_pred = predict(theta.l1, theta.l2, X);

fprintf('\nProf. Training Set Accuracy: %f\n', mean(prof_pred (:) == y(:)) * 100);
fprintf('\nMy Training Set Accuracy: %f\n', mean(mine_pred (:) == y(:)) * 100);
               
%  To give an idea of the network's output, we can run
%  through the examples one at the a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(theta.l1,theta.l2,X(rp(i),:));
    tru = y(rp(i),1);
    
    fprintf('\n NN Prediction: %d (True label %d) \n', pred, tru )
    
     % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
    
end

    