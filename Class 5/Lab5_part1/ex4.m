%% ML Lab 5: Neural Network Training
% %You will need to complete the following function

%    sigmoidGradient.m 

%% Initialization
clear all; close all; clc

%% Setup the NN parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load ex4data1.mat
    
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));


%% ================ Part 2: Loading Parameters ================
% Here, we load some pre-initialized NN parameters.
% Load the weights into variables Theta1 and Theta2
load ex4weights.mat

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];


%% ================ Part 3: Sigmoid Gradient  ================
%  Before NN implementation, complete the code in 
%sigmoidGradient.m file to compute the gradient
%
fprintf('\nEvaluating sigmoid gradient...\n')
g = sigmoidGradient([-1 -0.5 0 0.5 1]);

fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');


%% Part 4: Compute Cost (Feedforward path of NN training) ==========
%  Implementation of the feedforward part of the NN that returns the cost only.
%  nnCostFunction.m to returns the cost. 
% Compute the unregularized cost. Regularization parameter=0
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
                   
fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.287629)\n'], J);


%% ==== Part 5: Implement Regularization ===============

fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% Compute the cost with regularization parameter=1
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);

%% ================ Part 6: Initializing Pameters ================
%  2 layer ANN to classify digits.
%  Implement a function to initialize the NN weights(randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% === Part 7: Training NN ===================
%  To train the NN, we use "fmincg", which works similarly to "fminunc". 
%
fprintf('\nTraining Neural Network... \n')

%  After you tested the code with MaxIter=50, change MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


%% ===== Part 8: Visualize Weights =================
%  You can now "visualize" what the NN is learning by displaying 
%the hidden units to see what features they are capturing in  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

%% ====== Part 9: Implement Predict =================
%  After training the NN, we use it to predict the labels. 
% Implement the "predict" function to use the trained NN
%to predict the labels of the training set. 
%This computes the training accuracy !!!

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred(:) == y)) * 100);

rp = randperm(size(X,1));

for i = 1:size(X,1)
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (true digit %d)\n', pred, y(rp(i)) )
    
    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end


%% ================= Part 10 Train/Test data sets and errors =================

%Divide data into test and train sub sets, and compute the training and testing errors. 
%For example choose randomly 4000 images for training and 1000 images for testing.

%randomize dataset
sel = randperm(size(X, 1));%returns mixed indexes

%first 4000 for training
X_train = X(sel(1:4000),:);
y_train = y(sel(1:4000),:);
%last 1000 for testing
X_test = X(sel(4001:end),:);
y_test = y(sel(4001:end),:);

%------------------TRAINING-----------------
fprintf('\nTraining Neural Network... \n')

%  After you tested the code with MaxIter=50, change MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 500);

%  You should also try different values of lambda
lambda = 10;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%------------------TESTING-----------------
pred = predict(Theta1, Theta2, X_train);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred(:) == y_train)) * 100);

pred = predict(Theta1, Theta2, X_test);

fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred(:) == y_test)) * 100);











