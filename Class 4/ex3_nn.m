% ML Lab work, Part 2: Multiclass classification (One-vs-all) with Neural Networks

%You will need to complete the following functions in this exericse:
%     predict.m

%% Initialization
clear ; close all; clc

%% Setup the parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10  (0 is mapped to label 10)

%% =========== Part 1: Load and Visualize Some Images =============
% Load Training Data
load ex3data1.mat
m = length(y);%number of examples

% Randomly select 100 data points to display
sel = randperm(m);
sel = sel(1:100);

displayData(X(sel, :));

%% ================ Part 2: Loading NN Pameters ================
% Here, we load some pre-trained neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('ex3weights.mat');

%% ================= Part 3: Implement Predict =================
%  After training the NN, we would like to use it to predict the labels. 
%You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(pred (:) == y(:)) * 100);

%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1,Theta2,X(rp(i),:))
    tru = y(rp(i),1);
    
    fprintf('\n NN Prediction: %d (True label %d) \n', pred, tru )
    
     % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
    
end
