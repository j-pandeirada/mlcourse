%% ML Ex. Linear Regression with multiple features

%  You will need to complete the following functions:
%     featureNormalize.m

%%  Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Step 1: Load Data from file ex1data2.txt
data = load('ex1data2.txt');
X = data(:,1:2);
y = data(:,3);
m = length(y)  % number of training examples

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

%% Normalize features (step 2)
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add a column of ones to X
X = [ones(length(X),1) X];

%%  Step 3: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.3;
num_iters = 100;

% Initialize vector of parameters Theta = zeros
theta = zeros(size(X,2),1);

% compute and display the cost
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Plot the cost J_history
plot(1:length(J_history),J_history)
xlabel('Number of iterations');
ylabel('Cost J');

% Display the final theta result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

%Repeat the learning by changing the learning rate (Step 4) 
alpha = [0.01,0.03,0.1,0.3];
figure
xlabel('Number of iterations');
ylabel('Cost J');
for i=1:length(alpha)
    % Initialize vector of parameters Theta = zeros
    theta = zeros(size(X,2),1);
    % compute and display the cost
    [theta, J_history] = gradientDescent(X, y, theta, alpha(i), num_iters);
    plot(1:length(J_history),J_history)
    hold on
end    
%%  ssd
% Estimate the price of a 1650 sq-ft, 3 br house
% YOUR CODE HERE ======================
%test data
X_test= [1650 3];
%normalize test data
X_testn = (X_test-mu(1,:))./sigma(1,:);
%add one more collums and predict the price
X_testn = [1 X_testn];
price = X_testn*theta;
% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

