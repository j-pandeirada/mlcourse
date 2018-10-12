%% ML lab work: Logistic Regression with regularization
%You will need to complete the following function in this exercise:
%     costFunctionReg.m

%% Initialization
clear ; close all; clc

%% Load and plot Data

%% Step 1: Load and Plot Data from file ex2data2.txt
%  The first two columns contains the X values and the third column
%  contains the label (y).
load ex2data2.txt

X = ex2data2(:,1:2);
y = ex2data2(:,3);

plotData(X, y);

% Put labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y=1','y=0')
hold off;

%% =========== Part 1: Regularized Logistic Regression ============
%  This dataset is not linearly separable. 
%However, you can still use logistic regression to classify data. 
%
%  To do so, you introduce more features -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%
% Note that mapFeature also adds a column of ones, so the intercept term is handled
X = mapFeature(X(:,1), X(:,2)); 

%  Data matrix dimension
[m, n] = size(X);

% Initialize fitting parameters =0
initial_theta = zeros(n,1);

% Set regularization parameter lambda
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);


%% ============= Part 2: Regularization and Accuracies =============
%%  Try different values of lambda and 
%  see how regularization affects the decision boundary
%
%  For example lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda (you should vary this)
lambda = [0 1 10 100];

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
for i=1:length(lambda),
    % Optimize
    [theta, J, exit_flag] = ...
        fminunc(@(t)(costFunctionReg(t, X, y, lambda(i))), initial_theta, options);

    % Plot Boundary
    plotDecisionBoundary(theta, X, y);
    hold on;
    title(sprintf('lambda = %g', lambda(i)))

    % Labels and Legend
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')

    legend('y = 1', 'y = 0', 'Decision boundary')
    hold off;

    % Compute accuracy on our training set
    p = predict(theta, X);
    fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
end

%% Using proper lambda and predicting

% Set regularization parameter lambda (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
    [theta, J, exit_flag] = ...
        fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

%predict a new example: Xtest=(-0.25, 1.5), y=0 (rejected)
%Do not forget to apply first the mapFeature function
Xtest = [-0.25 1.5];
Xtest = mapFeature(Xtest(:,1), Xtest(:,2)); 
p_test= predict(theta,Xtest)


