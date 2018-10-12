%% ML lab work -Unregularized  Logistic Regression
%Complete the following functions in this exericse:
%
%     plotData.m
%     costFunction.m
%     predict.m

%% Initialization
clear ; close all; clc

%% Step 1: Load and Plot Data from file ex2data1.txt
%  The first two columns contains the exam scores and the third column
%  contains the label.
load ex2data1.txt

X = ex2data1(:,1:2);
y = ex2data1(:,3);

fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

plotData(X, y);

% Add labels and legend
xlabel('Exam 1 score')
ylabel('Exame 2 score')

% Specified in the plot order
legend('Admitted','Not admitted')
hold off;

%% ============ Part 2: Compute Cost and Gradient ============
%  Implement the cost and gradient for logistic regression. 
%Complete the code in costFunction.m

%  Data matrix dimension
[m, n] = size(X);

% Add extra FIRST column of 1 to X
X = [ones(m,1) X];

% Initialize fitting parameters =0
initial_theta = zeros(n+1,1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);


%% ============= Part 3: Optimizing using fminunc  =============
%  You will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary curve
plotDecisionBoundary(theta, X, y);
hold on;

% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

%% ============== Part 4: Predict New data and Train Accuracy ==============
%  After learning the parameters, you'll use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.
%
probNewData = 1./(1+exp(-([1 45 85]*theta)))*100;
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n\n'], probNewData);

% Compute accuracy on the training set
%  Complete the code in predict.m
p= predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p==y)) * 100);

