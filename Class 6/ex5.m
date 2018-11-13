%% ML  Lab - Regularized Linear Regression and Bias-Variance

%% You will need to complete the following functions:

%     linearRegCostFunction.m
%     polyFeatures.m 
%     validationCurve.m

clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ex5data1.mat

% Number of examples
m = size(X, 1);

% Plot training data X y
plot(X,y,'rx')
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

%% =========== Part 2: Regularized Linear Regression Gradient =============
%  You should implement the cost and gradient for regularized linear regression.
%
theta = [1 ; 1];
lambda=1;
[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, lambda);

fprintf(['Cost at theta = [1 ; 1]: %f '...
         '\n(this value should be about 303.993192)\n'], J);
         
fprintf(['Gradient at theta = [1 ; 1]:  [%f; %f] '...
         '\n(this value should be about [-15.303016; 598.250744])\n'], ...
         grad(1), grad(2));

     
%% =========== Part 3: Train Linear Regression =============
%  Once you have implemented the cost and gradient correctly, the
%  trainLinearReg function will use your cost function to train 
%  regularized linear regression.
% 
% Note: The data is non-linear, so this will not give a great fit.
%
%  Train linear regression with lambda = 0
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

%  Plot fit over the data
figure
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;

%% =========== Part 4: Learning Curve for Linear Regression =============
%  Note: Since the model is underfitting the data, we expect to
%                 see a graph with "high bias" 
%
lambda = 0;
[error_train, error_val] = ...
  learningCurve([ones(m, 1) X], y,[ones(size(Xval, 1), 1) Xval], yval,lambda);

figure
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end


%% =========== Part 5: Feature Mapping for Polynomial Regression =============
%  One solution to this is to use polynomial regression. You should now
%  complete polyFeatures to map each example into its powers
%
p = 8;

% Map each onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
%One way to implement feature normalization code 
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)  
%Second way to implement feature normalization code 
X_poly_test = polyFeatures(Xtest, p);
[rows,col]=size(X_poly_test);
X_poly_test=X_poly_test-repmat(mu,rows,1);
X_poly_test=X_poly_test./repmat(sigma,rows,1);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test]; % Add Ones


% Map X_poly_val and normalize (using mu and sigma)
%Third way to implement feature normalization code 
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

fprintf('Normalized Training Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));

%% =========== Part 7: Learning Curve for Polynomial Regression =============
%  Now, you will experiment polynomial regression with multiple
%  values of lambda. The code below runs polynomial regression with 
%  lambda = 0. You should try running the code with different values of
%  lambda to see how the fit and learning curve change.
%

lambda = 2;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
figure;
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

figure;
[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

%% =========== Part 8: Selecting the best Lambda =============
%  You will now implement validationCurve to test various values of lambda on a validation set. 
%You will then use this to select the "best" lambda value.
%
[lambda_vec, error_train, error_val] = validationCurve(X_poly, y, X_poly_val, yval);

figure
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

%% =========== Part 9: Final results for best lambda =============
lambda=3;
[theta] = trainLinearReg(X_poly, y, lambda);

%Compute the final train, validation, test errors (consult learningCurve.m)
error_train =(X_poly*theta-y)'*(X_poly*theta-y)/(2*m);
error_val =(X_poly_val*theta-yval)'*(X_poly_val*theta-yval)/(2*m);
error_test = (X_poly_test*theta-ytest)'*(X_poly_test*theta-ytest)/(2*m);

fprintf('Training error:%f\nValidation error:%f\nTest error:%f\n',...
            error_train,error_val,error_test);




