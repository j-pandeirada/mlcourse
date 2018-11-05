%ML:  Support Vector Machine (SVM)
% 
%gaussianKernel.m - Gaussian kernel for SVM (you need to finish this function)
%dataset3Params.m - Parameters to use for Dataset 3 (you need to slightly change this function)

%% Initialization
clear ; close all; clc

%% ======== Part 1: Load and Plot Dataset 1 (linearly separable) ================
% Load from ex6data1.mat: You will get X, y 
load ex6data1.mat
plot(X(y==1,1),X(y==1,2),'k+')
hold on
plot(X(y==0,1),X(y==0,2),'ro')
 
%% ============ Part 2: Training Linear SVM ====================
%  The following code train a linear SVM on the dataset and plot the
%  decision boundary learned.

% Change C value and see how the decision boundary varies (e.g., try C = 1000)
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, model);

%% ==== Part 3: Implementing Gaussian Kernel ===============
%Complete the code in gaussianKernel.m to implement Gaussian kernel and
%test it with the following example 

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :' ...
         '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n'], sigma, sim);


%% ===== Part 4: Load and plot Dataset 2 (nonlinearly separable) ================
% Load from ex6data2: You will get X, y
load ex6data2.mat
plot(X(y==1,1),X(y==1,2),'k+')
hold on
plot(X(y==0,1),X(y==0,2),'ro')

%% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
%  Use the implemented kernel to train the SVM classifier.

% SVM Parameters
C = 1.3; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run
% faster. In practice, you will want to run the training to convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%figure 
visualizeBoundary(X, model);

%% ===Part 6: Load and plot Dataset 3 (nonlinearly separable) ================
% Load from ex6data3: You will get X, y, Xval, yval
load ex6data3.mat
plot(X(y==1,1),X(y==1,2),'k+')
hold on
plot(X(y==0,1),X(y==0,2),'ro')

%% Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

% Try different SVM Parameters here
[C, sigma, error] = dataset3Params(X, y, Xval, yval);

% Train the SVM with the optimal C and sigma parameters
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, model);
