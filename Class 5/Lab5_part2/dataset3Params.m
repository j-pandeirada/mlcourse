function [C, sigma, error, var1, var2] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns the optimal (C, sigma) learning parameters to use for SVM
%with Gaussian kernel  based on the min error of the validation set Xval, yval. 


%Choose a number of values for C and sigma
var= [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]

%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:length(var)
  for j=1:length(var)
    
    C=var(i);
    sigma=var(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    error(i,j)=mean(double(predictions ~= yval));
    indexC(i,j)=C;
    indexsigma(i,j)=sigma;
  end
end

[min_error,ind]=min(error(:));
var1=indexC(:);
var2=indexsigma(:);
C=var1(ind);
sigma=var2(ind);
% =========================================================================

end