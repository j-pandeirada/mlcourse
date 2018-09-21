clear all
close all
clc

%% 1 - Load and Plot the data
data = load('data.txt');
x = data(:,1);
y = data(:,2);
plotData(x,y);

%% 2 and 3 - Polynomial approximation and normalization
ord = 0;
err = 1000;
m = length(x); % number of data points

%data normalization
x_norm = x./(max(x));
y_norm = y./(max(y));

%APROXIMATE WITH VARIOUS POLYNOMIAL ORDERS UNTIL ERROR IS LESS THAN 10
while(err>10),
    ord = ord+1;
    
    %model for raw data
    f = polyfit(x,y,ord);
    y_est = polyval(f,x);
    
    %model for normalized data
    f_norm = polyfit(x_norm,y_norm,ord);
    y_est_norm = polyval(f_norm,x_norm);
    
    %calculated error for both models
    e_plot(ord) = (1/m)*(sum((y-y_est).^2));
    e_plot_norm(ord) = (1/m)*(sum((y_norm-y_est_norm).^2));
    err = e_plot(ord);
end


%PLOT MODELED DATA AND MSE GRAPHS
figure,
subplot(2,2,[1 3])
plot(x,y,'rx',x,y_est),xlabel('x'),ylabel('y');
title('file data.txt'),legend('data points','model');
subplot(222)
plot(1:length(e_plot),e_plot),title('MSE(original data'),xlabel('polynomial order');
subplot(224)
plot(1:length(e_plot_norm),e_plot_norm),title('MSE(normalized data'),xlabel('polynomial order');

%% 4 and 5

S = [x flip(x) (x+flip(x))./2];

%random data between [-1 1]
M = rand(5,4)*2 - 1; 
%data with mean = -2 and variance=0.5
N = -2 + sqrt(0.5).*randn(4,3);

P = M*N;
posi_perc = sum(sum(P>0))/numel(P) * 100;






















