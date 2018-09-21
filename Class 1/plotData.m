function [] = plotData(x,y)
%SHOW SCATTER PLOT OF DATA
%SHOW HISTOGRAM OF X
%SHOW HISTOGRAM OF Y
%ALL PROPERLY TITLED AND LABELED

figure,
subplot(2,2,[1 3])
plot(x,y,'rx'),xlabel('x'),ylabel('y');
title('file data.txt'),legend('data points');
subplot(222)
hist(y),title('histogram of y');
subplot(224)
hist(x),title('histogram of x');

end

