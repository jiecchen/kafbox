% 1-step ahead prediction on Lorenz attractor time-series data
%
% This file is part of the Kernel Adaptive Filtering Toolbox for Matlab.
% http://sourceforge.net/projects/kafbox/

[X,Y] = kafbox_data(struct('name','Lorenz','embedding',6));

% make a kernel adaptive filter object of class ivmkrls with options: 
% ALD threshold 1E-4, Gaussian kernel, and kernel width 32
kaf =  ivmkrls(struct('nu',4E-5,'kerneltype','gauss','kernelpar',32, 'opt',30, 'M', 70));

%% RUN ALGORITHM
N = size(X,1);
Y_est = zeros(N,1);
for i=1:N,
    if ~mod(i,floor(N/10)), fprintf('.'); end % progress indicator, 10 dots
    Y_est(i) = kaf.evaluate(X(i,:)); % predict the next output
    kaf = kaf.train(X(i,:),Y(i)); % train with one input-output pair
end
fprintf('\n');
SE = (Y-Y_est).^2; % test error

%% OUTPUT
fprintf('MSE after first 1000 samples: %.2fdB\n\n',10*log10(mean(SE(1001:end))));
