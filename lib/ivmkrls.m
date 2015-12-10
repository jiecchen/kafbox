% Kernel Recursive Least Squares algorithm with Approximate Linear
% Dependency criterion
%
% Y. Engel, S. Mannor, and R. Meir, "The kernel recursive least-squares
% algorithm," IEEE Transactions on Signal Processing, vol. 52, no. 8, pp. 
% 2275-2285, Aug. 2004, http://dx.doi.org/10.1109/TSP.2004.830985
%
% Comment: implementation includes a maximum dictionary size M
%
% This file is part of the Kernel Adaptive Filtering Toolbox for Matlab.
% http://sourceforge.net/projects/kafbox/

classdef ivmkrls
    
    properties (GetAccess = 'public', SetAccess = 'private')
        M = 100; % maximum dictionary size
        kerneltype = 'gauss'; % kernel type
        kernelpar = 1; % kernel parameter
        beta = 0.8; 
        opt = 10;
    end
    
    properties (GetAccess = 'public', SetAccess = 'private')
        dict = []; % dictionary
        alpha = []; % expansion coefficients
        P = []; % inverse A'*A
        Kinv = []; % inverse kernel matrix
        K = []; % kernel matrix
    end
    
    methods
        
        function kaf = ivmkrls(parameters) % constructor
            if (nargin > 0) % copy valid parameters
                for fn = fieldnames(parameters)',
                    if ismember(fn,fieldnames(kaf)),
                        kaf.(fn{1}) = parameters.(fn{1});
                    end
                end
            end
        end
        
        function y_est = evaluate(kaf,x) % evaluate the algorithm
            if size(kaf.dict,1)>0
                k = kernel(kaf.dict,x,kaf.kerneltype,kaf.kernelpar);
                y_est = k'*kaf.alpha;
            else
                y_est = zeros(size(x,1),1);
            end
        end
        
        function kaf = train(kaf,x,y) % train the algorithm
            k = kernel([kaf.dict; x],x,kaf.kerneltype,kaf.kernelpar);
            kt = k(1:end-1);
            ktt = k(end);
            if numel(kt)==0, % initialize
                kaf.Kinv = 1/ktt;
                kaf.K = ktt;
                kaf.alpha = y/ktt;
                kaf.P = 1;
                kaf.dict = x;
            else
                at = kaf.Kinv*kt;
                delta = ktt - kt'*at;
                K_new = [kaf.K, kt; kt', ktt]; % expaned kernel matrix
                sz = size(kaf.dict, 1);
                log_det_old = logdet(kaf.beta * kaf.K + eye(sz));
                marginal_gain = logdet(kaf.beta * K_new + eye(sz + 1)) - log_det_old;
                
                
                if (sz < kaf.M  &&  marginal_gain > (kaf.opt/2 - log_det_old) / (kaf.M - sz)), % expand
                    kaf.dict = [kaf.dict; x];
                    fprintf('size of dict = %d\n', size(kaf.dict, 1));
                    kaf.Kinv = 1/delta*[delta*kaf.Kinv + at*at', -at; -at', 1];
                    Z = zeros(size(kaf.P,1),1);
                    kaf.P = [kaf.P Z; Z' 1];
                    ode = 1/delta*(y-kt'*kaf.alpha);
                    kaf.alpha = [kaf.alpha - at*ode; ode];
                    kaf.K = K_new;
                else % only update alpha
                    q = kaf.P*at/(1+at'*kaf.P*at);
                    kaf.P = kaf.P - q*(at'*kaf.P);
                    kaf.alpha = kaf.alpha + kaf.Kinv*q*(y-kt'*kaf.alpha);
                end
            end
        end
        
    end
end
