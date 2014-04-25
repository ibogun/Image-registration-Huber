function [ c,cvx_optval] = robustLeastSquaresCVX( d1,d2,M,variant,debug)
%ROBUSTLEASTSQUARESCVX Solves robust least squares problem with Huber penalty function.
%   The problem to be solved is sum_i huber(A_i x - b_,M) where huber is Huber
%   function and M is the threshold in it. All 3 problems are solved using
%   cvx convex solver. Consult [2] p.195 about how problems are setup.
%  
%   Note: CVX software has to be installed to run the code below.
%   Link: http://cvxr.com/cvx/
%
% author: Ivan Bogun
% date  : April 17, 2014
%% References
% [2] Boyd, Stephen P., and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.
if (nargin<4)
    debug=0;
end

d=[ones(1,size(d1,2));d1];
n=size(d,2);
switch variant
    case 1
        %% Robust least squares variant 1
        
        if (~debug)
            cvx_begin quiet;
        else
            cvx_begin
        end
        variables c(2,3);
        minimize (sum(huber_circ(d2-c*d,2,M)));
        cvx_end
        
        %% variant 2
    case 2
        
        if (~debug)
            cvx_begin quiet;
        else
            cvx_begin
        end
        variables c(2,3) w(1,n);
        
        minimize (sum(quad_over_lin((d2-c*d),w+ones(1,n)))+M^2*sum(w));
        w>=0;
        
        cvx_end
        
        %% variant 3
    case 3
        if (~debug)
            cvx_begin quiet;
        else
            cvx_begin
        end
        variables c(2,3) u(1,n) v(1,n);
        
        minimize (sum(u.^2 + 2*M*v));
        subject to
        norms(d2-c*d,2,1)<=u+v;
        0<=u<=M;
        v>=0;
        cvx_end
        
        
end

end

