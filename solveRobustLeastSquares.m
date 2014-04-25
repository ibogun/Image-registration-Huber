function [ x,w,lambda,objectiveValue ] = solveRobustLeastSquares( A,b,M,params )
%solveRobustLeastSquares Solves robust least squares problem with Huber penalty function.
%   The problem to be solved is sum_i phi(A_i x - b,M) where phi is Huber
%   function and M is the threshold in it.
%
%   Input:
%
%       A                       -               [m,n]  data matrix
%       b                       -               [m] data vector
%       M                       -               Huber function parameter
%       params (optional)       -               parameters
%
%           mu                  -               mu parameter in the
%                                           interior-point method
%           epsilon             -               tolerance of the
%                                           convergence
%           alpha               -               line search parameter
%           beta                -               line search parameter
%           debug               -               boolean 1 if output should
%                                           be printed
%   Output:
%
%       x                       -               solution vector x
%       w                       -               weights if large -> outlier
%       lambda                  -               solution of the dual
%       objectiveValue          -               value of the objective
%                                           function
%


%% Initial feasible point
% Algorithm has to start from a feasible point. In our case 
%
% $$x=(n,1) $$
%

[m,n]=size(A);
x=ones(n,1);
w=2*ones(m,1);
lambda=ones(m,1);

if (nargin<4)
    
    % Parameters
    %------------------------------------------------------------------------
    mu=10;
    epsilon=10e-8; % tolerance
    alpha=0.05; % typically chosen from  0.01 to 0.1
    beta=0.7;  % typically chosen from  0.3 to 0.8
    debug=0;
    %------------------------------------------------------------------------
else
    
    if (~isfield(params,'mu'))
        mu=10;
    else
        mu=params.mu;
    end
    
    if (~isfield(params,'epsilon'))
        epsilon=10e-8;
    else
        epsilon=params.epsilon;
    end
    
    if (~isfield(params,'alpha'))
        alpha=0.05;
    else
        alpha=params.alpha;
    end
    
    if (~isfield(params,'beta'))
        beta=0.45;
    else
        beta=params.beta;
    end
    
    if (~isfield(params,'debug'))
        debug=0;
    else
        debug=params.debug;
    end
end

%% Problem setup
objective=@(x,w) (A*x-b)'*((A*x-b)./(1+w))+M^2*ones(m,1)'*w;

%% Gradient
%------------------------------------------------------------------------
% gradient in x
grad_x=@(x_,w_) A'*(2*(A*x_-b)./(1+w_));
% gradient in w
grad_w=@(x_,w_) -((A*x_-b).^2)./((1+w_).^2)+ones(m,1)*M^2;
% total grad in both
grad=@(x_,w_) [grad_x(x_,w_);grad_w(x_,w_)];
%------------------------------------------------------------------------

%% duality gap
%------------------------------------------------------------------------
%nu=-f(x)'lambda -> w*lambda
nu_hat=@(w_,l_) w_'*l_;
nu=nu_hat(w,lambda);
%------------------------------------------------------------------------

%% Hessian
%------------------------------------------------------------------------
% part 1: Hessian in x
A_11=@(x_,w_) 2*A'*(A.*repmat(1./(1+w_),1,n));               % size [n,n]
% part 2: Hessian in w
A_22=@(x_,w_) 2*diag(((A*x_-b).^2)./((1+w_).^3));             %  size [m,m]
% part 3: Hessian in x,w
A_33=@(x_,w_) -2*repmat(((A*x_-b)./((1+w_).^2)),1,n).*A; % size [m,n]
% combine Hessian
H=@(x_,w_) [A_11(x_,w_),A_33(x_,w_)';A_33(x_,w_),A_22(x_,w_)];% size [m+n,m+n]
%------------------------------------------------------------------------

% matrix of derivatives
D=zeros(m,n+m);
D(1:m,n+1:n+m)=diag(-ones(m,1));

%% Residuals
%------------------------------------------------------------------------
r_dual=@(x_,w_,lambda_) grad(x_,w_)+D'*lambda_;
r_cent=@(x_,w_,lambda_,t_) -diag(lambda_)*(-w_)-(1/t_)*ones(m,1);
%------------------------------------------------------------------------

% KKT conditions
%------------------------------------------------------------------------
r_t=@(x_,w_,lambda_,t_) [grad(x_,w_)+D'*lambda_;...
    -diag(lambda_)*(-w_)-(1/t_)*ones(m,1)];
%------------------------------------------------------------------------
t=1;

iteration=1;
if (debug)
    fprintf('----------------------------------------------------------------------------\n');
    fprintf('Interior point Primal-Dual algorithm with alpha= %1.2f, beta= %1.2f. \n',...
        alpha,beta);
    fprintf('----------------------------------------------------------------------------\n');
    fprintf(' iter|    dual gap     |     r_cent      |     r_dual      |   objective     \n');
    fprintf('----------------------------------------------------------------------------\n');
end

while (abs(nu)>=epsilon || abs(norm(r_cent(x,w,lambda,t)))>=epsilon || ...
        norm(r_dual(x,w,lambda))>=epsilon)
    %% 1. Determine t
    t=mu*m/nu;
    
    %% 2. Compute primal-dual search direction
    Aeq=[H(x,w),D';-diag(lambda)*D,-diag(-w)];
    beq=-[r_dual(x,w,lambda);r_cent(x,w,lambda,t)];
    
    % solve linear system
    x_new=linsolve(Aeq,beq);
    
    % get variables from the solution
    delta_x=x_new(1:n);
    delta_w=x_new(n+1:n+m);
    delta_lambda=x_new(n+m+1:end);
    
    %% 3. Line search and update
    neg_lambda=delta_lambda<0;
    
    h=(-lambda./delta_lambda);
    if (isempty(min(h(neg_lambda))))
        s_max=1;
    else
        s_max=min(1,min(h(neg_lambda)));
    end
    
    s=0.99*s_max;
    
    v1=norm(r_t(x,w,lambda,t));
    v2=norm(r_t(x+s*delta_x,w+s*delta_w,lambda+s*delta_lambda,t));
    
    % using both conditions for line search (p.613):
    % 1) f(x^+) < 0
    % 2) ||r_t(x^+,w^+,lambda^+)||_2<=(1-beta*s)||r_t(x,w,lambda)||_2
    while (v2>(1-alpha*s)*v1 || any(w+s*delta_w<0))
        s=beta*s;
        v2=norm(r_t(x+s*delta_x,w+s*delta_w,lambda+s*delta_lambda,t));
        
    end
    
    % update
    x=x+s*delta_x;
    w=w+s*delta_w;
    
    lambda=lambda+s*delta_lambda;
    
    nu=nu_hat(w,lambda);
    
    if (debug)
        
        val1=abs(nu);
        val2=norm(r_cent(x,w,lambda,t));
        val3=norm(r_dual(x,w,lambda));
        val4=objective(x,w);
        
        str1=formatString(val1);
        str2=formatString(val2);
        str3=formatString(val3);
        str4=formatString(val4);
        
        combined=['%3d  |' str1 '|' str2 '|' str3 '|' str4 '\n'];
        
        fprintf(combined,iteration,...
            abs(nu),norm(r_cent(x,w,lambda,t)),norm(r_dual(x,w,lambda)),objective(x,w));
        % fprintf('%3d  | %4.8f  |  %4.8f  | %4.8f  |  %4.8f\n',iteration,...
        %    abs(nu),norm(r_cent(x,w,lambda,t)),norm(r_dual(x,w,lambda)),objective(x,w));
        
    end
    iteration=iteration+1;
end

objectiveValue=objective(x,w);

if (debug)
    fprintf('Optimal value found %2.5f \n',objectiveValue);
end

end

function res=formatString(value)
value=abs(value);
res=' %5.8f ';

if(le(value,100000)&&ge(value,10000))
    res=[' ' res ''];
elseif(le(value,10000)&&ge(value,1000))
    res=[' ' res ' '];
    
elseif(le(value,1000)&&ge(value,100))
    res=[' ' res '  '];
    
elseif (le(value,100)&&ge(value,10))
    res=[' ' res '   '];
    
elseif (le(value,10))
    res=[' ' res '    '];
    
end


end

