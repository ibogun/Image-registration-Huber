function pass = runTests
%UNTITLED3 Testcases for the interior point method solving robust least
%squares (with Huber penalty function)
%   Functions to be tested: gradient, Hessian, solution vector, optimal
%   value
%
% author: Ivan Bogun
% date  : April 17, 2014

% Generate Data
%------------------------------------------------------------------------
n=50;
m=80;

seed=1;
rng(seed,'twister');

A=randn(m,n);
b=randn(m,1);
M=100;
x=ones(n,1);
w=ones(m,1);
lambda=ones(m,1);


%------------------------------------------------------------------------


% Parameters
%------------------------------------------------------------------------
mu=10;
epsilon=10e-10; % tolerance
alpha=0.05; % typically chosen from  0.01 to 0.1
beta=0.45;  % typically chosen from  0.3 to 0.8

% testing parameter
tol=1e-5; % threshold
%------------------------------------------------------------------------

% Problem setup
objective=@(x,w) (A*x-b)'*((A*x-b)./(1+w))+M^2*ones(m,1)'*w;
obj1=@(x_) objective(x_,w);
obj2=@(w_) objective(x,w_);

% Gradient
%------------------------------------------------------------------------
% gradient in x
grad_x=@(x_,w_) A'*(2*(A*x_-b)./(1+w_));
% gradient in w
grad_w=@(x_,w_) -((A*x_-b).^2)./((1+w_).^2)+ones(m,1)*M^2;
% total grad in both
grad=@(x_,w_) [grad_x(x_,w_);grad_w(x_,w_)];
%------------------------------------------------------------------------

% duality gap
%------------------------------------------------------------------------
%nu=-f(x)'lambda -> w*lambda
nu_hat=@(w_,l_) w_'*l_;
nu=nu_hat(w,lambda);
%------------------------------------------------------------------------

% Hessian
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

options = optimset;
options.Display='off';
x_test=fminunc(obj1,x,options);

%assert(norm(grad_x(x_test,w))<0.1,'Grad in x didnt pass testcase');

%  Hessian Testing business
%------------------------------------------------------------------------
a_11=zeros(n,n);
a_22=zeros(m,m);
a_33=zeros(m,n);

for k=1:n
    for j=1:n
        for i=1:m
            a_11(j,k)=a_11(j,k)+2*(A(i,j)*A(i,k))/(1+w(i));
        end
    end
    
    for j=1:m
        a_33(j,k)=-2*((A(j,:)*x-b(j))*A(j,k))/(1+w(j))^2;
    end
end

for j=1:m
    a_22(j,j)=2*(A(j,:)*x-b(j))^2/(1+w(j))^3;
end
%------------------------------------------------------------------------
assert(norm(a_11-A_11(x,w))<tol,'Upper left part of the Hessian didnt pass');
assert(norm(a_22-A_22(x,w))<tol,'Upper right & lower left part of the Hessian didnt pass');
assert(norm(a_33-A_33(x,w))<tol,'Lower right part of the Hessian didnt pass');

params.debug=1;
[x_test,w_test,lambda_test,optval_test]=solveRobustLeastSquares(A,b,M,params);

cvx_begin quiet
cvx_precision best
variables x(n,1) w(m,1);

minimize (sum(quad_over_lin((A*x-b),w+ones(m,1),2)+M^2*sum(w)));
subject to
w>=0;

cvx_end

assert(norm(x-x_test)<tol,'X didnt pass testcase');
assert(norm(w-w_test)<tol,'w didnt pass testcase');
assert(norm(optval_test-cvx_optval)<tol,'Optimal value didnt pass testcase');

fprintf('All testcases passed! \n');

end

