function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%Calculating the cost
a = zeros(m);

H = sigmoid(transpose(theta)*transpose(X));
for i=1:m
    %H = 1/(1+ exp(-X(i,:)*theta));

    a(i) = -y(i)*log(H(i)) - (1-y(i))*log(1-H(i));
end

J = (1/m)*sum(a(:));

%Calculating the gradient of the cost with respect to theta
a = zeros(m, length(theta));
for j=1:size(theta)
    for i=1:m
        %H = 1/(1+ exp(-X(i,:)*theta));
        a(i,j) = X(i,j)*(H(i) - y(i));
    end
    grad(j) = (1/m)*sum(a(:,j));
end

% =============================================================

end
