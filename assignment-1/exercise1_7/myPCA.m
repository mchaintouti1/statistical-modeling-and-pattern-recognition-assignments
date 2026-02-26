function [U, S] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = principalComponentAnalysis(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE GOES HERE ======================
% Instructions: You should first compute the covariance matrix. Then, 
%  compute the eigenvectors and eigenvalues of the covariance matrix. 
%
% Note that the dataset X is normalized, when calculating the covariance

%covariance = (1/m)*(transpose(X)*X); %covariance matrix
covariance = (transpose(X)*X); %covariance matrix

[U,S] = eig(covariance);
eigenval = diag(S);
[eigenval, ind] = sort(eigenval, 1, 'descend');
U = U(:, ind);
S = diag(eigenval);

% =========================================================================

end
