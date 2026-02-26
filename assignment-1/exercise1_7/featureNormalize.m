function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
[m,n] = size(X);

mu = mean(X,1);

sigma = std(X,1); % standart deviation of each column

for i=1:n
    X_norm(:,i) =  (X(:,i)-mu(i))/sigma(i); %normalize each column independently
end

% ============================================================

end
