function v = fisherLinearDiscriminant(X1, X2)

    m1 = size(X1, 1);
    m2 = size(X2, 1);

    mu1 = mean(X1(1,:));  % mean value of X1
    mu2 = mean(X2(1,:)); % mean value of X2

   Nx1 = length(X1);  %Number of samples of X1 class
   Nx2 = length(X2);  %Number of samples of X2 class
   N = Nx1 + Nx2;     %Number of total samples

    P1 = Nx1/N; %Prior of X1 class
    P2 = Nx2/N; %Prior of X2 class

    S1 =cov(X1); % scatter matrix of X1
    S2 =cov(X2); % scatter matrix of X2

    Sw =  P1*S1 +  P2*S2;% Within class scatter matrix

    v = inv(Sw)*(mu1-mu2); % optimal direction for maximum class separation 

    v = v/norm(v); % return a vector of unit norm
