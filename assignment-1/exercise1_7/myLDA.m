function A = myLDA(Samples, Labels, NewDim, species)
% Input:    
%   Samples: The Data Samples 
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA
[NumSamples NumFeatures] = size(Samples);

	A=zeros(NumFeatures,NewDim);
    
    NumLabels = length(Labels);
    if(NumSamples ~= NumLabels)
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes);  %The number of classes

    IRIS(1,:,:) = Samples(strcmp(species,'setosa'), :);		    %Samples of Class 0
    IRIS(2,:,:) = Samples(strcmp(species,'versicolor'), :); 			%Samples of Class 1
    IRIS(3,:,:) = Samples(strcmp(species,'virginica'), :);			%Samples of Class 2

    

    Sw = 0;
    m0 = 0;

    %For each class i
    for i=1:NumClasses
	%Find the necessary statistics
    
    %Calculate the Class Prior Probability
	P(i)=length(IRIS(i))/NumSamples;

    %Calculate the Class Mean 
	mu(i,:) = mean(IRIS(i),1);

    temp(:,:) = IRIS(i,:,:) ;
    
    %Calculate the Within Class Scatter Matrix
	Sw=Sw+P(i)*cov(temp);

    %Calculate the Global Mean
	m0=m0 + mu(i,:);
    
    end

    m0 = m0/3;
    t(1,:) = mu(1,:)-m0;
    t(2,:) = mu(2,:)-m0;
    t(3,:) = mu(3,:)-m0;

    %Calculate the Between Class Scatter Matrix
	Sb= P(1)*transpose(t(1,:))*t(1,:)+P(2)*transpose(t(2,:))*t(2,:)+P(3)*transpose(t(3,:)*t(3,:));

    
    
    %Eigen matrix EigMat=inv(Sw)*Sb
    EigMat = inv(Sw)*Sb;
    
    %Perform Eigendecomposition
    [Q,L] = eig(EigMat);
    
    eigenval = diag(L);
    [eigenval, ind] = sort(eigenval, 1, 'descend');
    Q = Q(:, ind);
   
    
    %Select the NewDim eigenvectors corresponding to the top NewDim 
     %eigenvalues (Assuming they are NewDim<=NumClasses-1)
    
    A = Q(:,1:NewDim);
    
   %% You need to return the following variable correctly.
	% Return the LDA projection vectors
