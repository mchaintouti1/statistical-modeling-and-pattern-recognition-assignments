close all;
clear;
clc;

data_file = './data/mnist.mat';

data = load(data_file);

% Read the train data
[train_C1_indices, train_C2_indices,train_C1_images,train_C2_images] = read_data(data.trainX,data.trainY.');

% Read the test data
[test_C1_indices, test_C2_indices,test_C1_images,test_C2_images] = read_data(data.testX,data.testY.');

numOfData1 = length(train_C1_images);
numOfData2 = length(train_C2_images);

numOfTestData1 = length(test_C1_images);
numOfTestData2 = length(test_C2_images);
numOfTotalTestData = numOfTestData2+numOfTestData1;

aspectMatrix1 = zeros(numOfData1, 1);
aspectMatrix2 = zeros(numOfData2,1);

%% Compute Aspect Ratio


%Compute the aspect ratios of all images and store the value of the i-th image in aRatios(i)

for i=1:numOfData1
    img(:,:) = train_C1_images(i,:,:);
  aspectMatrix1(i) = computeAspectRatio(img,i);
end

for i=1:numOfData2
    img(:,:) = train_C2_images(i,:,:);
  aspectMatrix2(i) = computeAspectRatio(img,i);
end

min1 = min(aspectMatrix1);
min2 = min(aspectMatrix2);

if min1<min2
    minAspectRatio = min1;
else
    minAspectRatio = min2;
end

max1 = max(aspectMatrix1);
max2 = max(aspectMatrix2);

if max1>max2
    maxAspectRatio = max1;
else
   maxAspectRatio = max2;
end

message1 = (['Minimum aspect ratio: ',num2str(minAspectRatio)]);
message2 = (['Maximum aspect ratio: ',num2str(maxAspectRatio)]);
disp(message1);
disp(message2);

%% Bayesian Classifier


% Prior Probabilities
PC1 = length(train_C1_images)/(length(train_C2_images)+length(train_C1_images));
PC2 = length(train_C2_images)/(length(train_C2_images)+length(train_C1_images));

disp('...');
message1 = (['Prior Probability of C1: ',num2str(PC1)]);
message2 = (['Prior Probability of C2: ',num2str(PC2)]);
disp(message1);
disp(message2);

%Average Value
m1 = (1/length(train_C1_images))*sum(aspectMatrix1);
m2 = (1/length(train_C2_images))*sum(aspectMatrix2);

disp('...');
message1 = (['m1: ',num2str(m1)]);
message2 = (['m2: ',num2str(m2)]);
disp(message1);
disp(message2);

%Standard deviation
s1 = sqrt( (1/length(train_C1_images) )*(sum((aspectMatrix1 - m1).^2)));
s2 = sqrt((1/length(train_C2_images))*(sum((aspectMatrix2 - m2).^2)));

disp('...');
message1 = (['s1: ',num2str(s1)]);
message2 = (['s2: ',num2str(s2)]);
disp(message1);
disp(message2);


% Likelihoods
PgivenC1 = (1/sqrt(2*pi*s1^2))*exp((-1/(2*s1^2))*(aspectMatrix1-m1).^2);
PgivenC2 = (1/sqrt(2*pi*s2^2))*exp((-1/(2*s2^2))*(aspectMatrix2-m2).^2);

% Posterior Probabilities
PC1givenL = PC1*PgivenC1;
PC2givenL = PC2*PgivenC2;

aspectMatrix1 = sort(aspectMatrix1, 'descend');
aspectMatrix2 = sort(aspectMatrix2, 'descend');

% z = zeros( (length(aspectMatrix1)-length(aspectMatrix2)) , 1)+100;
% 
% if length(aspectMatrix2)<length(aspectMatrix1)
%     aspectMatrix2 = [aspectMatrix2; z];
% else
%     aspectMatrix1 = [aspectMatrix1; z];
% end
% 
% 
% 
% diff = aspectMatrix1-aspectMatrix2;
% diff = abs(diff);
% min = min(diff)
% find(diff==min,1)
% aspectMatrix1(5958)
% aspectMatrix2(5958)

a=s2^2-s1^2;
b=2*(s1^2)*m2-2*(s2^2)*m1;
c=(s2^2)*(m1^2)-(s1^2)*(m2^2)-2*(s1^2)*(s2^2)*(log(PC1/PC2)+log(sqrt((s2^2)/(s1^2))));

D = b^2-4*a*c;
x1 = (-b+sqrt(D))/(2*a);
x2 = (-b-sqrt(D))/(2*a);

disp('...');
message1 = (['x1: ',num2str(x1)]);
message2 = (['x2: ',num2str(x2)]);
disp(message1);
disp(message2);

labels1 = zeros(numOfTestData1, 1);
labels2 = zeros(numOfTestData2,1);

count_errors = 0;
aspect = 0;
testLabel=0;

% Classification result
for i=1:numOfTestData1
    img(:,:) = test_C1_images(i,:,:);
    aspect = computeAspectRatio(img,i);
    
    if(aspect<x1 | aspect>x2)
         testLabel = 0;
    else
        testLabel = 1;
    end

    if (testLabel ~= 0)
        count_errors = count_errors+1;
    else

    end
end

for i=1:numOfTestData2
    img(:,:) = test_C2_images(i,:,:);
    aspect = computeAspectRatio(img,i);
    
    if(aspect<x1 | aspect>x2)
         testLabel = 0;
    else
        testLabel = 1;
    end

    if (testLabel ~= 1)
        count_errors = count_errors+1;
    else

    end
end

disp('...');

accuracy = (1-count_errors/numOfTotalTestData)*100;

Error = 100 - accuracy;

disp(['Error percentage: ', num2str(Error),'%']);
disp(['Accuracy: ', num2str(accuracy), '%']);

% Count misclassified digits
%count_errors = sum(decision ~= [test_C1_indices test_C2_indices]);
%message1 = (['count errors: ',num2str(count_errors)]);
%disp(message1);

% Total Classification Error (percentage)
%Error = (count_errors/(length(test_C1_indices)+length(test_C2_indices)))*100;
%message1 = (['Percentage of total classification error: ',num2str(Error), '%']);
%disp(message1);