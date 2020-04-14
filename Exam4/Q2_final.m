%% Take Home Exam 4: Question 2
% Anja Deric | April 13, 2020

% Clear all variables and generate new training and testing data
clc; clear;
[train_data,train_labels] =  generateMultiringDataset(2,1000,1);
title('Training Data');
[test_data,test_labels] =  generateMultiringDataset(2,10000,2);
title('Validation Data');

%% Initial Training
% Train a Gaussian kernel SVM with cross-validation to select 
% hyperparameters that minimize probability of error 

% Hyperparameter initialization
CList = 10.^linspace(-3,7,11);
sigmaList = 10.^linspace(-2,3,13);
lossVal = zeros(length(sigmaList),length(CList));

% Loop through all hyperparameters to find best combination
for sigmaCounter = 1:length(sigmaList)
    sigma = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        % Train SVM model for hyperparameter combination
        SVMModel = fitcsvm(train_data',train_labels,'BoxConstraint',C,...
            'KernelFunction','rbf','KernelScale',sigma);
        % Get cross-validated model (10-fold cross validation)
        CVSVMModel = crossval(SVMModel);
        % Store loss for SVM model
        lossVal(sigmaCounter,CCounter) = kfoldLoss(CVSVMModel); 
    end
end

%% Final Training

% Find best sigma and C combination
minLoss = min(lossVal(:));
[sigmaBest_ind, CBest_ind] = find(lossVal == minLoss);

% Train final SVM and predict labels on test data
best_SVMModel = fitcsvm(train_data',train_labels,'BoxConstraint',...
    CList(CBest_ind),'KernelFunction','rbf','KernelScale',...
    sigmaList(sigmaBest_ind));
prediction = best_SVMModel.predict(test_data');

% Calculate accuracy of model
incorrect = sum(prediction' ~= test_labels);
accuracy = ((length(test_data) - incorrect)/length(test_data))*100

%% Plot Classified Labels

% Find and plot all correctly and incorrectly classified test samples
indINCORRECT = find(prediction' ~= test_labels); % incorrectly classified
indCORRECT = find(prediction' == test_labels);   % correctly classified
plot(test_data(1,indCORRECT),test_data(2,indCORRECT),'g.'); hold on;
plot(test_data(1,indINCORRECT),test_data(2,indINCORRECT),'r.'); axis equal;
title('Training Data (RED: Incorrectly Classified)');
xlabel('X_1'); ylabel('X_2');

%% Functions

function [data,labels] = generateMultiringDataset(numberOfClasses,numberOfSamples,fig)
    % Generates N samples from C ring-shaped class-conditional pdfs with equal priors

    % Randomly determine class labels for each sample
    thr = linspace(0,1,numberOfClasses+1); % split [0,1] into C equal length intervals
    u = rand(1,numberOfSamples); % generate N samples uniformly random in [0,1]
    labels = zeros(1,numberOfSamples);
    for l = 1:numberOfClasses
        ind_l = find(thr(l)<u & u<=thr(l+1));
        labels(ind_l) = repmat(l,1,length(ind_l));
    end

    a = [1:numberOfClasses].^3; b = repmat(2,1,numberOfClasses); % parameters of the Gamma pdf needed later
    % Generate data from appropriate rings
    % radius is drawn from Gamma(a,b), angle is uniform in [0,2pi]
    angle = 2*pi*rand(1,numberOfSamples);
    radius = zeros(1,numberOfSamples); % reserve space
    for l = 1:numberOfClasses
        ind_l = find(labels==l);
        radius(ind_l) = gamrnd(a(l),b(l),1,length(ind_l));
    end

    data = [radius.*cos(angle);radius.*sin(angle)];

    if 1
        colors = rand(numberOfClasses,3);
        figure(fig); clf;
        for l = 1:numberOfClasses
            ind_l = find(labels==l);
            plot(data(1,ind_l),data(2,ind_l),'.','MarkerFaceColor',colors(l,:)); 
            axis equal; hold on;
            xlabel('X_1'); ylabel('X_2');
        end
    end
end