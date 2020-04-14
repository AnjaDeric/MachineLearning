%% Take Home Exam 4: Question 3
% Anja Deric | April 13, 2020

% Clear all variables and load images in
clear all; close all;
filenames{1,1} = '3096_color.jpg';
filenames{1,2} = '42049_color.jpg';
 
for imageCounter = 1:2 %size(filenames,2)
    % Load and display original image
    imdata = imread(filenames{1,imageCounter}); 
    figure(1); subplot(size(filenames,2),3,(imageCounter-1)*3+1); 
    imshow(imdata); title('Original Image');
   
    % Create and normalize feature vector
    [R,C,D] = size(imdata); N = R*C; imdata = double(imdata);
    rowIndices = [1:R]'*ones(1,C); colIndices = ones(R,1)*[1:C];
    % Initialize with row and column indices
    features = [rowIndices(:)';colIndices(:)']; 
    % Add RGB values to feature vector
    for d = 1:D
        imdatad = imdata(:,:,d);
        features = [features;imdatad(:)'];
    end
    % Map all features to [0,1] range 
    minf = min(features,[],2); maxf = max(features,[],2);
    ranges = maxf-minf;
    normalized = diag(ranges.^(-1))*(features-repmat(minf,1,N)); 
    
    % Fit 2-component GMM to image
    params = statset('MaxIter',1000);
    GMModel_2 = fitgmdist(normalized',2,'regularizationValue',1e-10, ...
                'Options',params); 
    
    % Reshape and plot 2-component image
    labels = cluster(GMModel_2,normalized')==2;
    labelImage = reshape(labels,R,C);     
    figure(1); subplot(size(filenames,2),3,(imageCounter-1)*3+2); 
    imshow(uint8(labelImage*255)); title('2 Component Image');
 
    % 10-fold cross validation for 1-6 GMM component models
    kfold_split = cvpartition(length(normalized),'KFold',10); 
    M = 6; K = 10; log_likelihood = zeros(M,K);
    for m = 1:M         % component model
    	for k = 1:K     % cross-val
           
            % Get train and test data for each set 
            train_index = kfold_split.training(k);
            test_index = kfold_split.test(k);
            train_data = normalized(:,find(train_index));
            test_data = normalized(:,find(test_index));

            % Fit GMModel to training data
            GMModel = fitgmdist(train_data',m,'regularizationValue',...
                1e-10,'Options',params);
            all_GMModels{m,k} = GMModel;

            % Calculate and store validation log-likelihood
            GMM_pdf = pdf(GMModel,test_data');
            log_likelihood(m,k) = sum(log(GMM_pdf));
            
        end
    end
   
    % Average all likelihoods and find best model order
    averagemleTest = mean(log_likelihood',1)
    [~, best_model] = max(averagemleTest);
    
    % Fit ideal GMModel to image and create labels
    best_GMModel = fitgmdist(normalized',best_model,'regularizationValue',...
        1e-10,'Options',params);
    best_labels = cluster(best_GMModel,normalized')-1;
    
    % Reshape image into original shape and plot
    best_labelImage = reshape(best_labels,R,C); 
    figure(1); subplot(size(filenames,2),3,(imageCounter-1)*3+3);
    imshow(uint8(best_labelImage*255/(best_model-1)));
    title('Best Component Fit');
 
end
