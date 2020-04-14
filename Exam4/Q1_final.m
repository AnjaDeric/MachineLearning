%% Take Home Exam 4: Question 1
% Anja Deric | April 13, 2020

% Clear all variables and generate data
clc; clear;
all_train_data = generateData(1000, 'Training Data');
all_test_data = generateData(10000, 'Validation Data');

%% Neural Network Training

% Prepare 10-fold cross-calidations sets and perceptron values
kfold_split = cvpartition(length(all_train_data),'KFold',10);
max_perceptrons = 10; MSE = zeros(kfold_split.NumTestSets,max_perceptrons);

for perceptrons = 1:max_perceptrons
    for test_set = 1:kfold_split.NumTestSets
        [perceptrons test_set]
        % Get train and test data and labels for each test set
        train_index = kfold_split.training(test_set);
        test_index = kfold_split.test(test_set);
        train_data = all_train_data(:,find(train_index));
        test_data = all_train_data(:,find(test_index));
        
        % Train NN and get MSE value
        MSE(test_set,perceptrons) = mleMLPwAWGN(train_data,test_data,...
            perceptrons,0);
    end
end

% Pick number of perceptrons with lowest average MSE
average_MSEs = mean(MSE,1);
[~, best_perceptron] = min(average_MSEs);

% Train and test final network
final_MSE = mleMLPwAWGN(all_train_data,all_test_data,perceptrons,1);

%% Functions

function MSE = mleMLPwAWGN(train_data, test_data, perceptrons,final)
    % Maximum likelihood training of a 2-layer MLP assuming AWGN
    
    % Determine/specify sizes of parameter matrices/vectors
    nX = 1;	% input dimensions
    nY = 1; % number of classes
    sizeParams = [nX;perceptrons;nY];
    
    % True input and output data for training
    X = train_data(1,:); Y = train_data(2,:);

    % Initialize model parameters
    params.A = 0.3*rand(perceptrons,nX);
    params.b = 0.3*rand(perceptrons,1);
    params.C = 0.3*rand(nY,perceptrons);
    params.d = mean(Y,2);
    vecParamsInit = [params.A(:);params.b;params.C(:);params.d];
 
    % Optimize model using fminsearch
    vecParams = fminsearch(@(vecParams)(objectiveFunction(X,Y,...
        sizeParams,vecParams)),vecParamsInit);

    % Extract best parameters for the model
    params.A = reshape(vecParams(1:nX*perceptrons),perceptrons,nX);
    params.b = vecParams(nX*perceptrons+1:(nX+1)*perceptrons);
    params.C = reshape(vecParams((nX+1)*perceptrons+1:(nX+1+nY)*perceptrons),nY,perceptrons);
    params.d = vecParams((nX+1+nY)*perceptrons+1:(nX+1+nY)*perceptrons+nY);
    
    % Apply trained MLP to test data
    H = mlpModel(test_data(1,:),params);
    
    % Calculate mean-squared-error 
    MSE = (1/length(H))*sum((H-test_data(2,:)).^2);
    
    % If evaluating final model, plot results
    if final == 1
        % Plot true and estimated data pairs 
        figure;
        plot(test_data(1,:),test_data(2,:),'o',test_data(1,:),H,'*');
        xlabel('X_1'); ylabel('X_2 (Real and Estimated)');
        legend('Real samples','Estimated X_2');
        title('Real and Estimated Sample Pairs for Validation Data');
    end
end

function objFncValue = objectiveFunction(X,Y,sizeParams,vecParams)
    % Function to be optimized by neaural net
    N = size(X,2); 
    nX = sizeParams(1);
    nPerceptrons = sizeParams(2);
    nY = sizeParams(3);
    params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
    params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
    params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
    params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
    H = mlpModel(X,params); % neural net model
    objFncValue = sum(sum((Y-H).*(Y-H),1),2)/N;
    %objFncValue = sum(-sum(Y.*log(H),1),2)/N;
end 

function H = mlpModel(X,params)
    % Neutal Network Model
    N = size(X,2);                        % number of samples
    nY = size(params.d);                  % number of outputs
    U = params.A*X + repmat(params.b,1,N);  % u = Ax + b, x \in R^nX, b,u \in R^nPerceptrons, A \in R^{nP-by-nX}
    Z = activationFunction(U);     % z \in R^nP, using nP instead of nPerceptons
    V = params.C*Z + repmat(params.d,1,N);  % v = Cz + d, d,v \in R^nY, C \in R^{nY-by-nP}
    H = V; % linear output layer
end

function out = activationFunction(in)
    % Softplus activation function
    out = log(1+exp(in));
end 

function x = generateData(N,plot_title)
    % Generate and plot dataset with N samples

    % Data mean, variance, priors
    m(:,1) = [-9;-4]; Sigma(:,:,1) = 4*[1,0.8;0.8,1]; 
    m(:,2) = [0;0]; Sigma(:,:,2) = 3*[3,0;0,0.3]; 
    m(:,3) = [8;-3]; Sigma(:,:,3) = 5*[1,-0.9;-0.9,1]; 
    componentPriors = [0.3,0.5,0.2]; thr = [0,cumsum(componentPriors)];
    
    % Generate Data
    u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
    for l = 1:3
        indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
        L(1,indices) = l*ones(1,length(indices));
        x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
   end
    
    % Plot data
    figure; plot(x(1,:),x(2,:),'.');
    xlabel('X_1'); ylabel('X_2');
    title(plot_title);
end