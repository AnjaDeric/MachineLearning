function best_perf = q1_train_and_val(all_train_data,all_train_labels,...
    all_test_data,all_test_labels)   

    % Create K-fold cross valiadation sets
    kfold_split = cvpartition(length(all_train_data),'KFold',10);
    max_perceptrons = 6;
    total_act_funcs = 2;

    % Arrays to store performance for each activ. function + num perceptrons
    accuracy_relu = zeros(kfold_split.NumTestSets,max_perceptrons);
    accuracy_sig = zeros(kfold_split.NumTestSets,max_perceptrons);

    % Test out each perceptron and activation function combination for all 10
    % test sets created for 10-fold cross validation
    for act_func = 1:total_act_funcs 
        for perceptrons = 1:max_perceptrons
            for test_set = 1: kfold_split.NumTestSets

                % Get train and test data and labels for each test set
                train_index = kfold_split.training(test_set);
                test_index = kfold_split.test(test_set);
                train_data = all_train_data(:,find(train_index));
                test_data = all_train_data(:,find(test_index));
                train_labels = all_train_labels(find(train_index));
                test_labels = all_train_labels(find(test_index));

                % Train the NN and get performance measures for each combo
                if act_func == 1
                    accuracy_sig(test_set,perceptrons) = ...
                        mleMLPwAWGN(train_data,test_data,perceptrons,...
                        train_labels,test_labels,act_func);
                elseif act_func == 2
                    accuracy_relu(test_set,perceptrons) = ...
                        mleMLPwAWGN(train_data,test_data,perceptrons,...
                        train_labels,test_labels,act_func);
                end
            end 
        end
    end 

    % Find num of perceptrons and activation function combination that gives 
    % best accuracy
    [~, perceptron_sig] = max(mean(accuracy_sig,1));
    [~, perceptron_relu] = max(mean(accuracy_relu,1));
    best_perceptron = max([perceptron_sig perceptron_relu]);
    best_activation = find(best_perceptron == [perceptron_sig perceptron_relu],1);

    % Train and test NN on full data set with ideal perceptrons and act. func.
    test_runs = 10; final_perf = zeros(1,test_runs);  
    for run = 1:test_runs
        [final_perf(run), final_params(run)] = mleMLPwAWGN(all_train_data,all_train_data,...
            best_perceptron,all_train_labels,all_train_labels,best_activation);
    end
    
    % Train the 10,000 sample training set with the parameters from the
    % best performing model
    best_perf = find(final_perf == max(final_perf));
    best_params = final_params(best_perf);
    result_labels = mlpModel(all_test_data,best_params,best_activation);  
    
    % Calculate and report the performance of the model
    result_label = vec2ind(result_labels);
    incorrect = sum(result_label ~= all_test_labels);
    best_perf = (length(all_test_labels) - incorrect)/length(all_test_labels);
end

function [accuracy, best_params] = mleMLPwAWGN(train_data, test_data,...
    perceptrons, train_label, test_label, act_func)
    % Maximum likelihood training of a 2-layer MLP assuming AWGN
    
    % Determine/specify sizes of parameter matrices/vectors
    nX = 2;	% input dimensions
    nY = 3; % number of classes
    sizeParams = [nX;perceptrons;nY];
    
    % True NN parametes
    X = train_data;
    paramsTrue.A = 0.3*rand(perceptrons,nX);
    paramsTrue.b = 0.3*rand(perceptrons,1);
    paramsTrue.C = 0.3*rand(nY,perceptrons);
    paramsTrue.d = 0.3*rand(nY,1);
    Y = full(ind2vec(train_label));
    vecParamsTrue = [paramsTrue.A(:);paramsTrue.b;paramsTrue.C(:);paramsTrue.d];

    % Initialize model parameters
    params.A = 0.3*rand(perceptrons,nX);
    params.b = 0.3*rand(perceptrons,1);
    params.C = 0.3*rand(nY,perceptrons);
    params.d = mean(Y,2);
    vecParamsInit = [params.A(:);params.b;params.C(:);params.d];
 
    % Optimize model using fminsearch
    vecParams = fminsearch(@(vecParams)(objectiveFunction(X,Y,...
        sizeParams,vecParams,act_func)),vecParamsInit);

    % Visualize model output for training data
    params.A = reshape(vecParams(1:nX*perceptrons),perceptrons,nX);
    params.b = vecParams(nX*perceptrons+1:(nX+1)*perceptrons);
    params.C = reshape(vecParams((nX+1)*perceptrons+1:(nX+1+nY)*perceptrons),nY,perceptrons);
    params.d = vecParams((nX+1+nY)*perceptrons+1:(nX+1+nY)*perceptrons+nY);
    best_params = params;
    H = mlpModel(test_data,params,act_func);
    
    % Calculate accuracy based on number of misclassifications
    result_label = vec2ind(H);
    incorrect = sum(result_label ~= test_label);
    accuracy = (length(test_label) - incorrect)/length(test_label);
end
 
function objFncValue = objectiveFunction(X,Y,sizeParams,vecParams,act_func)
    % Function to be optimized by neaural net
    N = size(X,2); 
    nX = sizeParams(1);
    nPerceptrons = sizeParams(2);
    nY = sizeParams(3);
    params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
    params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
    params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
    params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
    H = mlpModel(X,params,act_func); % neural net model
    objFncValue = sum(-sum(Y.*log(H),1),2)/N;
end 

function H = mlpModel(X,params,act_func)
    % Neutal Network Model
    N = size(X,2);                          % number of samples
    nY = size(params.d);                  % number of outputs
    U = params.A*X + repmat(params.b,1,N);  % u = Ax + b, x \in R^nX, b,u \in R^nPerceptrons, A \in R^{nP-by-nX}
    Z = activationFunction(U,act_func);     % z \in R^nP, using nP instead of nPerceptons
    V = params.C*Z + repmat(params.d,1,N);  % v = Cz + d, d,v \in R^nY, C \in R^{nY-by-nP}
    H = exp(V)./repmat(sum(exp(V),1),nY,1); % softmax nonlinearity for second/last layer 
end

function out = activationFunction(in, act_func)
    % Possible Activation Functions
    if act_func == 1
        out = 1./(1+exp(-in)); % sigmoid
    elseif act_func == 2
        out = in./sqrt(1+in.^2); % ISRU
    end
end 