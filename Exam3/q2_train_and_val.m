function [performance,best_GMM_order] = q2_train_and_val(train_data,...
    train_labels,train_size,test_data,test_labels)
    
    % Separate training data by class labels
    class1_data = train_data(:,find(train_labels==1));
    class2_data = train_data(:,find(train_labels==2));
    class3_data = train_data(:,find(train_labels==3));
    
    % Estimate class priors from labels
    est_alpha = [length(class1_data), length(class2_data), ...
        length(class3_data)]./train_size;

    % Run EM algorith for 100 experiments to determine best model order for
    % each class
    num_experiments = 100;
    num_GMM_picks = zeros(3,6);
    for class_num = 1:3
        for experiment = 1:num_experiments
            % EM algorithm with 10-fold cross validation to determine ideal
            % number of components
            if class_num == 1
                GMM_pick = cross_val(class1_data);
            elseif class_num == 2
                GMM_pick = cross_val(class2_data);
            else
                GMM_pick = cross_val(class3_data);
            end
            % Keep track of how many times each model order is selected
            num_GMM_picks(class_num,GMM_pick) = num_GMM_picks(class_num,GMM_pick)+1;
        end
    end
    
    % Pick model order for each class based on maximum num of selections
    [~,best_GMM_order] = max(num_GMM_picks',[],1);

    %Run 10 trials finding best gmm for each class label
    num_experiments = 10;
	for experiment = 1:num_experiments
        likelihood = zeros(3,num_experiments);
        max_iter = statset('MaxIter',5000);
        for class_num = 1:3
            if class_num == 1
                GMModel = fitgmdist(class1_data',best_GMM_order(1),...
                    'regularizationValue',1e-10,'Options', max_iter);
            elseif class_num == 2
                GMModel = fitgmdist(class2_data',best_GMM_order(2),...
                    'regularizationValue',1e-10,'Options', max_iter);
            else
                GMModel = fitgmdist(class3_data',best_GMM_order(3),...
                    'regularizationValue',1e-10,'Options', max_iter);
            end
            all_GMModels{class_num,experiment} = GMModel;
            likelihood(class_num,experiment) = GMModel.NegativeLogLikelihood;
           
        end
    end

    [~,best_model] = min(likelihood',[],1);
    best_GMModels =[all_GMModels(1,best_model(1)),...
        all_GMModels(2,best_model(2)),all_GMModels(3,best_model(3))]; 

    % Use test data to get probabilty of each class label for each sample
    label1_prob = pdf(best_GMModels{1},test_data')*est_alpha(1);
    label2_prob = pdf(best_GMModels{2},test_data')*est_alpha(2);
    label3_prob = pdf(best_GMModels{3},test_data')*est_alpha(3);
    
    % Pick class label that is most likely out of 3
    label_prob = [label1_prob label2_prob label3_prob]; 
    [~, label_guess] = max(label_prob,[],2);
    
    % Calculate performance based on number of correct guesses
    performance = sum(test_labels' == label_guess)/size(test_labels',1);
end 
 
function best_GMM = cross_val(data)
    % Performs EM algorithm to estimate parameters and evaluete performance
    % on each data set B times, with 1 through M GMM models considered
    
    B = 10; M = 6;          % repetitions per data set; max GMM considered
    perf_array= zeros(B,M); % save space for performance evaluation
    
    kfold_split = cvpartition(length(data),'KFold',10);
    for b = 1:B
        % Pull out test and training data based on 10-fold K cross-val
        train_index = kfold_split.training(b);
        train_data = data(:,find(train_index));
         
        for m = 1:M
           % Run EM algorithm to estimate parameters
           max_iter = statset('MaxIter',5000);
           GMModel = fitgmdist(train_data',m,'regularizationValue',1e-10, ...
               'Options',max_iter);   
           % Record likelihood of model fit for this combination
           perf_array(b,m) = GMModel.BIC;
        end 
    end
       
    % Calculate average performance for each M and find best fit model
    avg_parf = sum(perf_array,1)./B;
    best_GMM = find(avg_parf == min(avg_parf));
end