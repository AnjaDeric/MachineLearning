% Question 3 | Take Home Exam #3
% Anja Deric | March 30, 2020
clear all; close all; clc;

%% Generating Data for Questions 1 & 2
[train100_data,train100_labels] =  generateMultiringDataset(3,100);
[train500_data,train500_labels] =  generateMultiringDataset(3,500);
[train1k_data,train1k_labels] =  generateMultiringDataset(3,1000);
[test_data,test_labels] = generateMultiringDataset(3,10000);

%% Question 1 - Neural Network Training
% Train and validate neaural netrwork for each data set
% For each set, report performance for 10 trials and the average

% Train and validate 100 sample data set
NNperf100 = q1_train_and_val(train100_data,train100_labels,...
    test_data,test_labels)

% Train and validate 500 sample data set
NNperf500 = q1_train_and_val(train500_data,train500_labels,...
    test_data,test_labels)

% Train and validate 1000 sample data set
NNperf1k = q1_train_and_val(train1k_data,train1k_labels,...
    test_data,test_labels)

%% Question 2 - GMM + MAP Classifier
% Train and valiade each data set using the GMM + EM algorithm
% For each set, report on performance and model order selection

% Train and validate 100 sample data set
[GMMperf100,GMMorder100] = q2_train_and_val(train100_data,...
    train100_labels,100,test_data,test_labels);
GMM_perf_summary(100,GMMperf100,GMMorder100);

% Train and validate 500 sample data set
[GMMperf500,GMMorder500] = q2_train_and_val(train500_data,...
    train500_labels,500,test_data,test_labels);
GMM_perf_summary(500,GMMperf500,GMMorder500);

% Train and validate 1000 sample data set
[GMMperf1k,GMMorder1k] = q2_train_and_val(train1k_data,...
    train1k_labels,1000,test_data,test_labels);
GMM_perf_summary(1000,GMMperf1k,GMMorder1k);

%% Functions
function GMM_perf_summary(samples,perf,best)
    fprintf('<strong>%i Samples EM/GMM Summary</strong>\n',samples);
    fprintf('Number of Gaussians, Class 1: %i\n',best(1));
    fprintf('Number of Gaussians, Class 2: %i\n',best(2));
    fprintf('Number of Gaussians, Class 3: %i\n',best(3));
    fprintf('Overall MAP Performance: %.2f%%\n\n',perf*100);
end

function [data,labels] = generateMultiringDataset(numberOfClasses,numberOfSamples)

    C = numberOfClasses;
    N = numberOfSamples;
    % Generates N samples from C ring-shaped 
    % class-conditional pdfs with equal priors

    % Randomly determine class labels for each sample
    thr = linspace(0,1,C+1); % split [0,1] into C equal length intervals
    u = rand(1,N); % generate N samples uniformly random in [0,1]
    labels = zeros(1,N);
    for l = 1:C
        ind_l = find(thr(l)<u & u<=thr(l+1));
        labels(ind_l) = repmat(l,1,length(ind_l));
    end

    a = [1:C].^3; b = repmat(2,1,C); % parameters of the Gamma pdf needed later
    % Generate data from appropriate rings
    % radius is drawn from Gamma(a,b), angle is uniform in [0,2pi]
    angle = 2*pi*rand(1,N);
    radius = zeros(1,N); % reserve space
    for l = 1:C
        ind_l = find(labels==l);
        radius(ind_l) = gamrnd(a(l),b(l),1,length(ind_l));
    end

    data = [radius.*cos(angle);radius.*sin(angle)];
end
