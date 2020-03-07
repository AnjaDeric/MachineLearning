%% Exam 2: Question 1
% Anja Deric | February 25, 2020
clear all; close all; clc;

%% Set-Up: given parameters and validation data
% Given parameters
n = 2;                      % number of feature dimensions
N_train = [10;100;1000];    % number of training samples
N_val = 10000;              % number of validation samples
p = [0.9,0.1];              % class priors
mu = [-2 2;0 0]; 
Sigma(:,:,1) = [1 -0.9;-0.9 2]; Sigma(:,:,2) = [2 0.9;0.9 1];  

% Generate true class labels and draw samples from each class pdf
label_val = (rand(1,N_val) >= p(1));
Nc_val = [length(find(label_val==0)),length(find(label_val==1))];
x_val = generate_data(n,N_val,label_val,mu,Sigma,Nc_val);

%% Part 1: Minimum P-Error Classifier
% Calculate dicriminant scores and tau based on classification rule
discriminantScore = log(evalGaussian(x_val,mu(:,2),Sigma(:,:,2)))- ...
    log(evalGaussian(x_val,mu(:,1),Sigma(:,:,1)));
tau = log(sort(discriminantScore(discriminantScore >= 0)));

% Find midpoints of tau to use as threshold values
mid_tau = [tau(1)-1 tau(1:end-1) + diff(tau)./2 tau(length(tau))+1];

% Make decision for every threshold and calculate error values
for i = 1:length(mid_tau)
    decision = (discriminantScore >= mid_tau(i));
    pFA(i) = sum(decision==1 & label_val==0)/Nc_val(1); % False alarm prob.
    pCD(i) = sum(decision==1 & label_val==1)/Nc_val(2); % Correct detection prob.
    pE(i) = pFA(i)*p(1)+(1-pCD(i))*p(2);                % Total error prob.
end

% Find minimum error and corresponding threshold + decisions
[min_error,min_index] = min(pE);
min_decision = (discriminantScore >= mid_tau(min_index));
min_FA = pFA(min_index); min_CD = pCD(min_index);

% Plot ROC curve with minimum error point labeled
figure(1); plot(pFA,pCD,'-',min_FA,min_CD,'o');
title('Minimum Expected Risk ROC Curve'); 
legend('ROC Curve', 'Calculated Min Error');
xlabel('P_{False Alarm}'); ylabel('P_{Correct Detection}');

% Create grid to cover all data points
h_grid = linspace(floor(min(x_val(1,:)))-2,ceil(max(x_val(1,:)))+2);
v_grid = linspace(floor(min(x_val(2,:)))-2,ceil(max(x_val(2,:)))+2);
[h,v] = meshgrid(h_grid,v_grid);

% Calculate discriminant score for each grid point (0 = decision bound.)
d_score_grid = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))- ... 
    log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1)))-mid_tau(min_index);
d_scores = reshape(d_score_grid,length(h_grid),length(v_grid));

% Plot classified data points and decision boundary
figure(2);
plot_classified_data(min_decision, label_val, Nc_val, p, [1 1 1], ...
    x_val', [h_grid;v_grid;d_scores], 'Q');

%% Part 2 & 3: Linear and Quadratic Regression
for i = 1:length(N_train)
    % Generate true labels and training data for each sample size
    label = (rand(1,N_train(i)) >= p(1));
    Nc = [length(find(label==0)),length(find(label==1))];
    x = generate_data(n,N_train(i),label,mu,Sigma,Nc);
    
    % Initialize training parameters (map x to linear and quadratic func.)
    x_L = [ones(N_train(i), 1) x'];         % linear parameters
    initial_theta_L = zeros(n+1, 1);
    x_Q = [ones(N_train(i), 1) x(1,:)' x(2,:)' (x(1,:).^2)' ...
        (x(1,:).*x(2,:))' (x(2,:).^2)'];    % quadratic parameters
    initial_theta_Q = zeros(6, 1);
    label = double(label)';
    
    % Compute gradient descent to get theta values for linear and quad.
    [theta_L, cost_L] = fminsearch(@(t)(cost_func(t, x_L, label, ...
        N_train(i))), initial_theta_L);
    [theta_Q, cost_Q] = fminsearch(@(t)(cost_func(t, x_Q, label, ...
        N_train(i))), initial_theta_Q);
    
    % Linear: choose points to draw straight boundary line
    plot_x1 = [min(x_L(:,2))-2,  max(x_L(:,2))+2];                      
    plot_x2 = (-1./theta_L(3)).*(theta_L(2).*plot_x1 + theta_L(1)); 
    
    % Linear: plot training data and trained classifier
    figure(3); plot_training_data(label, i, x_L, [plot_x1;plot_x2], 'L');
    title(['Training Based on ',num2str(N_train(i)),' Samples']);
    
    % Linear: use validation data (10k points) and make decisions
    test_set_L = [ones(N_val, 1) x_val'];
    decision_L = test_set_L*theta_L >= 0;
    
    % Linear: plot all decisions and boundary line
    figure(4);
    error_L(i) = plot_classified_data(decision_L, label_val', Nc_val, p, ...
        [1,3,i], test_set_L(:,2:3), [plot_x1;plot_x2],'L');
    title(['Classification Based on ',num2str(N_train(i)),' Samples']);
    
    % Quadratic: create grid to cover all data points and calculate scores   
    figure(5); subplot(2,3,i);
    h_grid = linspace(min(x_Q(:,2))-6, max(x_Q(:,2))+6);
    v_grid = linspace(min(x_Q(:,3))-6, max(x_Q(:,3))+6);
    score = get_boundary(h_grid,v_grid,theta_Q); % score of 0 = decision bound.
    
    % Quadratic: plot training data and trained classifier
    plot_training_data(label, i, x_Q, [h_grid;v_grid;score], 'Q')
    title(['Training Based on ',num2str(N_train(i)),' Samples']);
    
    % Quadratic: use validation data (10k points) and make decisions
    test_set_Q = [ones(N_val, 1) x_val(1,:)' x_val(2,:)' (x_val(1,:).^2)' ...
        (x_val(1,:).*x_val(2,:))' (x_val(2,:).^2)'];
    decision_Q = test_set_Q*theta_Q >= 0;
    
    % Quadratic: plot all decisions and boundary countour
    figure(6);
    error_quad(i) = plot_classified_data(decision_Q, label_val', Nc_val, ...
        p, [1,3,i], test_set_Q(:,2:3),[h_grid;v_grid;score],'Q');
    title(['Classification Based on ',num2str(N_train(i)),' Samples']);
end

%% Print all calculated error values 
fprintf('<strong>Minimum P(error) Achiavable:</strong> %.2f%%\n\n',min_error*100);
fprintf('<strong>Logistic Regression Total Error Values</strong>\n');
fprintf('Training Set Size\tLinear Approximation Error (%%)\tQuadratic Approximation Error (%%)\n');
fprintf('\t  %i\t\t\t\t\t  %.2f%%\t\t\t\t\t\t\t\t%.2f%%\n',[N_train';error_L;error_quad]);

%% Functions
function x = generate_data(n, N, label, mu, Sigma, Nc)
    % Generate N Gaussian samples based on distribution of class priors
    x = zeros(n,N);
    for L = 0:1
        x(:,label==L) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc(L+1))';
    end
end

function plot_training_data(label, fig, x, bound, type)
    % Plots original class labels and decision boundary
    
    subplot(1,3,fig); hold on;
    plot(x(label==0,2),x(label==0,3),'o',x(label==1,2),x(label==1,3),'+');
    
    if type == 'L'
        % Plot straight line if boundary is linear
        plot(bound(1,:), bound(2,:));
    elseif type == 'Q'
        % Plot decision countour if non-linear (discriminant scores are 0)
        contour(bound(1,:), bound(2,:), bound(3:end,:), [0, 0]);
    end
    
    % Restrict axis and add all labels
    axis([min(x(:,2))-2, max(x(:,2))+2, min(x(:,3))-2, max(x(:,3))+2]);
    legend('Class 0','Class 1','Classifier');
    xlabel('x_1'); ylabel('x_2'); hold on;
end

function error = plot_classified_data(decision, label, Nc, p, fig, x, bound, type)
    % Plots incorrect and correct decisions (and boundary) based on original class labels
    
    % Find all correct and incorrect decisions
    TN = find(decision==0 & label==0); 	% true negative
    FP = find(decision==1 & label==0); pFA = length(FP)/Nc(1); % false positive
    FN = find(decision==0 & label==1); pMD = length(FN)/Nc(2); % false negative
    TP = find(decision==1 & label==1);  % true positive
    error = (pFA*p(1) + pMD*p(2))*100;  % calculate total error

    % Plot all decisions (green = correct, red = incorrect)
    subplot(fig(1),fig(2),fig(3));
    plot(x(TN,1),x(TN,2),'og'); hold on;
    plot(x(FP,1),x(FP,2),'or'); hold on;
    plot(x(FN,1),x(FN,2),'+r'); hold on;
    plot(x(TP,1),x(TP,2),'+g'); hold on;
    
    % Plot boundary based on whether its linear(L) or non-linear(Q)
    if type == 'L'
        % Plot straight line from 2 points
        plot(bound(1,:), bound(2,:));
    elseif type == 'Q'
        % Plot decision contour (when discriminant scores are 0)
        contour(bound(1,:), bound(2,:), bound(3:end,:), [0, 0]); % p
    end
    
    % Restrict axis and add all labels
    axis([min(x(:,1))-2, max(x(:,1))+2, min(x(:,2))-2, max(x(:,2))+2])
    legend('Class 0 Correct Decisions','Class 0 Wrong Decisions', ...
        'Class 1 Wrong Decisions','Class 1 Correct Decisions','Classifier');
    xlabel('x_1'); ylabel('x_2');
end

function cost = cost_func(theta, x, label,N)
    % Cost function to be minimized to get best fitting parameters
    h = 1 ./ (1 + exp(-x*theta));	% Sigmoid function
    cost = (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))));
end

function score = get_boundary(hGrid, vGrid, theta)
    % Generates grid of scores that spans the full range of data (where
    % a score of 0 indicates decision boundary level)
    z = zeros(length(hGrid), length(vGrid));
    for i = 1:length(hGrid)
        for j = 1:length(vGrid)
            % Map to a quadratic function
            x_bound = [1 hGrid(i) vGrid(j) hGrid(i)^2 hGrid(i)*vGrid(j) vGrid(j)^2]; 
            % Calculate score
            z(i,j) = x_bound*theta;
        end
    end
    score = z';
end

function g = evalGaussian(x,mu,Sigma)
    % Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
    [n,N] = size(x);
    C = ((2*pi)^n * det(Sigma))^(-1/2);     % coefficient
    E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1); % exponent
    g = C*exp(E);   % final gaussian evaluation
end