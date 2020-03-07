%% Ecam 2: Question 2
% Anja Deric | February 24, 2020
clear; close all; clc;

%% Initialize all constants and parameters

N = 10;                         % Number of samples
SigmaV = 0.2;                   % Variance of 0-mean Gaussian noise
gamma_array = 10.^[-10:0.1:10]; % Array of gamma values
realizations = 100;             % Total experiments for each gamma

% True parameter array (values picked so y has 3 real roots) 
SigmaV=0.005;
a=1; b=-0.15; c=-0.015; d=0.001;
w_true = [a; b; c; d];

% Calculate noise and input values
v = SigmaV^0.5*randn(1,N);
x = unifrnd(-1,1,1,N);

% Map to a cubic function and calculate output y
zC = [x.^3; x.^2; x; ones(1,N)];
y = zC'*w_true + v';

%% Estimate MAP Parameters and Plot 1 Realization Only

% MAP estimation
for i = 1:length(gamma_array)
    gamma = gamma_array(i);
    w_MAP(:,i) = inv((zC*zC')+SigmaV^2/gamma^2*eye(size(zC,1)))* ...
        sum(repmat(y',size(zC,1),1).*zC,2);
end

% Calculate x and y coordinates to plot MAP line of best fit
x_fit = linspace(-1,1);
best_theta = w_MAP(:,end);
y_fit = best_theta(1).*x_fit.^3+best_theta(2).*x_fit.^2+best_theta(3).*x_fit+best_theta(4);

% Plot original points with the MAP line of best fit
figure; scatter(x,y'); hold on; box on;
plot(x_fit,y_fit); legend('Data Points','MAP estimate');
title('1 Realization of Polynomial Function with MAP Estimate');
xlabel('x');ylabel('y');

%% Plot MAP Parameter Variation With Gamma

% Plot tue parameters
figure; hold on; box on; ax=gca; ax.XScale = 'log';
axis([gamma_array(1) gamma_array(end) min([w_true;w_MAP(:)])-0.5 ...
    max([w_true;w_MAP(:)])+2]);

% Plot paramater estimates for all gamma values
plot(gamma_array,repmat(w_true,1,length(gamma_array)),'--','LineWidth',2);
set(gca,'ColorOrderIndex',1); 
plot(gamma_array,w_MAP,'-','LineWidth',2);

% Add labels and legend
xlabel('Gamma, \gamma'); ylabel('Parameters, \theta'); title('MAP Parameter Estimation: Quadratic Model')
lgnd=legend('a','b','c','d','a estimate','b estimate','c estimate','d estimate');
lgnd.Location = 'north'; lgnd.Orientation = 'horizontal'; lgnd.NumColumns = 4; box(lgnd,'off');

%% Estimate MAP across all gammas for 100 realizations

clearvars -except w_true mu Sigma SigmaV gamma_array realizations N;
for n = 1:realizations
    % Generate noise and input values
    v = SigmaV^0.5*randn(1,N);
    x = unifrnd(-1,1,1,N);

    % Map to a cubic function and calculate true and noisy output
    zC = [x.^3; x.^2; x; ones(1,N)];
    y_truth{1,n} = zC'*w_true;
    y = y_truth{1,n} + v';

    % Estimate parameters and error for all gamma values
    for i = 1:length(gamma_array)
        gamma = gamma_array(i);
        % Calculate MAP parameter estimate
        %w_MAP{1,n}(:,i) = inv((zC*zC')+SigmaV^2/gamma^2*eye(size(zC,1)))*...
        %    sum(repmat(y',size(zC,1),1).*zC,2);
        w_MAP{1,n}(:,i) = inv((zC*zC')+SigmaV^2/gamma^2*eye(size(zC,1)))*...
            (zC*y);
        % Calculate squared-error-value (2nd-norm)
        L2_norm(n,i) = norm(w_true - w_MAP{1,n}(:,i),2).^2;   
    end
    
    avMsqError(n,1:length(gamma_array)) = length(w_true)\sum((w_MAP{1,n} - ...
        repmat(w_true,1,length(gamma_array))).^2);
end

%%
percentileArray = [0,25,50,75,100];
figure;
ax = gca; hold on; box on;
prctlMsqError = prctile(avMsqError,percentileArray,1);
p=plot(ax,gamma_array,prctlMsqError,'LineWidth',2);
xlabel('gamma'); ylabel('average mean squared error of parameters'); ax.XScale = 'log';
lgnd = legend(ax,p,[num2str(percentileArray'),...
    repmat(' percentile',length(percentileArray),1)]); lgnd.Location = 'southwest';


%% Plot MAP Ensemble Squared-Error Values

% Calculate min, 25%, median, 75%, and max sq-error for each gamma
prctl_array = [0,25,50,75,100];
L2_norm_prctl = prctile(L2_norm,prctl_array,1);

% Plot change over all gamma values
figure; semilogx(gamma_array,L2_norm_prctl);
title('Change in Squared-Error With Changing Gamma');
xlabel('Gamma, \gamma'); ylabel('Squared-Error of Parameters, L_2');
lgnd = legend('Minimum', '25 percentile','Median','75 percentile','Maximum'); 
lgnd.Location = 'northwest';

% Plot change over all gamma values (both axes log scale)
figure; loglog(gamma_array,L2_norm_prctl);
title('Change in Squared-Error With Changing Gamma');
xlabel('Gamma, \gamma'); ylabel('Squared-Error of Parameters, L_2');
lgnd = legend('Minimum', '25 percentile','Median','75 percentile','Maximum'); 
lgnd.Location = 'northwest';
