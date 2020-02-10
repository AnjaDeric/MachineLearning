% Question 2 | Take Home Exam #1
% Anja Deric | February 10, 2020
clear all; close all; clc;

n = 2;      % number of feature dimensions
N = 1000;   % number of iid samples

% Class 0 parameters (2 gaussians)
mu(:,1) = [3.5;0]; mu(:,2) = [5.5;4];
Sigma(:,:,1) = [5 1;1 4]/3; Sigma(:,:,2) = [3 -2;-2 15]/10;
p0 = [0.8 0.2]; % Class 0 mixture coefficients

% Class 1 parameters (2 gaussians)
mu(:,3) = [0.5;1]; mu(:,4) = [2.5;3]; 
Sigma(:,:,3) = [3 -2;-2 15]/13; Sigma(:,:,4) = [15 1;1 3]/13;
p1 = [0.25 0.75]; % Class 1 mixture coefficients

% Class priors for class 0 and 1 respectively
p = [0.4,0.6]; 

% Generating true class labels
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))];

% Draw samples from each class pdf
x = zeros(n,N); % save up space
for i = 1:N
    % Generating class 0 samples
    if label(i) == 0
        % Split samples based on mixture coefficients for class 0
        if (rand(1,1) > p0(1)) == 0
            x(:,i) = mvnrnd(mu(:,1),Sigma(:,:,1),1)';
        else
            x(:,i) = mvnrnd(mu(:,2),Sigma(:,:,2),1)';
        end
    end
    
    % Generating class 1 samples
    if label(i) == 1
        % Split samples based on mixture coefficients for class 1
        if (rand(1,1) > p1(1)) == 0
            x(:,i) = mvnrnd(mu(:,3),Sigma(:,:,3),1)';
        else
            x(:,i) = mvnrnd(mu(:,4),Sigma(:,:,4),1)';
        end
    end
end

% Plot samples with true class labels
figure(1);
plot(x(1,label==0),x(2,label==0),'o',x(1,label==1),x(2,label==1),'+');
legend('Class 0','Class 1'); title('Data and True Class Labels');
xlabel('x_1'); ylabel('x_2');

% Calculate threshold based on loss values
lambda = [0 1;1 0];
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); 

% Calculate discriminant score based on class pdfs
class0pdf = p0(1)*evalGaussian(x,mu(:,1),Sigma(:,:,1)) + p0(2)*evalGaussian(x,mu(:,2),Sigma(:,:,2));
class1pdf = p1(1)*evalGaussian(x,mu(:,3),Sigma(:,:,3)) + p1(2)*evalGaussian(x,mu(:,4),Sigma(:,:,4));
discriminantScore = log(class1pdf)-log(class0pdf);

% Compare score to threshold to make decisions
decision = (discriminantScore >= log(gamma));

% Calculate error probabilities
TN = find(decision==0 & label==0); % true negative
FP = find(decision==1 & label==0); % false positive
FN = find(decision==0 & label==1); % false negative
TP = find(decision==1 & label==1); % true positive

% Calculate and print error values
pFA = length(FP)/Nc(1); % prob. false alarm
pMD = length(FN)/Nc(2); % prob. missed detection
pE = pFA*p(1)+pMD*p(2); % toatal prob. of error
fprintf('Minimum Probability of Error: %.2f%%',pE*100);

% Plot correct and incorrect decisions
% class 0 circle, class 1 +, correct green, incorrect red
figure(2);
plot(x(1,TN),x(2,TN),'og'); hold on;
plot(x(1,FP),x(2,FP),'or'); hold on;
plot(x(1,FN),x(2,FN),'+r'); hold on;
plot(x(1,TP),x(2,TP),'+g'); hold on;

% Grid based on class PDFs
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
class0grid = p0(1)*evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1)) + p0(2)*evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2));
class1grid = p1(1)*evalGaussian([h(:)';v(:)'],mu(:,3),Sigma(:,:,3)) + p1(2)*evalGaussian([h(:)';v(:)'],mu(:,4),Sigma(:,:,4));

% Decision boundary grid
discriminantScoreGridValues = log(class1grid)-log(class0grid) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);

% Plot discriminant grid contours (level 0 = decision boundary)
contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]);
legend('correct decision, class 0','wrong decision, class 0','wrong decision, class 1','correct decision, class 1','equilevel contours of the discriminant function');
title('Classified Data vs True Class Labels');
xlabel('x_1'); ylabel('x_2');