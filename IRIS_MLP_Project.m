%% Plotting the pie chart for each class to check distribution of classes
% Load the IRIS dataset
load fisheriris

% Get the flower class labels
classLabels = unique(species);

% Count the occurrences of each flower class
classCounts = histcounts(categorical(species));

% Calculate the total count of all classes
totalCount = sum(classCounts);

% Calculate the percentage of each class
classPercentages = (classCounts / totalCount) * 100;

% Create a pie chart of the distribution
pie(classCounts, classLabels)

% Add count labels within each pie slice
textObjs = findobj(gca, 'Type', 'text');
for i = 1:numel(classLabels)
    str = sprintf('%s\n%.1f%%', classLabels{i}, classPercentages(i));
    textObjs(i).String = str;
end

title('Distribution of Flower Classes in IRIS Dataset')

%% Plotting the Box plot for each perticular class to visualize the feature values
% Load the IRIS dataset as it already present in MATLAB database
load fisheriris
data = meas; % Input features
speciesLabels = species; % Class labels

% Create a box plot for each feature
figure;
for feature = 1:size(data, 2)
    subplot(2, 2, feature);
    boxplot(data(:, feature), speciesLabels);
    xlabel('Species');
    ylabel('Feature Value');
    title(['Box Plot - Feature ' num2str(feature)]);
end
%%
% MLP Model with Backpropagation and L2 Regularization on IRIS Dataset
% Load the IRIS dataset
load fisheriris 
X = meas; % Input features
X = normalize(X);
Y = zeros(size(species, 1), 3); % Target outputs
Y(strcmp(species, 'setosa'), 1) = 1;
Y(strcmp(species, 'versicolor'), 2) = 1;
Y(strcmp(species, 'virginica'), 3) = 1;

% Split the dataset into training and testing sets
trainX = X(1:120, :);
trainY = Y(1:120, :);
testX = X(121:end, :);
testY = Y(121:end, :);

% Set the hyperparameters
learningRate = 0.1; % Learning rate
epochs = 50; % Number of training epochs
lambda = 0.1; % Regularization parameter

% Initialize the weights and biases with 4 inputs and 3 outputs
hiddenWeights = randn(4, 4); % Weights of the hidden layer 
hiddenBiases = randn(1, 4); % Biases of the hidden layer
outputWeights = randn(4, 3); % Weights of the output layer
outputBias = randn(1, 3); % Biases of the output layer

% Initialize error array for plotting
errorArray = zeros(epochs, 1);

% Training loop
for epoch = 1:epochs
    % Forward propagation
    hiddenLayerOutput = sigmoid(trainX * hiddenWeights + hiddenBiases)
    predictedOutput = softmax(hiddenLayerOutput * outputWeights + outputBias)
    
    % Backpropagation
    outputError = trainY - predictedOutput;
    outputDelta = outputError;
    hiddenError = outputDelta * outputWeights';
    hiddenDelta = hiddenError .* sigmoidDerivative(hiddenLayerOutput);
    
    % Update weights and biases with regularization
    outputWeights = outputWeights + learningRate * (hiddenLayerOutput' * outputDelta - lambda * outputWeights);
    outputBias = outputBias + learningRate * sum(outputDelta);
    hiddenWeights = hiddenWeights + learningRate * (trainX' * hiddenDelta - lambda * hiddenWeights);
    hiddenBiases = hiddenBiases + learningRate * sum(hiddenDelta);
    
    % Calculate total error for plotting
    errorArray(epoch) = sum(sum(abs(outputError)));
end

% Testing
hiddenLayerOutput = sigmoid(testX * hiddenWeights + hiddenBiases);
predictedOutput = softmax(hiddenLayerOutput * outputWeights + outputBias);
[~, predictedLabels] = max(predictedOutput, [], 2);
[~, trueLabels] = max(testY, [], 2);
accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
disp(['Testing Accuracy: ' num2str(accuracy * 100) '%']);

% Plot the total error
figure;
plot(1:epochs, errorArray,'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10);
grid on
xlabel('Epoch');
ylabel('Total Error');
title('Total Error vs. Epoch');

% Sigmoid activation function
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

% Softmax activation function
function y = softmax(x)
    ex = exp(x);
    y = ex ./ sum(ex, 2);
end

% Derivative of the sigmoid function
function y = sigmoidDerivative(x)
    y = sigmoid(x) .* (1 - sigmoid(x));
end