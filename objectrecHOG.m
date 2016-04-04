% %  Load the training and testing images
trainDir = dir('H:\MITpeddatabaseTrain\*.png'); 
testDir = dir('H:\MITpeddatabaseTest\*.png'); 
nTrainImages = length(trainDir);
nTestImages = length(testDir);

disp(nTrainImages);
disp(nTestImages);

dimImg = imread(strcat('H:\MITpeddatabaseTrain\',trainDir(1).name));
dimImg = imresize(dimImg, [80 52]);

[hog_4x4, vis4x4] = extractHOGFeatures(dimImg,'CellSize',[8 8]);
cellSize = [8 8];
hogFeatureSize = length(hog_4x4);
disp('Size of Feature Vector');
disp(size(hog_4x4));

%% Extract HOG Features from all training images & populate matrix
trainingFeatures = [];
trainingLabels = [];

features = zeros(nTrainImages, hogFeatureSize);
labels = zeros(nTrainImages, 1);

for j=1:nTrainImages
    currentImage = ...
        strcat('H:\MITpeddatabaseTrain\',trainDir(j).name);
    name = trainDir(j).name;
    img = imread(currentImage);
    
    % Resize images to a constant height
    img = imresize(img, [80 52]);
    
    % Extract HOG Features and HOG Visulization
    [feat, visu] = extractHOGFeatures(img, 'CellSize', [8 8]);
   
    features(j, :) = feat;
   
    s = trainDir(j).name;
    
    % Assign labels based on image name
    if strfind(s, 'per')
        labels(j) = 1;
    end
    if strfind(s, 'crop')
        labels(j) = 1;
    end
end

trainingFeatures = [trainingFeatures; features];
trainingLabels = [trainingLabels; labels];
disp('Training Set Completed');


%% Extract HOG Features from all testing images & populate matrix
testingFeatures = [];
testingLabels = [];

tfeatures = zeros(nTestImages, hogFeatureSize, 'single');
tlabels = zeros(nTestImages, 1); 
for k=1:nTestImages
    currentImage = ...
        strcat('H:\MITpeddatabaseTest\',testDir(k).name);
    name = testDir(k).name;
    img = imread(currentImage);
    
    % Resize images to a constant height
    img = imresize(img, [80 52]);
   
    % Extract HOG Features and HOG Visulization
    [tfeat, visu] = extractHOGFeatures(img, 'CellSize', [8 8]);
    currImageLength = length(tfeat);
   
    tfeatures(k, :) = tfeat;
   
    p = testDir(k).name;
 
    % Assign labels based on image name
    if strfind(p, 'per')
        tlabels(k) = 1;
    end
    if strfind(p, 'crop')
        tlabels(k) = 1;
    end
end

testingFeatures = [testingFeatures; tfeatures];
testingLabels = [testingLabels; tlabels];

disp('Testing Set Completed');

%% Compress testingFeatures and trainingFeatures 
compSize = 800;
randMat = zeros(length(trainingFeatures), compSize);

for j=1:compSize
    for k=1:length(trainingFeatures)
        R = floor(1 + (7-1).*rand(1));
        if R == 1 || R == 3 || R == 5
            randMat(k, j) = 1;
        elseif R == 2 || R == 4 || R == 6
            randMat(k, j) = -1;
        end
    end
end

trainingFeatures = trainingFeatures*randMat; 

for l=1:compSize
    for q=1:length(testingFeatures)
        R = floor(1 + (7-1).*rand(1));
        if R == 1 || R == 3 || R == 5
            randMat(q, l) = 1;
        elseif R == 2 || R == 4 || R == 6
            randMat(q, l) = -1;
        end
    end
end

testingFeatures = testingFeatures*randMat;

%% Train the object classifier using fitcesvm learner
classifier = fitcsvm(trainingFeatures, trainingLabels);
predictedLabels = predict(classifier, testingFeatures);
disp('SVM Labels Predited...');

linearclassifier = fitcdiscr(trainingFeatures, trainingLabels);
linearLabels = predict(linearclassifier, testingFeatures);
disp('Linear Discriminant Labels Predited...');

t = ClassificationDiscriminant.template();
ens = fitensemble(trainingFeatures, trainingLabels, 'AdaBoostM1', 50, 'Tree');
ensPredict = predict(ens, testingFeatures);
disp('AdaBoost Labels Predicted...');

%% Calculate SVM predictor accuracy
count = 0;
falsePos = 0;
falseNeg = 0;

for p=1:length(testingLabels)
    if testingLabels(p)== predictedLabels(p)
        count = count + 1;
    end
    if testingLabels(p)== 0 &&  predictedLabels(p) == 1
        falsePos = falsePos +1;
    end 
    if testingLabels(p)== 1 &&  predictedLabels(p) == 0
        falseNeg = falseNeg +1;
    end 
end
disp('---------------SVM-----------------');
disp('Number of Correct Predictions SVM');
disp(count);
disp(length(testingLabels));

disp('Accuracy SVM');
idx = (testingLabels()==1);

p = length(testingLabels(idx));
n = length(testingLabels(~idx));
N = p+n;

tp = sum(testingLabels(idx)==predictedLabels(idx));
tn = sum(testingLabels(~idx)==predictedLabels(~idx));
accuracy = (tp+tn)/N;
disp(accuracy);

disp('Number of False Pos');
disp(falsePos);
disp('Number of False Neg');
disp(falseNeg);
%% Calculate AdaBoost accuracy
count = 0;
falsePos = 0;
falseNeg = 0;

for q=1:length(testingLabels)
    if testingLabels(q)==ensPredict(q)
        count = count + 1;
    end
    if testingLabels(q)== 0 &&  ensPredict(q) == 1
        falsePos = falsePos +1;
    end 
    if testingLabels(q)== 1 &&  ensPredict(q) == 0
        falseNeg = falseNeg +1;
    end 
end
disp('---------------ADABOOST-----------------');
disp('Number of Correct Predictions AdaBoost');
disp(count);
disp(length(testingLabels));

disp('Accuracy AdaBoost');
idx = (testingLabels()==1);

p = length(testingLabels(idx));
n = length(testingLabels(~idx));
N = p+n;

tp = sum(testingLabels(idx)==ensPredict(idx));
tn = sum(testingLabels(~idx)==ensPredict(~idx));

accuracy = (tp+tn)/N;

disp(accuracy);
disp('Number of False Pos');
disp(falsePos);
disp('Number of False Neg');
disp(falseNeg);
%% Calculate Lin Disc accuracy
count = 0;
falsePos = 0;
falseNeg = 0;

for r=1:length(testingLabels)
    if testingLabels(r)==linearLabels(r)
        count = count + 1;
    end
    if testingLabels(r)== 0 &&  linearLabels(r) == 1
        falsePos = falsePos +1;
    end 
    if testingLabels(r)== 1 &&  linearLabels(r) == 0
        falseNeg = falseNeg +1;
    end 
end
disp('---------------LIN DISC-----------------');
disp('Number of Correct Predictions Lin Disc');
disp(count);
disp(length(testingLabels));

disp('Accuracy Lin Disc');
idx = (testingLabels()==1);

p = length(testingLabels(idx));
n = length(testingLabels(~idx));
N = p+n;

tp = sum(testingLabels(idx)==linearLabels(idx));
tn = sum(testingLabels(~idx)==linearLabels(~idx));

accuracy = (tp+tn)/N;

disp(accuracy);

disp('Number of False Pos');
disp(falsePos);
disp('Number of False Neg');
disp(falseNeg);