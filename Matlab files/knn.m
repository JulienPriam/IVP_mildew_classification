%% Classes to consider: 0 & 9
clear all;

X = []; % samples
Y = []; % associated labels


for m = 1:2

    if m == 1
        myFolder = 'potato_dataset\train\healthy';
    else
        myFolder = 'potato_dataset\train\blight';
    end


    filePattern = fullfile(myFolder, '*.JPG'); % Change to whatever pattern you need.
    theFiles = dir(filePattern);
    
    for k = 1 : length(theFiles)
    
        file = fullfile(theFiles(k).folder, theFiles(k).name);
        img = imread(file);
        
        % Extract color channels.
        img_r = img(:,:,1); % Red channel
        img_g = img(:,:,2); % Green channel
        img_b = img(:,:,3); % Blue channel
    
        img_r = reshape(img_r, [1, 65536]);
        img_g = reshape(img_g, [1, 65536]);
        img_b = reshape(img_b, [1, 65536]);
    
        sample = [img_r img_g img_b];
        
        X = [X; sample];
        if m == 1
            Y = [Y; 0];
        else
            Y = [Y; 1];
        end
        
    end
end



%% For partition of 80% - 20%

[trainId,valId, testId] = dividerand(size(X,1), 0.80, 0.20, 0);
X_train = X(trainId,:);
Y_train = Y(trainId,:);
X_val = X(valId,:);
Y_val = Y(valId,:);
% X_test = X(testId,:);
% Y_test = Y(testId,:);

accuracy = zeros(15, 1);

for k = 1:50
    model = fitcknn(X_train, Y_train,'NumNeighbors',k);
    Y_pred = predict(model, X_val);
    accuracy(k) = sum(Y_val == Y_pred,'all')/numel(Y_pred);
end

plot(accuracy)