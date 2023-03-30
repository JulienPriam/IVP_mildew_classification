%% Perform edge detection with Sobel mask
clear all;

features_tab = [];



for m = 1:2
    if m == 1
        myFolder = 'potato_dataset\train\healthy';
    else 
        myFolder = 'potato_dataset\train\blight';
    end
    
    filePattern = fullfile(myFolder, '*.JPG'); % Change to whatever pattern you need.
    theFiles = dir(filePattern);
    
    for k = 1 : length(theFiles)
        features = [];
    
        file = fullfile(theFiles(k).folder, theFiles(k).name);
        img = imread(file);
        img_gray = rgb2gray(img);
    
        map = img_gray <= 20;
        new_value = img_gray(1, 1);
        
        for i = 1:256
            for j = 1:256
                if map(i, j) == 1
                    img_gray(i, j) = new_value;
                end
            end
        end
        
        img_gauss = imgaussfilt(img_gray, 3);
        
        numColors = [2, 3, 4, 5];
        for i = numColors
            L = imsegkmeans(img_gray, i);
            img_seg = labeloverlay(img_gray, L);
        
            L_gauss = imsegkmeans(img_gauss, i);
            img_seg_gauss = labeloverlay(img_gauss, L_gauss);
        
     
    %         figure;
    %         subplot(221), imshow(img), title('Original Image');
    %         subplot(222), imshow(img_gray), title('Gray image with shadow removed');
    %         subplot(223), imshow(img_seg), title('Segmentation on original image');
    %         subplot(224), imshow(img_seg_gauss), title('Segmentation with filtering');
        
            
            [X_no_dither,~] = rgb2ind(img_seg_gauss,i,'nodither');
            h = histogram(X_no_dither).Values;
            h = sort(h);
        
            features = [features, h];
        
        end
        features_tab = [features_tab; features];
    end
end

T = array2table(features_tab);
T.Properties.VariableNames(1:14) = {'seg_1', 'seg_2', 'seg_3', 'seg_4', 'seg_5', 'seg_6', 'seg_7', 'seg_8', 'seg_9', 'seg_10', 'seg_11', 'seg_12', 'seg_13', 'seg_14'};
writetable(T,'seg_features.csv')





