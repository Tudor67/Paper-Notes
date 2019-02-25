clear;
close all;

%% read filenames
dir_name = './counting datasets/trancos/TRANCOS_v3/images';
files = dir(dir_name);

%% imshow images
for i = 3:4:6
    img_name = files(i).name;
    dots_coord_name = files(i + 1).name;
    dots_name = files(i + 2).name;
    mask_name = files(i + 3).name;
    
    % load images and coordinates
    img = imread([dir_name '/' img_name]);
    dots_coord = load([dir_name '/' dots_coord_name]);
    dots = imread([dir_name '/' dots_name]);
    mask = getfield(load([dir_name '/' mask_name]), 'BW');

    img_mask = img;
    for ch = 1:3
        img_mask(:,:,ch) = img_mask(:,:,ch) .* uint8(mask);
    end
    
    dots_coord_img = zeros(size(mask));
    for k = 1:length(dots_coord)
        dots_coord_img(dots_coord(k, 2) + [-1, 0, 1], dots_coord(k, 1) + [-1, 0, 1]) = 1;
    end
    
    % imshow
    figure;
    subplot(1, 4, 1);
    imshow(img);
    
    %figure;
    subplot(1, 4, 2);
    imshow(mask);
    
    %figure;
    subplot(1, 4, 3);
    imshow(img_mask);
    
    %figure;
    subplot(1, 4, 4);
    imshow(dots_coord_img);
    
    %figure;
    %imshow(dots_coord_img);
end