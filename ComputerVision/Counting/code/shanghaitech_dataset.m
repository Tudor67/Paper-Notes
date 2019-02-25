clear;
close all;

%% read img_names
dataset_dir_name = './ShanghaiTech/part_B/train_data';
img_dir_name = [dataset_dir_name '/images'];
gt_dir_name = [dataset_dir_name '/ground-truth'];
img_names = dir(img_dir_name);

figure;
%% imshow images
for i = 3:4
    img_name = img_names(i).name;

    % load images and coordinates
    img = imread([img_dir_name '/' img_name]);
    dots_coord = load([gt_dir_name '/GT_' img_name(1:end-4) '.mat']);
    dots_coord = int32(dots_coord.image_info{1, 1}.location);
    count = size(dots_coord, 1);
    
    [h, w, ch] = size(img);
    dots_coord_img = img;
    size(dots_coord_img)
    for k = 1:length(dots_coord)
        x = dots_coord(k, 2) + int32([-2, -1, 0, 1, 2]);
        x = max(1, min(h, x));
        y = dots_coord(k, 1) + int32([-2, -1, 0, 1, 2]);
        y = max(1, min(w, y));
        dots_coord_img(x, y, 2) = 255;
    end
    
    % imshow
    %figure;
    subplot(1, 2, i - 2);
    imshow(dots_coord_img);
    title(count); 
    %pause;
end