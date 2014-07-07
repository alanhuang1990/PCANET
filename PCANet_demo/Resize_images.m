function [ cell_images ] = Resize_images( cell_images, dest_size )
%RESIZE_IMAGES Summary of this function goes here
%   Detailed explanation goes here

n = length(cell_images);

for i=1:n
    cell_images{i} = imresize(cell_images{i},dest_size);
end

end

