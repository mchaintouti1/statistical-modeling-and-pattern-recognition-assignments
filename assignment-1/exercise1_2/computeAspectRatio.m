function aRatio = computeAspectRatio(image,i)

    aRatio = 0;
    
    [num_rows, num_cols] = size(image);

    %num_rows
    %num_cols

    % Fill your code
    
    % Set the threshold value
    threshold = 0; % Change the threshold value if needed

    % Initialize variables for min and max row and column indices
    min_row = num_rows;
    max_row = 0;
    min_col = num_cols;
    max_col = 0;

    % Loop through each pixel in the image
    for row = 1:num_rows
       for col = 1:num_cols
           % Check if the pixel value is above the threshold
           if image(row, col) > threshold
                % Update min and max row and column indices
               if row < min_row
                   min_row = row;
               end
                if row > max_row
                   max_row = row;
                end
               if col < min_col
                   min_col = col;
               end
                if col > max_col
                   max_col = col;
               end
            end
        end
    end
% Calculate the height and width of the smaller shape
height = max_row - min_row + 1;
width = max_col - min_col + 1;

    aRatio = width/height;
    
    if (i==1304)
      figure();
      colormap(bone);
      imagesc(image);
    
      hold on;
      rectangle('Position', [min_col-0.5, min_row-0.5, width, height], ...
      'EdgeColor', 'r', 'LineWidth', 2);
      hold off;
    end
end

