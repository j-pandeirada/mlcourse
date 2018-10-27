function y_cod = onehotenconding(y,num_labels)

    y_cod = zeros(length(y),num_labels);
    for i=1:length(y)
        y_cod(i,y(i)) = 1;
    end
    
    y_cod = y_cod';
end

