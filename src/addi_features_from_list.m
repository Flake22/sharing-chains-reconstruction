function [Features, file_path] = addi_features_from_list(file)
%   This function extracts medadata-based features of images provided in
%   a file
%   Input:
%       file: the file containing all paths
%   Output:
%       features: extracted features
%       file_path: a list containing file paths (used to split data later)
    fid = fopen(file, 'r');
    file_path = textscan(fid, '%s', 'delimiter', '\n', 'whitespace', '');
    file_path = file_path{1};
    N = length(file_path);
    fprintf('\nNumber of images: %d \n', N);
    Features = zeros(N, 152);
    for i = 1:N
        [~,file_name,ext] = fileparts(file_path{i});
        fprintf('process file: %s\n', [file_name ext]);
        Features(i,:) = addi_features(file_path(i));
    end
end

