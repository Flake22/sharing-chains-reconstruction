function [Hist, file_path] = dct_coef_hist_from_list(file)
%   This function extracts DCT coefficient histograms of images provided in
%   a file
%   Input:
%       file: the file containing all paths
%       channel: 1 (luminance), 2 (chrominance)
%       Nc: number of analyzed DCT coeff in each 8x8 blocks
%       BT: the limit of histogram ==> number of bins = 2*BT + 1
%   Output:
%       Hist: the nxd histogram matrix; n is the number of images; d is the
%       number of histogram bins
%       file_path: a list containing file paths (used to split data later)


    Nc = 9;         % number of AC coefficients (zig zag scan)
    BT = 20;        % maximum bin value => number of bins = 2*BT+1  

    fid = fopen(file, 'r');
    file_path = textscan(fid, '%s', 'delimiter', '\n', 'whitespace', '');
    file_path = file_path{1};
    N = length(file_path);
    fprintf('\nNumber of images: %d \n', N);
    
    Hist = zeros(N, (2*BT+1)*Nc);
    for i = 1:N
        %[~,file_name,ext] = fileparts(file_path{i});
        %fprintf('process file: %s\n', [file_name ext]);
        fprintf('process file: %s\n', file_path{i});
        Hist(i,:) = dct_coef_hist(file_path{i});
    end
end
