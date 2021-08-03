clear all; close all; clc;

%% Request settings
%dataset
addpath(genpath('src'));
d = dir('dataset');
d = d(3:end);
subFolders = d([d.isdir]);
fn = {subFolders.name};
[indx,tf] = listdlg('PromptString', 'Which dataset do you want to use?',...
    'SelectionMode','single','ListString',fn);
ds = fn{indx};

if not(isfolder('features'))
    mkdir('features');
end

%configuration
conf = input('Which configuration do you want to use? 1, 2 or 3. 1 = default\n');
if not(ismember(conf, [1 2 3]))
    conf = 1;   
end

%features
features = input('Type 1 to extract metadata features from JPEG or 2 for DCT histograms or 3 for HEADER. Default runs all\n');

%% Extract features
split = {'train', 'test', 'validation'};

for i=1:length(split)
    fname_in = sprintf('dat/%s_config_%d_%s.dat', ds, conf, split{i})
    if features == 1
        [Features, file_path] = addi_features_from_list(fname_in);
        fname_out = sprintf('features/addi_features_%s_%d_%s.mat', ds, conf, split{i});
        save(fname_out, 'Features', 'file_path');
    elseif features == 2
        [Features, file_path] = dct_coef_hist_from_list(fname_in);
        fname_out = sprintf('features/dct_coef_%s_%d_%s.mat', ds, conf, split{i});
        save(fname_out, 'Features', 'file_path');
    elseif features == 3

        if not(isfolder('src/html_files'))
            mkdir('src/html_files');
        end

        [Features, file_path] = header_from_list(fname_in);
        fname_out = sprintf('features/header_%s_%d_%s.mat', ds, conf, split{i});
        save(fname_out, 'Features', 'file_path');
    else
        [Features, file_path] = addi_features_from_list(fname_in);
        fname_out = sprintf('features/addi_features_%s_%d_%s.mat', ds, conf, split{i});
        save(fname_out, 'Features', 'file_path');

        [Features, file_path] = dct_coef_hist_from_list(fname_in);
        fname_out = sprintf('features/dct_coef_%s_%d_%s.mat', ds, conf, split{i});
        save(fname_out, 'Features', 'file_path');

        [Features, file_path] = header_from_list(fname_in);
        fname_out = sprintf('features/header_features_%s_%d_%s.mat', ds, conf, split{i});
        save(fname_out, 'Features', 'file_path');
    end
end