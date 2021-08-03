function extract_dct_coef_map(input_path, channel, output_path, out_file)
% This function extracts the map of dct coefficients
% Input:
%   input_path: root folder
%   channel: 1: Chrominance, 2: Chrominance-Blue, 3: Chrominance-Red
%   output_path: where to save the maps. Saved files having the same name
%   as input files, but end with .hdf5

    [file_path, file_name] = get_file_list(input_path, [], []);
    mapping = cell(length(file_name), 2); 
    err_inds = false(length(file_name),1);
    fprintf('There are %d files\n', length(file_name));
    counter = 0;
    for i = 1:length(file_name)
        name = file_name{i};
        fprintf('Processing: %s\n', name);
        if ~contains(name, '.jpg') && ~contains(name, '.jpeg')...
           && ~contains(file_name{i}, '.JPEG') && ~contains(name, '.JPEG')
            err_inds(i) = true;
            continue
        end
        jpg_obj = jpeg_read(file_path{i});
        roi = [floor(jpg_obj.image_height/8)*8, floor(jpg_obj.image_width/8)*8];
        dct_coef_map = jpg_obj.coef_arrays{channel};
        dct_coef_map = int16(dct_coef_map(1:roi(1), 1:roi(2))); % size of dct_coef_map is multiple of 8
        %[~,name,~] = fileparts(name);
        new_name = ['shared' '-' sprintf('%04d',counter) '.hdf5'];
        ascii_file_path = int16(file_path{i});
        if exist([output_path filesep new_name], 'file')
            delete([output_path filesep new_name]);
        end
        h5create([output_path filesep new_name], '/map', size(dct_coef_map), 'Datatype', 'int16');
        h5create([output_path filesep new_name], '/path', size(ascii_file_path), 'Datatype', 'int16');
        h5write([output_path filesep new_name], '/map', dct_coef_map);
        h5write([output_path filesep new_name], '/path', ascii_file_path);
        mapping{i,1} = file_path{i}; % from
        mapping{i,2} = [output_path filesep new_name]; % to
        counter = counter + 1;
    end
    mapping(err_inds, :) = [];
    save(out_file, 'mapping');
end

