function [out] = generate_html(name,path)
%   This function generates html file
%   Input:
%       key: name of the file
%       path:path of the file
%   Output:
%       this function generate the html file and return
%       out:file path of the html file
    
        name_html=strcat(name,'.html');
        folder=('src/html_files');
        out=fullfile(folder,name_html);
        command_str=sprintf("src/exiftool -htmlDump %s > %s",string(path),out);
        system(command_str);
end

