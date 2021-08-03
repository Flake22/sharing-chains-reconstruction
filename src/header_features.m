function features = header_features(key,path)
%	This function extracts features from a JPEG header
%   Input:
%       key: name of the file
%       path:path of the file
%   Output:
%       features: the output features
    file_path=generate_html(key,path);
    html=(fileread(file_path));
    
    word='DHT';
    word1='unused';
    word2='APP13';
    word3='APP2';
    word4='SOF0';
    word5='SOF2';
    %word6='COM Comment segment';
    word6='cmp3';
    word7='JPEG DRI';
    
    word8='af 03 01 22';
    word9='b0 03 01 22';
    
    words={word,word1,word2,word3,word4,word5,word6,word7,word8,word9};
    features = zeros(1,10); %number of features
    for w=1:length(words) 
        features(w)=count(html,words(w));
    end
end

