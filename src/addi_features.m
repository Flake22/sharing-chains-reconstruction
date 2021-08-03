function features = addi_features(imName)
%	This function extracts additional features from a JPEG header
%   Input:
%       jheader: JPEG header (returned obj of jpeg_read)
%   Output:
%       features: the output features

    jheader = jpeg_read(char(imName));
    
    quant_tab_1 = zeros(1,64);
    if ~isempty(jheader.quant_tables{1})
        quant_tab_1 = reshape(jheader.quant_tables{1}, 1, []);
    end
    quant_tab_2 = zeros(1,64);
    if length(jheader.quant_tables) > 1 && ~isempty(jheader.quant_tables{2})
        quant_tab_2 = reshape(jheader.quant_tables{2}, 1, []);
    end
    n_dc_huff = size(jheader.dc_huff_tables, 2);
    n_ac_huff = size(jheader.ac_huff_tables, 2);
    comp_info = zeros(1, 18);
    nComp = size(jheader.comp_info, 2);
    for c = 1:nComp
        comp_info((c-1)*6+1:c*6) = [jheader.comp_info(c).component_id,... 
                                    jheader.comp_info(c).h_samp_factor,...
                                    jheader.comp_info(c).v_samp_factor,...
                                    jheader.comp_info(c).quant_tbl_no,...
                                    jheader.comp_info(c).dc_tbl_no,...
                                    jheader.comp_info(c).ac_tbl_no];
    end
    opt_coding = jheader.optimize_coding;
    prog_mode = jheader.progressive_mode;
    features = [quant_tab_1 quant_tab_2 n_dc_huff n_ac_huff comp_info opt_coding prog_mode ...
        max([jheader.image_width jheader.image_height]) min([jheader.image_width jheader.image_height])];
end

