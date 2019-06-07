
for plate=1:18
    overall_path = strcat('Z:/sorger/data/IN_Cell_Analyzer_6000/Connor/Fixed MCF10 Common/20x full exp/20180905_Updated/plate', num2str(plate),'/analysis/');
    cd(overall_path);
    list_masks = struct2cell(dir('*_allFeatures.txt'));
    list_masks = list_masks(1,:);
    output_fn = strcat('plate',num2str(plate));
    cd('d:/histocat/');
    for mask = list_masks
        mask_fn = char(mask);
        mask_fn = strcat(mask_fn(1:8),'_nucleiLM.tif');
        img_fn = strcat(mask_fn(1:8),'_FFC.tif');
        Headless_histoCAT_loading(overall_path,img_fn, overall_path,mask_fn, 'd:/histocat/mcf10a_histoCAT_loading.csv',output_fn)
        clear Tiff_name
        clear Mask_all
        clear Fcs_Interest_all
        clear HashID
    end
    
end