function [test_labels, num_correct] = lltreeComplete(image_dir, data_dir, dSize, K, gridN, gridM)
folderNames = dir(fullfile(image_dir, '*'));
folderNames = folderNames(3:size(folderNames,1));
depth = 1;
num_folders = size(folderNames,1);

num_test_files = 0;
for ind = 1:num_folders
    fnames = dir(fullfile([image_dir,folderNames(ind).name],'\*.jpg'));
    num_test_files = num_test_files + max(0,size(fnames,1)-100);
end

total_files = 1;

testFileNames = cell(num_test_files,1);
testLabelVector = [];
for ind = 1:num_folders
    fnames = dir(fullfile([image_dir,folderNames(ind).name],'\*.jpg'));
    num_files = size(fnames,1) - min(100, size(fnames,1));
    testLabelVector(total_files:total_files+num_files-1) = ind;
    for f = 0:num_files-1
        testFileNames{total_files+f} = [folderNames(ind).name,'\',fnames(101+f).name];
    end
    total_files = total_files + num_files;
end

try
  parpool(8);
catch
end
run('vlfeat-0.9.18/toolbox/vl_setup')

pfig = sp_progress_bar('LLC!!!!');

[test_labels, num_correct] = lltree(image_dir, data_dir, dSize, K, gridN, gridM, folderNames, depth, testFileNames, testLabelVector, 1:num_folders, pfig);