% Example of how to use the BuildPyramid function
% set image_dir and data_dir to your actual directories
image_dir = 'L:\scene_categories\'; 
data_dir = 'L:\scene_data\train\';

folderNames = dir(fullfile(image_dir, '*'));
num_folders = size(folderNames,1);

total_files = 1;

filenames = cell(100*(num_folders-2),1);
training_label_vector = [];
num_test_files = 0;
for ind = 3:num_folders
    fnames = dir(fullfile([image_dir,folderNames(ind).name],'\*.jpg'));
    num_files = min(100,size(fnames,1));
    num_test_files = num_test_files + max(0,size(fnames,1)-100);
    training_label_vector(total_files:total_files+num_files-1) = ind;
    for f = 0:num_files-1
        filenames{total_files+f} = [folderNames(ind).name,'\',fnames(f+1).name];
    end
    total_files= total_files + num_files;
end
% for other parameters, see BuildPyramid

%fnames = dir(fullfile(image_dir, '*\*.jpg'));
%num_files = size(fnames,1);
%filenames = cell(num_files,1);



% return pyramid descriptors for all files in filenames
params.numTextonImages = total_files;
pyramid_all = BuildPyramid(filenames,image_dir,data_dir,params); %n X 21 X 200

% build a pyramid with a different dictionary size without re-generating the
% sift descriptors.
%params.dictionarySize = 400;
%pyramid_all2 = BuildPyramid(filenames,image_dir,data_dir,params,1); % n X 21 X dictionarySize


%control all the parameters
%params.maxImageSize = 1000;
%params.gridSpacing = 1;
%params.patchSize = 16;
%params.dictionarySize = 200;
%params.numTextonImages = 300;
%params.pyramidLevels = 2;
%pyramid_all = BuildPyramid(filenames,image_dir,[data_dir '2'],params,1); %n X maxImageSize

% compute histogram intersection kernel
%K = hist_isect(pyramid_all, pyramid_all); %n X n

% for faster performance, compile and use hist_isect_c:
K = hist_isect(pyramid_all, pyramid_all);

training_label_vector = training_label_vector.';
Training_instance_matrix = sparse(K);
model = train(training_label_vector, Training_instance_matrix);

total_files = 1;
data_dir = 'L:\scene_data\test\';

filenames = cell(num_test_files,1);
testing_label_vector = [];
for ind = 3:num_folders
    fnames = dir(fullfile([image_dir,folderNames(ind).name],'\*.jpg'));
    num_files = size(fnames,1) - min(100, size(fnames,1));
    testing_label_vector(total_files:total_files+num_files-1) = ind;
    for f = 0:num_files-1
        filenames{total_files+f} = [folderNames(ind).name,'\',fnames(101+f).name];
    end
    total_files = total_files + num_files;
end

params.numTextonImages = total_files;
pyramid_all2 = BuildPyramid(filenames,image_dir,data_dir, params); 
testing_label_vector = testing_label_vector.';
K = hist_isect(pyramid_all2, pyramid_all2);
testing_instance_matrix = sparse(K);

[predicted_label, accuracy, prob_estimates] = predict(testing_label_vector, testing_instance_matrix, model);
%[predicted_label] = predict(testing_label_vector, testing_instance_matrix, model [, 'liblinear_options', 'col']);

