function llc (image_dir, data_dir)

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

N = 100;  % feature dimension
K = 5;  % number of nearest neighbours

params.maxImageSize = 1000;
params.gridSpacing = 8;
params.patchSize = 16;
params.dictionarySize = 200;
params.numTextonImages = 50;
params.pyramidLevels = 3;

if(~exist('canSkip','var'))
    canSkip = 1
end
if(~exist('saveSift','var'))
    saveSift = 1
end

pfig = sp_progress_bar('Building Spatial Pyramid');

% construct codebook
if(saveSift)
    GenerateSiftDescriptors( filenames, image_dir, data_dir, params, canSkip, pfig )
end
CalculateDictionary( filenames, image_dir, data_dir, '_sift.mat', params, canSkip, pfig );
inFName = fullfile(data_dir, sprintf('dictionary_%d.mat', params.dictionarySize));
load(inFName,'dictionary');
fprintf('Loaded texton dictionary: %d textons\n', params.dictionarySize);
%compute features


%ones matrix
one = ones(K, 1);

% compute data covariance matrix
B_1x = B - one *x';
C = B_1x * B_1x';

% reconstruct LLC code
c_hat = C \ one;
c_hat = c_hat /sum(c_hat);

c_hat

model = train(training_label_vector, c_hat);

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

%compute features

%ones matrix
one = ones(K, 1);

% compute data covariance matrix
B_1x = B - one *x';
C = B_1x * B_1x';

% reconstruct LLC code
c_hat = C \ one;
c_hat = c_hat /sum(c_hat);

c_hat

[predicted_label, accuracy, prob_estimates] = predict(testing_label_vector, c_hat, model);