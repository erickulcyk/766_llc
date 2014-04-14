function llcComplete (image_dir, data_dir)

K=5;

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

total_files = total_files-1;

params.gridSpacing = 8;
params.patchSize = 16;
params.dictionarySize = 1024;
params.numTextonImages = total_files;
params.pyramidLevels = 3;

if(~exist('canSkip','var'))
    canSkip = 1;
end
if(~exist('saveSift','var'))
    saveSift = 1;
end

pfig = sp_progress_bar('LLC!!!!');

% construct codebook
if(saveSift)
    GenerateSiftDescriptors( filenames, image_dir, data_dir, params, canSkip, pfig )
end
CalculateDictionary( filenames, image_dir, data_dir, '_sift.mat', params, canSkip, pfig );
inFName = fullfile(data_dir, sprintf('dictionary_%d.mat', params.dictionarySize));
load(inFName,'dictionary');
fprintf('Loaded texton dictionary: %d textons\n', params.dictionarySize);

train_c_out = sparse(GetLLCFeatures(K, dictionary, data_dir, filenames, pfig));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


total_files = 1;

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

total_files = total_files-1;

% construct codebook
if(saveSift)
    GenerateSiftDescriptors( filenames, image_dir, data_dir, params, canSkip, pfig )
end

test_c_out = sparse(GetLLCFeatures(K, dictionary, data_dir, filenames, pfig));

predictedClass = zeros(total_files,1);
maxEstimate = 10*ones(total_files,1);

for ind = 3:num_folders

    label_vector = (double(training_label_vector==ind)).';
    model = train(label_vector, train_c_out, '-s 0 -e .001');
    test_label_vector = (double(testing_label_vector==ind)).';
    [predicted_label, accuracy, prob_estimates] = predict(test_label_vector, test_c_out, model, '-b 1');
    
    for i = 1:size(test_label_vector,1)
      if(abs(prob_estimates(i,2)-prob_estimates(i,1))<=maxEstimate(i))
          maxEstimate(i) =abs(prob_estimates(i,2)-prob_estimates(i,1));
          predictedClass(i) = ind;
      end
    end
end

correct = 0;
for i = 1:size(test_label_vector,1)
    if(predictedClass(i)==testing_label_vector(i))
        correct = correct+1;
    end
end

correct/total_files

cm = confusionmat(testing_label_vector.', predictedClass);
cm



