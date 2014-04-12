% Example of how to use the BuildPyramid function
% set image_dir and data_dir to your actual directories
image_dir = 'L:\scene_categories\'; 
train_data_dir = 'L:\scene_data\train\';

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

% return pyramid descriptors for all files in filenames
 params.numTextonImages = total_files;
[pyramid_all, scalingVector] = BuildPyramid(filenames,image_dir,train_data_dir, train_data_dir,params,1, ones(1,128), 1); %n X 21 X 200

% for faster performance, compile and use hist_isect_c:
K = hist_isect(pyramid_all, pyramid_all);
Training_instance_matrix = sparse(K);



total_files = 1;
test_data_dir = 'L:\scene_data\test\';

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
pyramid_all2 = BuildPyramid(filenames,image_dir,test_data_dir, train_data_dir, params, 1, scalingVector, 1); 
K = hist_isect(pyramid_all2, pyramid_all2);
testing_instance_matrix = sparse(K);



predictedClass = zeros(total_files-1,1);
maxEstimate = -10*ones(total_files-1,1);
minNegEstimate = 10*ones(total_files-1,1);
minNegpredict = zeros(total_files-1,1);
for ind = 3:num_folders
  label_vector = (double(training_label_vector==ind)).';
  label_model = train(label_vector, Training_instance_matrix, '-s 0');
  test_label_vector = (double(testing_label_vector==ind)).';

test_label_vector = label_vector;
testing_instance_matrix = Training_instance_matrix;
  
  [predicted_label, accuracy, prob_estimates] = predict(test_label_vector, testing_instance_matrix, label_model, '-b 1');
  for i = 1:size(test_label_vector,1)
      if (predicted_label(i) && abs(prob_estimates(i,1))>=maxEstimate(i))
          maxEstimate(i) = abs(prob_estimates(i,1));
          predictedClass(i) = ind;
      end
      if(~predicted_label(i) && abs(prob_estimates(i,1))<=minNegEstimate(i))
          minNegEstimate(i) = abs(prob_estimates(i,1));
          minNegpredict(i) = ind;
      end
  end
end

correct = 0;
for i = 1:size(test_label_vector,1)
    if(predictedClass(i) == 0)
        predictedClass(i) = minNegpredict(i);
    end
    
    if(predictedClass(i)==testing_label_vector(i))
        correct = correct+1;
    end
end

correct/(total_files-1)

cm = confusionmat(testing_label_vector.', predictedClass);
cm
