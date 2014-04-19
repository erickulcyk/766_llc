function Example(image_dir, train_data_dir)

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

total_files = total_files-1;

% return pyramid descriptors for all files in filenames
 params.numTextonImages = total_files;
 params.dictionarySize = 512;
[pyramid_all] = BuildPyramid(filenames,image_dir,train_data_dir, train_data_dir,params,1, 1); %n X 21 X 200

Training_instance_matrix = sparse(pyramid_all);

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

params.numTextonImages = total_files;
params.dictionarySize = 1024;

pyramid_all2 = BuildPyramid(filenames,image_dir,train_data_dir, train_data_dir, params, 1, 1); 
disp('Done with pyramid 2');
testing_instance_matrix = sparse(pyramid_all2);
disp('Done with sparse');

predictedClass = zeros(total_files,1);
maxEstimate = 10*ones(total_files,1);

for ind = 3:num_folders
    
  label_vector = (double(training_label_vector==ind)).';
  label_model = train(label_vector, Training_instance_matrix, '-c 1000 -s 0 -e .001');
  test_label_vector = (double(testing_label_vector==ind)).';

  [predicted_label, accuracy, prob_estimates] = predict(test_label_vector, testing_instance_matrix, label_model, '-b 1');
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
