function Example(image_dir, train_data_dir)

folderNames = dir(fullfile(image_dir, '*'));
num_folders = size(folderNames,1);

total_files = 1;

train_svm_num = 98;

nn_train_label_vector = [];
total_nn_files = 1;
nn_filenames = cell((100-train_svm_num)*(num_folders-2),1);
nn_svm_label = [];

filenames = cell(train_svm_num*(num_folders-2),1);
training_label_vector = [];
num_test_files = 0;
for ind = 3:num_folders
    fnames = dir(fullfile([image_dir,folderNames(ind).name],'\*.jpg'));
    num_files = min(100,size(fnames,1));
    actual_train_files = min(train_svm_num,num_files);
    num_test_files = num_test_files + max(0,size(fnames,1)-100);
    training_label_vector(total_files:total_files+actual_train_files-1) = ind;
    nn_svm_label(total_nn_files:total_nn_files+num_files-actual_train_files-1) = ind;
    
    nn_label = zeros(1,num_folders-2)';
    nn_label(ind-2) = 1;
    a = repmat(nn_label,1,num_files-actual_train_files);
    nn_train_label_vector(1:size(nn_label,1),total_nn_files:total_nn_files+num_files-actual_train_files-1) = a ;
    
    for f = 0:actual_train_files-1
        filenames{total_files+f} = [folderNames(ind).name,'\',fnames(f+1).name];
    end
    
    for f = actual_train_files: num_files-1
        nn_filenames{total_nn_files+f-actual_train_files} = [folderNames(ind).name,'\',fnames(f+1).name];
    end

    total_nn_files = total_nn_files + num_files-actual_train_files;
    total_files= total_files + actual_train_files;
end

total_nn_files = total_nn_files-1;
total_files = total_files-1;

% return pyramid descriptors for all files in filenames
 params.numTextonImages = total_files;
 params.dictionarySize = 256;
[pyramid_all] = BuildPyramid(filenames,image_dir,train_data_dir, train_data_dir,params,1, 1); %n X 21 X 200

[nn_pyramid] = BuildPyramid(nn_filenames, image_dir, train_data_dir, train_data_dir, params,1,1);
nn_instance_matrix = sparse(nn_pyramid);

Training_instance_matrix = sparse(pyramid_all);

total_files = 1;

nn_test_label_vector = [];

filenames = cell(num_test_files,1);
testing_label_vector = [];
for ind = 3:num_folders
    fnames = dir(fullfile([image_dir,folderNames(ind).name],'\*.jpg'));
    num_files = size(fnames,1) - min(100, size(fnames,1));
    testing_label_vector(total_files:total_files+num_files-1) = ind;
    
    nn_label = zeros(1,num_folders-2)';
    nn_label(ind-2) = 1;
    a = repmat(nn_label,1,num_files);
    nn_test_label_vector(1:size(nn_label,1),total_files:total_files+num_files-1) = a ;
    
    for f = 0:num_files-1
        filenames{total_files+f} = [folderNames(ind).name,'\',fnames(101+f).name];
    end
    total_files = total_files + num_files;
end

total_files = total_files-1;

params.numTextonImages = total_files;
params.dictionarySize = 256;

pyramid_all2 = BuildPyramid(filenames,image_dir,train_data_dir, train_data_dir, params, 1, 1); 
disp('Done with pyramid 2');
testing_instance_matrix = sparse(pyramid_all2);
disp('Done with sparse');

%testing_instance_matrix = Training_instance_matrix;
%    testing_label_vector = training_label_vector;
 %   total_files = size(training_label_vector,2);

predictedClass = zeros(total_files,1);
maxEstimate = 10*ones(total_files,1);

setdemorandstream(391418381);
net = patternnet(num_folders-2);

nn_train_input = zeros(num_folders-2,size(nn_train_label_vector,2));
nn_test_input = zeros(num_folders-2,size(testing_label_vector,2));


for ind = 3:num_folders
    
  label_vector = (double(training_label_vector==ind)).';

  label_model = train(label_vector, Training_instance_matrix, '-c 1000 -s 4 -e .01');
  test_label_vector = (double(testing_label_vector==ind)).';

  [predicted_label, accuracy, prob_estimates] = predict(test_label_vector, testing_instance_matrix, label_model, '-b 1');
  [nn_predicted_label, nn_accuracy, nn_prob_estimates] = predict(nn_svm_label', nn_instance_matrix, label_model, '-b 1');
  nn_train_input(ind-2,:) = nn_prob_estimates(:,1)';
  nn_test_input(ind-2,:) = prob_estimates(:,1)';
  
  av = mean(prob_estimates(:,1));
  for i = 1:size(test_label_vector,1)
      if(ind==3 && .5-prob_estimates(i,1)<=maxEstimate(i) )
          maxEstimate(i) =.5-prob_estimates(i,1);
          %maxEstimate(i) = (.5-prob_estimates(i,1))/av;
          predictedClass(i) = ind;
      end
      if (ind>3 && prob_estimates(i,1)<=maxEstimate(i))
      %if(abs(prob_estimates(i,2)-prob_estimates(i,1))<=maxEstimate(i))
      %    maxEstimate(i) =abs(prob_estimates(i,2)-prob_estimates(i,1));
          maxEstimate(i) =prob_estimates(i,1);
          %maxEstimate(i) =prob_estimates(i,1)/av;
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

[net,tr] = train(net,nn_train_input,nn_train_label_vector);
testY = net(nn_test_input);
testIndices = vec2ind(testY);
plotconfusion(nn_test_label_vector,testY)

[c,cm] = confusion(nn_test_label_vector,testY)

fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

plotroc(nn_test_label_vector,testY)

