function [test_labels, num_correct] = lltree (image_dir, data_dir, dSize, K, gridN, gridM, folderNames, depth, testFileNames, testLabelVector, id, pfig)

folderNamesCompact = folderNames(id~=-1);

num_folders = size(folderNames,1);

num_folders_compact = size(folderNamesCompact,1);
folderConcat = cell(1,num_folders_compact);
for i = 1:num_folders_compact
    folderConcat(i) = cellstr(folderNamesCompact(i).name);
end

folderConcat = strjoin(folderConcat,'_');

total_files = 1;

filenames = cell(100*num_folders_compact,1);
training_label_vector = [];
num_test_files = 0;
for ind = 1:num_folders
    if(id(ind)==-1)
        continue;
    end
    
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

params.gridSpacing = 8;
params.patchSize = 16;
params.dictionarySize = dSize;
params.numTextonImages = total_files;

params.maxImageSize = 1000
params.oldSift = false

canSkip = 1;
saveSift = 1;
    
GenerateSiftDescriptors( filenames, image_dir, data_dir, params, canSkip, pfig )
CalculateDictionary( filenames, image_dir, data_dir, '_sift.mat', ['_' num2str(depth) '_' num2str(size(filenames,1)) '_' folderConcat], params, canSkip, pfig );
inFName = fullfile(data_dir, sprintf('dictionary_%s_%d.mat', [num2str(depth) '_' num2str(size(filenames,1)) '_' folderConcat], params.dictionarySize));
load(inFName,'dictionary');
fprintf('Loaded top level texton dictionary: %d textons\n', params.dictionarySize);

featureSize = gridN*gridM*params.dictionarySize;

max_train_fileName = [ data_dir num2str(depth) '_' folderConcat '_train_c_out_' num2str(params.dictionarySize) '_' num2str(K) '_' num2str(gridN) '-' num2str(gridM) '_' num2str(size(filenames,1)) '_maxL2.mat'];
sum_train_fileName = [ data_dir num2str(depth) '_' folderConcat '_train_c_out_' num2str(params.dictionarySize) '_' num2str(K) '_' num2str(gridN) '-' num2str(gridM) '_' num2str(size(filenames,1)) '_sum.mat'];

if(exist(max_train_fileName,'file')~=0 && exist(sum_train_fileName,'file')~=0 && canSkip)
else
    [train_c_out_max,train_c_out_sum] = GetLLCFeatures(K, gridN, gridM, dictionary, data_dir, filenames, pfig);
    train_c_out_max = sparse(reshape(train_c_out_max,[size(filenames,1) featureSize]));
    train_c_out_sum = sparse(reshape(train_c_out_sum,[size(filenames,1) featureSize]));
    save(max_train_fileName,'train_c_out_max');
    save(sum_train_fileName,'train_c_out_sum');
end

load(max_train_fileName, 'train_c_out_max');
load(sum_train_fileName, 'train_c_out_sum');

centers = zeros(num_folders, featureSize);
for i = 1:size(filenames,1)
    label = training_label_vector(i);
    %a = squeeze(train_c_out_max(i,:));
    a = squeeze(train_c_out_sum(i,:));
    b = squeeze(centers(label,:));
    centers(label,:) = a + b;
end

if(size(id(id~=-1),2)>2)
    largest = [-1,-1];
    for i = 1:num_folders
        minimum = 100000;
        if(id(i)==-1)
            continue
        end
        
        for j=1:num_folders
            if(i~=j && (id(i)~= id(j)) && id(i)~=-1 && id(j)~=-1)
                b = centers(i,:)-centers(j,:);
                c = dot(b,b);
                
                if(c<minimum)
                    minimum=c;
                end
            end
        end
        
        if(minimum>largest(1,1))
            largest(1,1)=minimum;
            largest(1,2)=i;
        end
    end
    
    fid = -1;
    for i = 1:num_folders
        if(i~=largest(1,2) && id(i)~=-1)
            if(fid==-1)
                fid = id(i);
            else
                id(i) = fid;
            end
        end
    end
end
%{
while (size(unique(id(id~=-1)),2)>2)
    e = 10000*ones(num_folders,2);
    smallest = [10000,-1, -1];
    for i = 1:num_folders
        for j=1:num_folders
            if(i~=j && (id(i)~= id(j)) && id(i)~=-1 && id(j)~=-1)
                b = centers(i,:)-centers(j,:);
                c = dot(b,b);
                
                if(c<smallest(1,1))
                    smallest(1,1)=c;
                    smallest(1,2)=id(i);
                    smallest(1,3)=id(j);
                end
                
                if(c<e(i,1))
                    e(i,1)=c;
                    e(i,2)=j;
                end
            end
        end
    end
    
    for i = 1:num_folders
        if(id(i)== smallest(2))
            id(i) = smallest(3);
        end
    end
end
     %}
for i = 1:size(id,2)
    if(id(i)~=-1)
        fid = id(i);
        break;
    end
end

label_vector = (double(id(training_label_vector)==fid)).';
%model = train(label_vector, train_c_out_max, '-c 1000 -s 0 -e .001');
model = train(label_vector, train_c_out_sum, '-c 1000 -s 2 -e .001');

%%%%%%%%%%%%%%%%%%%%%%START TEST%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

test_label_vector = (double(id(testLabelVector)==fid)).';

max_test_fileName = [ data_dir num2str(depth) '_' folderConcat '_test_c_out_' num2str(params.dictionarySize) '_' num2str(K) '_' num2str(gridN) '_' num2str(gridM) '-' num2str(size(test_label_vector,1)) '_maxL2.mat'];
sum_test_fileName = [ data_dir num2str(depth) '_' folderConcat '_test_c_out_' num2str(params.dictionarySize) '_' num2str(K) '_' num2str(gridN) '_' num2str(gridM) '-' num2str(size(test_label_vector,1)) '_sum.mat'];
roc_fileName = [ data_dir num2str(depth) '_' folderConcat '_test_c_out_' num2str(params.dictionarySize) '_' num2str(K) '_' num2str(gridN) '_' num2str(gridM) '-' num2str(size(test_label_vector,1)) '_roc.mat'];


% construct codebook
if(saveSift)
    GenerateSiftDescriptors( testFileNames, image_dir, data_dir, params, canSkip, pfig )
end

if(exist(max_test_fileName,'file')~=0 && exist(sum_test_fileName,'file')~=0 && canSkip)
else
    [test_c_out,test_c_out_sum] = GetLLCFeatures(K, gridN, gridM, dictionary, data_dir, testFileNames, pfig);
    test_c_out_max = sparse(reshape(test_c_out,[size(testFileNames,1) gridN*gridM*params.dictionarySize]));
    test_c_out_sum = sparse(reshape(test_c_out_sum,[size(testFileNames,1) gridN*gridM*params.dictionarySize]));
    save(max_test_fileName,'test_c_out_max');
    save(sum_test_fileName,'test_c_out_sum');
end

%save memory by loading precomputed arrays later

load(max_test_fileName,'test_c_out_max');
load(sum_test_fileName,'test_c_out_sum');

[predicted_label, accuracy, prob_estimates] = predict(test_label_vector, test_c_out_sum, model, '-b 1');
%{
[predicted_label_train, accuracy_train, prob_estimates_train] = predict(label_vector, train_c_out_sum, model, '-b 1');

prob_estimates_index = [ (1:size(prob_estimates_train,1))', prob_estimates_train];
[Y,I] = sort(prob_estimates_index(:,1));
prob_estimates_index = prob_estimates_index(I,:);
roc = zeros(size(prob_estimates_index,1),1);
s=size(prob_estimates_index,1);
lv = label_vector;
for i=1:s
    if(i==1)
       if( lv(prob_estimates_index(i,1))==0)
           roc(i)=1/s;
       end
    else
        if( lv(prob_estimates_index(i,1))==0)
           roc(i)=roc(i-1)+1/s;
        else
           roc(i)=roc(i-1);
        end
    end
end
%}
%save(roc_fileName, 'roc');

%[predicted_label, accuracy, prob_estimates] = predict(test_label_vector, test_c_out_max, model, '-b 1');


binId = id==fid;
nbinId = id~=fid & id~=-1;

thres = .5;

rightFolderNames = folderNames(nbinId(id~=-1));
if(size(rightFolderNames,1)==1)
    for i = 1:size(nbinId,2)
       if(nbinId(i))
           break;
       end
    end
    
    a = i*double(prob_estimates(:,1)<thres)==testLabelVector';
    b = testLabelVector(a);
    rightCorrect = size(b,2);
    rightLabels = predicted_label*id(i);
    %rightCorrect = size(testLabelVector((prob_estimates(probestimates>=.5))==testLabelVector);
else
    rightid = id;
    for i = 1:size(binId,2)
        if(binId(i) || id(i)==-1)
           rightid(i)=-1;
        else
           rightid(i)=i;
        end
    end
    
    rightTestFiles = testFileNames(prob_estimates(:,1)<thres);
    rightTestLabelVector = testLabelVector(prob_estimates(:,1)<thres);
    [rightLabels, rightCorrect] = lltree (image_dir, data_dir, dSize, K, gridN, gridM, folderNames, depth+1, rightTestFiles, rightTestLabelVector,rightid, pfig);
end

leftFolderNames = folderNames(binId(id~=-1));
if(size(leftFolderNames,1)==1)
    for i = 1:size(binId,2)
       if(binId(i))
           break;
       end
    end
    
    a = i*double(prob_estimates(:,1)>=thres)==testLabelVector';
    b = testLabelVector(a);
    leftCorrect = size(b,2);
    leftLabels = predicted_label*id(i);
    %leftCorrect = size((prob_estimates(prob_estimates(:,1)>=.5))==testLabelVector,2);
else
    leftid = id;
    for i = 1:size(nbinId,2)
        if(nbinId(i) || id(i)==-1)
           leftid(i)=-1;
        else
           leftid(i)=i;
        end
    end
    
    leftTestFiles = testFileNames(prob_estimates(:,1)>=thres);
    leftTestLabelVector = testLabelVector(prob_estimates(:,1)>=thres);
    [leftLabels, leftCorrect] = lltree (image_dir, data_dir, dSize, K, gridN, gridM, folderNames, depth+1, leftTestFiles, leftTestLabelVector, leftid, pfig);
end



num_correct = rightCorrect+leftCorrect;
test_labels = [];
disp(['done. Num correct, total, fraction: ' num2str(num_correct) ' ' num2str(size(testLabelVector,2)) ' ' num2str(num_correct/size(testLabelVector,2))]);