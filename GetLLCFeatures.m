function [ cout ] = GetLLCFeatures (K, dictionary, data_dir, file_names,pfig)

cout = zeros(size(file_names,1),size(dictionary,1));
x_all = cell(size(file_names,1));

%The number of images to load into ram at once
%Increasing this reduces the number of times the parfor has to be set up
SIZE = 160; %Tested with SIZE = 160; consumes approximately 3GB of RAM in addition to matlab overhead
            %Probably can be increased to 480 without running out of memory.
            %size(file_names,1) is way too big
            
for u = 1:SIZE:size(file_names,1)
    for i = u:min(u+SIZE,size(file_names,1))
        sp_progress_bar(pfig,3,4,i,size(file_names,1),'Loading SIFT Feature Data: ');
        [dirN base] = fileparts(file_names{i});
        baseFName = fullfile(dirN, base);
        inFName = fullfile(data_dir, sprintf('%s%s', baseFName, '_sift.mat'));
        x_fn = inFName;
        load(x_fn, 'features');
        x_all{i} = features.data; 
    end
    
    parfor i = u:min(u+SIZE,size(file_names,1))
        disp(['Computing LLC Weights for ' num2str(i)]);
        x = x_all{i};
    
        c_out = zeros(size(x,1),size(dictionary,1));%zeros(size(file_names,1),size(x,1),size(dictionary,1));
        % find k codebook words
        for j = 1:size(x,1)
            B = zeros(K,size(dictionary,2));
            dists = zeros(size(dictionary,1),2);
            for k = 1:size(dictionary,1)
                dists(k,1) = sum((dictionary(k,:)-x(j,:)).*(dictionary(k,:)-x(j,:)));
                dists(k,2) = k;
            end
            [Y,I] = sort(dists(:,1));
            distsort = dists(I,:);
            B(:,:) = dictionary(distsort(1:K,2),:);
    
            %ones matrix
            one = ones(K, 1);
    
            % compute data covariance matrix
            B_1x = B - one * x(j,:);
            c_j = B_1x * B_1x';
    
            % reconstruct LLC code
            c_hat_j = c_j \ one;
            c_hat_j = c_hat_j /sum(c_hat_j);
    
            %c_hat
            c_in = zeros(size(dictionary,1),1);
            for k = 1:K
                c_in(distsort(k,2)) = c_hat_j(k);
            end
    
            c_out(j,:) = c_in';% cout(i,j,:)
        end
        cout(i,:) = sum(c_out(:,:),1); %sum(c_out(i,:,:),2);
        cout(i,:) = cout(i,:)./sum(cout(i,:));
    end
end

