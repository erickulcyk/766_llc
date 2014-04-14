function [ c_out ] = GetLLCFeatures (K, dictionary, data_dir, file_names,pfig)

c_out = zeros(size(file_names,1),size(dictionary,1));

for i = 1:size(file_names,1)
    %if(mod(i,10)==0)
        sp_progress_bar(pfig,3,4,i,size(file_names,1),'Computing LLC weights: ');
    %end

    [dirN base] = fileparts(file_names{i});
    baseFName = fullfile(dirN, base);
    inFName = fullfile(data_dir, sprintf('%s%s', baseFName, '_sift.mat'));
    x_fn = inFName;
    load(x_fn, 'features');
    x = features.data;
    
    % find k codebook words
    for j = 1:size(x,1)
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
        
        c_out(i,:) = c_in' +c_out(i,:);% max(c_in',c_out(i,:));
    end
    c_out(i,:) = c_out(i,:)./sum(c_out(i,:));

end

