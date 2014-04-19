function [ cout,coutsum ] = GetLLCFeatures (K, N, M, dictionary, data_dir, file_names,pfig)
cout = zeros(size(file_names,1),N*M,size(dictionary,1));
coutsum = zeros(size(file_names,1),N*M,size(dictionary,1));

%The number of images to load into ram at once
%Increasing this reduces the number of times the parfor has to be set up
SIZE = 480; %Tested with SIZE = 160; consumes approximately 3GB of RAM in addition to matlab overhead
%Probably can be increased to 480 without running out of memory, but depends on size of N*M.
%size(file_names,1) is way too big

x_all = cell(size(SIZE,1));
x_all_N = cell(size(SIZE,1));
x_all_M = cell(size(SIZE,1));

%for N*M=25*25, codebook 512, this crashes for anything but SIZE = 2
dictionaryt = dictionary';
kdtree = vl_kdtreebuild(dictionaryt);

for u = 1:SIZE:size(file_names,1)
    for i = u:min(u+SIZE-1,size(file_names,1))
        sp_progress_bar(pfig,3,4,i,size(file_names,1),'Loading SIFT Feature Data: ');
        [dirN base] = fileparts(file_names{i});
        baseFName = fullfile(dirN, base);
        inFName = fullfile(data_dir, sprintf('%s%s', baseFName, '_sift.mat'));
        x_fn = inFName;
        load(x_fn, 'features');
        x_all{i-u+1} = features.data;
        x_x = (features.x - min(features.x) + 1);
        x_all_N{i-u+1} = ceil(N.*double(x_x)./max(x_x));
        x_y = (features.y - min(features.y) + 1);
        x_all_M{i-u+1} = ceil(M.*double(x_y)./max(x_y));
    end

    parfor i = u:min(u+SIZE-1,size(file_names,1))
        disp(['Computing LLC Weights for ' num2str(i)]);
        x = x_all{i-u+1};
        xt = x';
        x_n = x_all_N{i-u+1};
        x_m = x_all_M{i-u+1};
        ind = (x_n-1).*M + x_m;
        
        t = getCurrentTask();
        disp(['Task ID: ' num2str(t.ID)]);
        
        c_out = zeros(N*M,size(x,1),size(dictionary,1));
        % find k codebook words
        for j = 1:size(x,1)
            B = zeros(K,size(dictionary,2));
            [index, ~] = vl_kdtreequery(kdtree, dictionaryt, xt(:,j), 'NumNeighbors', K) ;

            %dists = zeros(size(dictionary,1),2);
            %for k = 1:size(dictionary,1)
            %    dists(k,1) = sum((dictionary(k,:)-x(j,:)).*(dictionary(k,:)-x(j,:)));
            %    dists(k,2) = k;
            %end
            %[Y,I] = sort(dists(:,1));
            %distsort = dists(I,:);
            %B(:,:) = dictionary(distsort(1:K,2),:);
            B(:,:) = dictionary(index,:);
            
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
            c_in(index) = c_hat_j;
    
            c_out(ind(j),j,:) = c_in';
        end
        %{
        cout(i,:,:) = max(c_out(:,:,:),[],2);
        cnorm = L2Mean2(squeeze(cout(i,:,:)));
        cout(i,:,:) = squeeze(cout(i,:,:))./cnorm;
        coutsum(i,:,:) = sum(c_out(:,:,:),2);
        csumabs = repmat(sum(abs(squeeze(coutsum(i,:,:))),2),[1 size(dictionary,1)]);
        coutsum(i,:,:) = squeeze(coutsum(i,:,:))./csumabs;
        %}
        
        tmp = zeros(N*M,size(dictionary,1));
        tmp = reshape(max(c_out(:,:,:),[],2),size(tmp));
        cnorm = L2Mean2(tmp);
        tmp =  tmp./cnorm;
        tmp(isnan(tmp(:))) = 0;
        cout(i,:,:) = tmp;
        tmp = reshape(sum(c_out(:,:,:),2),size(tmp));
        csumabs = repmat(sum(abs(tmp),2),[1 size(dictionary,1)]);
        tmp = tmp./csumabs;
        tmp(isnan(tmp(:))) = 0;
        coutsum(i,:,:) = tmp;
        
    end
end

