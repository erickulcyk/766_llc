function [ l2mean2 ] = L2Mean2( m )
%Calls L2Mean on each row in m
    l2mean2 = zeros(size(m,1),size(m,2));
    for i=1:size(m,1)
        l2mean2(i,:) = L2Mean(m(i,:));
    end
end

