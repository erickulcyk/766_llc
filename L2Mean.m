function [ l2mean ] = L2Mean( v )
%Outputs the l2mean of a vector v

    l2mean = v./sqrt(sum(v.*v));

end

