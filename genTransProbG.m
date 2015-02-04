function [ pTable ] = genTransProbG(topo, sigma, alpha, sp )
%GENTRANSPROB Create a transition probability table 

nElements = prod(topo);
pTable = zeros(nElements);

%format of the table:   
%row:current state
%column: next state
%sp -> 1 for peaks, 2 for random tp table

if(nargin<4)
   sp=0; 
end

if(alpha>1 && alpha <0)
   error('alpha must be between 0 and 1');  
end


pUni = 1./(prod(topo)); 

if (sp==0)
    for row=1: topo(1)
        for col=1:topo(2)
           rownum= sub2ind(topo, row,col);
                      
           for el=1: nElements
              [x y] = ind2sub(topo, el);
              pTable(rownum, el) = exp(-0.5*((x-row).^2 + (y-col).^2)./sigma^2);
              %pTable(rownum, el) = alpha;
              %sum= sum + pTable(rownum, el);
           end
           
           %pTable(rownum, :) =pTable(rownum, :)./sum; 
            
        end
    end
    

    %get the sum of line in the middle and normalize by it
    nc = sum(pTable(ceil(topo(1)/2), :));
    pTable = pTable./nc;
    
end %endif

if (sp==1)
    pTable = eye(nElements);   
end %endif

if (sp==2)
    pTable = rand(nElements);  
    pTable = bsxfun(@rdivide, pTable, sum(pTable,2));
end %endi
    
    
    %add uniform probability
    pTable = alpha*pUni*ones(nElements) + (1-alpha)*pTable;
    %pTable = pUni*ones(nElements) + pTable;
    %pTable = pUni*ones(nElements);

end

