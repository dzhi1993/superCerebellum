%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test script for the comparison of the mean correlation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A = [];
% B = [];
% 
% for i=1:1000 
%     % simulate 20 voxel pairs 
%     X1=normrnd(0,1,5,200);
%     X1=bsxfun(@minus,X1,mean(X1,1)); 
%     [N, K] = size(X1);
%     %X2 = X1+  normrnd(0,1,5,20);
%     %X2=bsxfun(@minus,X2,mean(X2,2)); 
%     
%     % 1. Calcualte correlation for each pair- averaged across pairs 
%     corMat = triu(corr(X1),1);    
%     avrgCor_1 = sum(corMat(:)) / (((1+K)*K/2)-K); 
%     
%     % 2. Calculate covariance and varaince 
%     SD = sqrt(sum(X1.^2)/N);
%     VAR = triu(SD'*SD, 1);
%     COV = triu(X1'*X1/N, 1);
%     
%     % Caluculate mean(COV)/sqrt(mean(VAR)*mean(VAR))
%     meanCOV = sum(COV(:)) / ((1+K)*K/2-K);
%     meanVAR = sum(VAR(:)) / ((1+K)*K/2-K);
%     avrgCor_2 = meanCOV / meanVAR;
%     % Record the result 
%     A = [A; avrgCor_1];
%     B = [B; avrgCor_2];
% end
%     
% % Compare means - should be the same.  
% diff = mean(A)-mean(B);
% 
% % Compare the variability across the simulation 
% [h,p,ci,stat] = ttest2(A,B);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Making random square parcellation
numParcel = 16;
parcel=zeros(numParcel);
for i = 1:numParcel
    [row,col] = ind2sub([sqrt(numParcel) sqrt(numParcel)],i);
    row_start = (row-1)*8+1;
    row_end = row*8;
    col_start = (col-1)*8+1;
    col_end = col*8;
    parcel(row_start:row_end,col_start:col_end)=i;
end
parcel = parcel(:);
sameReg=(bsxfun(@ne,parcel',parcel)+1);
sameReg=sameReg(:);

% Making distance metric 1024 nodes
distMat = zeros(1024);
N = size(distMat,1);
for i=1:N
    for j=1:N
        [a1, b1] = ind2sub([32 32], i);
        [a2, b2] = ind2sub([32 32], j);
        distMat(i,j) = sqrt((a1-a2)^2 + (b1-b2)^2);
    end
end
avrgDs = distMat(:);

RR=[];
for k=1:100
    % making random functional map (This probably wrong!)
    Data =[];
    for i=1:61
        random = randn(32);
        this_Data = smooth_kernel2D(random,5);
        this_Data = this_Data(:);
        %this_Data = bsxfun(@minus,this_Data,mean(this_Data));
        Data = [Data this_Data];
    end

    % Compute mean VAR and COV
    Data = bsxfun(@minus,Data,mean(Data,2));
    Data = single(Data');
    [K,P]=size(Data);
    clear A;

    SD = sqrt(sum(Data.^2)/K);
    VAR = (SD'*SD);
    COV = Data'*Data/K;
    COR = corrcoef(Data);
    COR = COR(:);
    
    dist = [1,2,3,4,5,6,7];
    
    for i=1:length(dist)
        for bw=[1 2]
            fprintf('.\n');
            inBin = avrgDs==dist(i) & sameReg==bw;
            %R.meanVAR(in,1) = nanmean(VAR(inSp(inBin)));
            %R.meanCOV(in,1) = nanmean(COV(inSp(inBin)));
            R.corr(dist(i),bw) = nanmean(COR(inBin));
        end
    end
    R.DCBC = R.corr(:,1) - R.corr(:,2);
    RR = addstruct(RR,R);
end



