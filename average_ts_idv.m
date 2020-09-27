%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get the average beta value for each subject, then transfer from voxel-based 
% to surface-based
%
% Author: Da Zhi, Sep 5th, 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


type=[];
% % (1) Directories
% rootDir           = '/Users/jdiedrichsen/Data/super_cerebellum';
baseDir         = 'D:/data';
rootDir         = 'Z:\data\super_cerebellum_new';
% rootDir         = '/Volumes/MotorControl/data/super_cerebellum_new';
wbDir           = fullfile(rootDir,'sc1','surfaceWB');
sc1Dir          = [rootDir '/sc1'];
sc2Dir          = [rootDir '/sc2'];
behavDir        = ['data'];
imagingDir      = ['imaging_data'];
imagingDirRaw   = ['imaging_data_raw'];
dicomDir        = ['imaging_data_dicom'];
anatomicalDir   = ['anatomicals'];
suitDir         = ['suit'];
caretDir        = ['surfaceCaret'];
regDir          = ['RegionOfInterest'];
connDir         = ['connectivity_cerebellum'];
saveDir         = 'D:\python_workspace\brain_parcellation_project\data';

%==========================================================================

% % (2) Hemisphere and Region Names
Hem       = {'L','R'};
hemname   = {'CortexLeft','CortexRight'};
subj_name = {'s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11',...
    's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23','s24',...
    's25','s26','s27','s28','s29','s30','s31'};
goodsubj  = [2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31];
returnSubjs=[2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31];
expStr    = {'sc1','sc2'};
resolution = '32k';


% for s = 1:length(goodsubj)
%     load(fullfile(sc1Dir,regDir,'glm4',subj_name{goodsubj(s)},sprintf('betas_cortex.mat')));
%     
%     for sess = 1:16 % 1-8 runs (average of sc1)
%         tempBeta_L = B{1}.betasUW( (sess-1)*29+2 : (sess-1)*29+29, :);
%         surfbased_beta_L = sc1_sc2_neocortical('SURF:voxel2vertex', 'data', tempBeta_L.', 'hem', 1, 'sn', goodsubj(s));
%         avrgBeta_L(:,1:28,sess)=surfbased_beta_L;
%         
%         tempBeta_R = B{2}.betasUW( (sess-1)*29+2 : (sess-1)*29+29, :);
%         surfbased_beta_R = sc1_sc2_neocortical('SURF:voxel2vertex', 'data', tempBeta_R.', 'hem', 2, 'sn', goodsubj(s));
%         avrgBeta_R(:,1:28,sess)=surfbased_beta_R;
%         
%         clear tempBeta_L tempBeta_R surfbased_beta_L surfbased_beta_R
%     end
%     
%     avrgBeta_sc12_L=nanmean(avrgBeta_L,3);
%     avrgBeta_sc12_R=nanmean(avrgBeta_R,3);
%     
%     avrgBeta_sc1_L=nanmean(avrgBeta_L(:,:,1:8),3);
%     avrgBeta_sc1_R=nanmean(avrgBeta_R(:,:,1:8),3);
%     
%     avrgBeta_sc2_L=nanmean(avrgBeta_L(:,:,9:16),3);
%     avrgBeta_sc2_R=nanmean(avrgBeta_R(:,:,9:16),3);
%     
%     [m, n] = find(isnan(avrgBeta_sc12_L));
%     nanIndex_L = unique(m);
%     
%     clear m n;
%     [m, n] = find(isnan(avrgBeta_sc12_R));
%     nanIndex_R = unique(m);
%     
%     save(fullfile(saveDir, sprintf('beta_avrg_noInstr_%s.mat', subj_name{goodsubj(s)})), 'avrgBeta_sc1_L', 'avrgBeta_sc1_R',...
%         'avrgBeta_sc2_L', 'avrgBeta_sc2_R', 'avrgBeta_sc12_L', 'avrgBeta_sc12_R', 'nanIndex_L', 'nanIndex_R');
%     
%     clear B m n;
% end


% for s = 1:length(goodsubj)
%     load(fullfile(saveDir,sprintf('beta_avrg_noInstr_%s.mat', subj_name{goodsubj(s)})));
%     
%     avrgBetaL(:,:,s) = avrgBeta_sc12_L;
%     avrgBetaR(:,:,s) = avrgBeta_sc12_R;
%     
%     % clear avrgBeta_sc1_L avrgBeta_sc1_R avrgBeta_sc2_L avrgBeta_sc2_R avrgBeta_sc12_L avrgBeta_sc12_L;
% end
% 
% avrgBeta_sc12_L=nanmean(avrgBetaL,3);
% avrgBeta_sc12_R=nanmean(avrgBetaR,3);
% 
% % find the row index of NaN values for both hemisphere
% [nanIndex_L n] = find(isnan(avrgBeta_sc12_L(:,1)));
% [nanIndex_R n] = find(isnan(avrgBeta_sc12_R(:,1)));
% 
% save(fullfile(saveDir, sprintf('group_beta_noInstr.mat')), 'avrgBeta_sc12_L', 'avrgBeta_sc12_R', 'nanIndex_L', 'nanIndex_R');

% K = 17;
% normalisation = 3;
% 
% load(sprintf('affinity_%d.mat', resolution));
% cl = SpectralClustering(affinity,K,normalisation);
% cl = full(cl);
% [m, n] = size(cl);
% 
% for i=1:m
%     labels(i,1) = find(cl(i,:));
% end
% 
% labels_L = gifti(sprintf('Icosahedron-%d.32k.L.label.gii', resolution));
% labels_R = gifti(sprintf('Icosahedron-%d.32k.R.label.gii', resolution));

% hem = [1 2];
% taskSet = [1 2];
% D=dload(fullfile(baseDir,'sc1_sc2_taskConds.txt'));
% % condIdx=find(D.overlap == 0 & D.StudyNum == ts);
% 
% for s=returnSubjs
%     for h=hem
%         A=gifti(fullfile(rootDir,'sc1','surfaceWB',subj_name{s},sprintf('%s.%s.swcon.%s.func.gii',subj_name{s},Hem{h},resolution)));
%         for ts=taskSet
%             condIdx=find(D.StudyNum == ts);
%             data = A.cdata(:,condIdx);
%             outfile = fullfile(saveDir,subj_name{s},sprintf('%s.%s.swcon.exp%d.%s.mat',subj_name{s},Hem{h},ts,resolution));
%             save(outfile, 'data');
%         end
%     end
% end


%%%%%%%%%%%%%%%%% Making random parcellations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sphere = gifti('fs_LR.32k.L.sphere.surf.gii');
% parcel = gifti('Icosahedron-162.32k.L.label.gii');
% 
% parcels = parcel.cdata;
% mask = ones(size(parcels));
% mask(parcels==0) = 0;
% parcels = parcels(parcels~=0);
% rotation = [];
% 
% for i = 1:100
%     [nullModel, moved] = generate_null_model(parcels, sphere, mask);
%     rotation(:,i) = nullModel; 
% end
% 
% % Loading medial wall
% M = gifti('Yeo_JNeurophysiol11_7Networks.32k.L.label.gii');
% rotation(M.cdata==0,:)=0;
% save('nullModel_rotated_162_L.mat', 'rotation');


%%%%%%%%%%%%%%%%%%%% Eval %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%par = [42,162,362,642,1002];
par = 162;
%bins = [0.2, 1, 2.5, 5];
bins = 5;

for j = 1:length(par)
    left = load(sprintf('nullModel_rotated_%d_L.mat', par(j)));
    %right = load(sprintf('nullModel_rotated_%d_R.mat', par(j)));
    for b = 1:length(bins)
        for i = 51:100
            %parcel = [left.rotation(:, i) right.rotation(:, i)];
            parcel = left.rotation(:, i);

            T=sc1_sc2_neocortical('Eval:DCBC','hem',1,'parcel',parcel,'condType','unique','taskSet',1,'distFile','distSphere_sp','bins',[0:bins(b):35]);
            save(sprintf('Eval_Icosahedron_%d_Sphere_L_unique_%d_%s.mat',par(j),i,string(bins(b))),'-struct','T');
        end
    end
end


%%%%%%%%%%%%%%%%%%%% simulation of re-weighted method %%%%%%%%%%%%%%%%%%%
% num_W = [0       27918      140688      184642      138420];
% num_B = [0        1254        7176       16072       14082];
% 
% %load('nums.mat')
% %num_W = [0           0           0       27918      140688           0       36562      148080      116188       22232];
% %num_B = [0           0           0        1254        7176           0        2660       13412       11582        2500];
% 
% DCBC=[];
% for iter= 1:100
%     rw=[];
%     rb=[];
%     ratio_w=[];
%     ratio_b=[];
%     rw_adjust=[];
%     rb_adjust=[];
%     rw_weighted=[];
%     rb_weighted=[];
%     for i = 1:length(num_W)
%         this_rw=normrnd(5-4*(i-1)/length(num_W),1,[1,num_W(i)]);
%         this_rb=normrnd(5-4*(i-1)/length(num_B),1,[1,num_B(i)]);
%         this_w_ratio = 1/((num_W(i)/sum(num_W))/((num_B(i)/sum(num_B) + num_W(i)/sum(num_W))/2));
%         this_b_ratio = 1/((num_B(i)/sum(num_B))/((num_B(i)/sum(num_B) + num_W(i)/sum(num_W))/2));
% 
%         rw=[rw this_rw];
%         rb=[rb this_rb];
%         ratio_w = [ratio_w this_w_ratio];
%         ratio_b = [ratio_b this_b_ratio];
%         rw_adjust = [rw_adjust this_w_ratio*mean(this_rw)];
%         rb_adjust = [rb_adjust this_b_ratio*mean(this_rb)];
%         rw_weighted=[rw_weighted this_rw*this_w_ratio];
%         rb_weighted=[rb_weighted this_rb*this_b_ratio];
%     end
%     rw_mean = nanmean(rw);
%     rb_mean = nanmean(rb);
% 
%     %rw_adjust = [this_w_ratio_2*rw2 this_w_ratio_3*rw3 this_w_ratio_4*rw4 this_w_ratio_5*rw5];
%     %rb_adjust = [this_b_ratio_2*rb2 this_w_ratio_3*rb3 this_w_ratio_4*rb4 this_w_ratio_5*rb5];
%     rw_mean_adjust = nansum(rw_adjust)/nansum(ratio_w);
%     rb_mean_adjust = nansum(rb_adjust)/nansum(ratio_b);
% 
%     rw_mean_weighted = nanmean(rw_weighted);
%     rb_mean_weighted = nanmean(rb_weighted);
%     this_DCBC = rw_mean_weighted-rb_mean_weighted;
%     DCBC = [DCBC this_DCBC];
% end
% DCBC_mean = mean(DCBC);




% %%%%%----------- Plot number of vertices pairs in small bins %%%%%
% g1 = [0.5 0.5 0.5]; % Gray 1
% %toPlot={'Icosahedron_42','Icosahedron_162','Icosahedron_362','Icosahedron_642','Icosahedron_1002'};
% %toPlot={'Icosahedron_42'};
% CAT.linecolor={'r','b','b','b','g','k','k',g1};
% CAT.linestyle={'-','-',':','--','-','-',':','-'};
% CAT.linewidth=2;
% CAT.markertype='none';
% CAT.errorcolor={'r','b','b','b','g','k','k',g1};
%         
% D.N_B=[];
% D.N_W=[];
% D.distmin=[];
% D.distmax=[];
% for s = 1:10
%     temp=load(sprintf('Eval_Icosahedron_42_Sphere_all_%d.mat',s));
%     D.distmin=[D.distmin;temp.distmin(1:10,:)];
%     D.distmax=[D.distmax;temp.distmax(1:10,:)];
%     D.N_W=[D.N_W;temp.N(1:10,:)];
%     D.N_B=[D.N_B;temp.N(11:20,:)];
% end

% Here is the bin - also the x axis for the plotting
% D.binC = (D.distmin+D.distmax)/2;
% lineplot(D.binC,D.N_W,'CAT',CAT,'linecolor','r','errorcolor','k');
% hold on
% lineplot(D.binC,D.N_B,'CAT',CAT,'linecolor','b','errorcolor','k');


%%%%%%----------- Making color map for the random parcellation
% labelRGBA       = zeros(max(nullModel),4);
% labelRGBA(1,:)  = [0 0 0 1];
% col = hsv(max(nullModel));
% col = col(randperm(max(nullModel)),:); % shuffle the order so it's more visible
% for i=1:max(nullModel)
%     labelRGBA(i+1,:)=[col(i,:) 1];
% end
% 
% % G = surf_makeLabelGifti(nullModel,'anatomicalStruct',hemname{1},'labelRGBA',[colorcube(size(unique(nullModel),1)) ones(size(unique(nullModel),1),1)]);
% G = surf_makeLabelGifti(nullModel,'anatomicalStruct',hemname{1},'labelRGBA',labelRGBA);
% save(G, 'rotated_Icosahedron-1002.32k.L.label.gii');





%%%%%% case 'Eval:DCBC'             % Get the DCBC evaluation
% sn=returnSubjs;
% hem = [1 2];
% resolution = '32k';
% taskSet = [1 2];
% condType = 'unique'; % Evaluate on all or only unique conditions?
% bins = [0:5:40];     % Spatial bins in mm
% parcel = [];         % N*2 matrix for both hemispheres
% RR=[];
% distFile = 'distAvrg_sp';
% icoRes = 2562;
% rate = 5;            % Indicate how many small bins in each of the bins
% vararginoptions(varargin,{'sn','hem','bins','parcel','condType','taskSet','resolution','distFile','icoRes','rate'});
% D=dload(fullfile(baseDir,'sc1_sc2_taskConds.txt'));
% numBins = numel(bins)-1;
% for h=hem
%     load(fullfile(wbDir,'group32k',distFile));
% 
%     % Now find the pairs that we care about to safe memory 
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % CHANGE: Exclude only medial wall! 
%     % The node indices of medial wall are defined by taking the union of seven existing parcellations, including
%     % 'Glasser','Yeo17','Yeo7','Power2011','Yeo2015','Desikan', and 'Dextrieux', which stored as external 
%     % .mat files "medialWallIndex_%hem.mat" for both hems.
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % mw = load(fullfile(wbDir, sprintf('group%s', resolution), sprintf('medialWallIndex_%s.mat', Hem{h})));
%     mw = gifti(fullfile(wbDir, sprintf('group%s', resolution), sprintf('Icosahedron-%d.%s.%s.label.gii',icoRes,resolution,Hem{h})));
% 
%     % Here, we use it to find the union with the label-0 of Icosahedron-(icoRes) as our final medial wall
%     % vertIdx is the indices that we want (without medial wall)
%     vertIdx = setdiff(1:size(avrgDs), find(mw.cdata(:,1)==0)); 
%     avrgDs = avrgDs(vertIdx,vertIdx);
%     par    = parcel(vertIdx,h);
%     % END CHANGE 
% 
%     [row,col,avrgD]=find(avrgDs);
%     inSp = sub2ind(size(avrgDs),row,col);
%     sameReg=(bsxfun(@ne,par',par)+1); % 1 same region, 2 different region
%     sameReg=sameReg(inSp);
%     clear avrgDs par;
%     for ts = taskSet
%         switch condType
%             case 'unique'
%                 % if funcMap - only evaluate unique tasks in sc1 or sc2
%                 condIdx=find(D.overlap == 0 & D.StudyNum == ts); % get index for unique tasks
%             case 'all'
%                 condIdx=find(D.StudyNum == ts);
%         end
%         for s=sn
%             % CHANGE: This should be re-written to start from the wcon data
%             % Start 
%             A=gifti(fullfile(rootDir,'sc1','surfaceWB',subj_name{s},sprintf('%s.%s.wcon.%s.func.gii',subj_name{s},Hem{h},resolution)));
% 
%             % End CHANGE 
%             Data = A.cdata(vertIdx,condIdx); % Take the right subset
%             Data = bsxfun(@minus,Data,mean(Data,2));
%             Data = single(Data');
%             [K,P]=size(Data);
%             clear A;
% 
%             SD = sqrt(sum(Data.^2)/K);
% 
%             VAR = (SD'*SD);
%             COV = Data'*Data/K;
%             %COR = corrcoef(Data); % Try different way to calculate the corrlation coefficient
%             fprintf('%d',s);
%             for i=1:numBins
%                 for bw=[1 2] % 1-within, 2-between
%                     fprintf('.');
%                     %in = i+(bw-1)*numBins;
%                     in = 2*(i-1)+bw;
%                     inBin = avrgD>bins(i) & avrgD<=bins(i+1) & sameReg==bw;
%                     R.SN(in,1)      = s;
%                     R.hem(in,1)     = h;
%                     R.studyNum(in,1) = ts;
%                     R.N(in,1)       = sum(inBin(:));
%                     R.avrDist(in,1) = mean(avrgD(inBin));
%                     R.bwParcel(in,1)= bw-1;
%                     R.bin(in,1)     = i;
%                     R.distmin(in,1) = bins(i);
%                     R.distmax(in,1) = bins(i+1);
% 
%                     R.meanCOR(in,1) = full(nanmean( COV(inSp(inBin)) ./ sqrt(VAR(inSp(inBin))) ));
%                     R.meanCOV(in,1) = full(nanmean(COV(inSp(inBin))));
%                     R.meanVAR(in,1) = full(nanmean(VAR(inSp(inBin))));
%                     %R.meanCOR(in,1) = full(nanmean(COR(inSp(inBin)))); % Add the mean corrcoef in this bin
%                 end
%                 %%%%% Changes to eliminate the positive bias %%%%%
%                 miniBin=(bins(i+1)-bins(i))/rate;
%                 corr_W=[];
%                 corr_B=[];
%                 ratio_W=[];
%                 ratio_B=[];
%                 for j=1:rate
%                     inMiniBin_W = avrgD>(bins(i)+(j-1)*miniBin) & avrgD<=(bins(i)+j*miniBin) & sameReg==1; % within
%                     inMiniBin_B = avrgD>(bins(i)+(j-1)*miniBin) & avrgD<=(bins(i)+j*miniBin) & sameReg==2; % between
%                     inBin_W = avrgD>bins(i) & avrgD<=bins(i+1) & sameReg==1;
%                     inBin_B = avrgD>bins(i) & avrgD<=bins(i+1) & sameReg==2;
%                     N_W = sum(inBin_W(:));
%                     N_B = sum(inBin_B(:));
%                     this_N = sum(inMiniBin_W(:))+sum(inMiniBin_B(:));
% 
%                     %this_w_ratio = (sum(inMiniBin_B(:))/N_B)/((sum(inMiniBin_B(:))/N_B + sum(inMiniBin_W(:))/N_W)/2);
%                     %this_b_ratio = (sum(inMiniBin_W(:))/N_W)/((sum(inMiniBin_B(:))/N_B + sum(inMiniBin_W(:))/N_W)/2);
% 
%                     this_w_ratio = 1/((sum(inMiniBin_W(:))/N_W)/((sum(inMiniBin_B(:))/N_B + sum(inMiniBin_W(:))/N_W)/2));
%                     this_b_ratio = 1/((sum(inMiniBin_B(:))/N_B)/((sum(inMiniBin_B(:))/N_B + sum(inMiniBin_W(:))/N_W)/2));
% 
%                     this_corrW = this_w_ratio*full(nanmean( COV(inSp(inMiniBin_W)) ./ sqrt(VAR(inSp(inMiniBin_W))) ));
%                     this_corrB = this_b_ratio*full(nanmean( COV(inSp(inMiniBin_B)) ./ sqrt(VAR(inSp(inMiniBin_B))) ));
%                     corr_W = [corr_W this_corrW];
%                     corr_B = [corr_B this_corrB];
%                     ratio_W = [ratio_W this_w_ratio];
%                     ratio_B = [ratio_B this_b_ratio];
%                 end
%                 R.adjustMeanCOR(2*(i-1)+1,1) = nansum(corr_W)/nansum(ratio_W);
%                 R.adjustMeanCOR(2*(i-1)+2,1) = nansum(corr_B)/nansum(ratio_B);
%             end
%             clear VAR COV;
% 
%             R.corr=R.meanCOV./sqrt(R.meanVAR);
%             fn = fieldnames(R);
% 
%             for idx = 1:numel(fn)
%                 fni = string(fn(idx));
%                 field = R.(fni);
%                 odd_sub = field(1:2:end,:);
%                 even_sub = field(2:2:end,:);
%                 R.(fni) = [odd_sub; even_sub];
%                 clear odd_sub even_sub
%             end
% 
%             fprintf('\n');
%             RR = addstruct(RR,R);
%             clear R;
%         end
%     end
% end
% varargout={RR};



 