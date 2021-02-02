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
wbDir           = fullfile(baseDir,'sc1','surfaceWB');
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
parDir          = 'D:\superCerebellum\group';
atlasDir        = 'D:\data\Atlas_templates\standard_mesh';
icosDir         = 'D:\matlab_workspace\fs_LR_32';

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
studyDir  = {'sc1','sc2'};
resolution = '32k';

% for h=1:1
%     for s=returnSubjs
%         % load individual distance file
%         load(fullfile(wbDir,subj_name{s},sprintf('distances.%s.mat',Hem{h})));
%         D(~isfinite(D)) = 0;
%         avrgDs = sparse(double(D));
%         save(fullfile(wbDir,subj_name{s},sprintf('distances_sp.%s.mat',Hem{h})),'avrgDs');
%         clear D avrgDs;
%     end
% end

%%%%%%%%%%%%%%%%% Making random functional map %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% conNum = [29, 32];
% for s=returnSubjs
%     for i=1:100
%         for ts=1:2
%             for h = 1:2
%                 outfile=fullfile(wbDir,subj_name{s},sprintf('%s.%s.%s.rcon_%d.%s.func.gii',subj_name{s},Hem{h},studyDir{ts},i,'32k'));
%                 random = normrnd(0,0.3,[32492,conNum(ts)]);
%                 G=surf_makeFuncGifti(single(random));
%                 save(G,outfile);
%                 %com = sprintf('wb_command -metric-smoothing Z:/data/super_cerebellum_new/sc1/surfaceWB/group32k/fs_LR.32k.%s.midthickness.surf.gii temp.func.gii %d %s -fix-zeros',Hem{h},kernel,oname);
%                 %system(com);
%             end
%         end
%     end
% end

%%%%%%%%%%%%%%%%%%%%%%%%% Simulation on sphere %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute geodesics distance between vertices on the surface 
% surf='Sphere.1k.L.surf.gii';
% G = gifti(surf);
% 
% % Fake medial wall
% %com = sprintf('wb_command -surface-geodesic-rois %s %s %s %s', surf, num2str(50), 'vertex_list.txt', 'roi.func.gii');
% 
% % Generate distance matrix
% outname = 'temp.func.gii';
% distGOD = [];
% for i = 1:length(G.vertices)
%     com = sprintf('wb_command -surface-geodesic-distance %s %s %s', surf, num2str(i-1), outname);
%     system(com);
%     dist = gifti(outname);
%     distGOD = [distGOD dist.cdata];
% end
% save('distSphere_test.mat','distGOD');

% %Making parcellation
% parcel = surf_icosahedron('D:/superCerebellum','freq',2,'res',32,'thres',0);
% DCBC=[];
% bin=2.5;
% %%%%%%% Do evaluation %%%%%%%%%%%%%%%%
% for k = 1:20
%     T=sc1_sc2_neocortical('Eval:DCBC_test','hem',1,'parcel',parcel,'condType','all','taskSet',1,'distFile','distSphere_sp','bins',[0:bin:65]);
%     save(sprintf('testEval_sphere_%d_%d.mat',k,bin),'-struct','T');
%     within = T.weightedCorr(T.bwParcel==0);
%     between = T.weightedCorr(T.bwParcel==1);
%     within = reshape(within,[13, 24]);
%     between = reshape(between,[13, 24]);
%     DCBC = [DCBC nanmean(within-between)'];
% end
% save('DCBC_simulation_1.mat','DCBC');

%%%%%%%%%%%%%%%%% Making random parcellations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sphere = gifti('fs_LR.32k.L.sphere.surf.gii');
% parcel = gifti('Icosahedron-1002.32k.L.label.gii');
% 
% parcels = parcel.cdata;
% mask = ones(size(parcels));
% mask(parcels==0) = 0;
% parcels = parcels(parcels~=0);
% rotation = [];
% % 
% for i = 1:100
%     parcel = surf_icosahedron('D:/superCerebellum','freq',10,'res',32,'thres',0);
%     %[nullModel, moved] = generate_null_model(parcels, sphere, mask);
%     rotation(:,i) = parcel; 
% end
% 
% % Loading medial wall
% M = gifti('Yeo_JNeurophysiol11_7Networks.32k.L.label.gii');
% rotation(M.cdata==0,:)=0;
% save('nullModel_rotated_1002_L.mat', 'rotation');


%%%%%%%%%%%%%%%%%%%% Eval %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%par = [42,162,362,642,1002];
par = 162;
%bins = [0.2, 1, 2.5, 5];
bins = 1;
%toEval = {'Glasser','Yeo17','Yeo7','Power2011','Yeo2015','Desikan','Dextrieux'};

toEval = {'Glasser_2016','Yeo_2015','Yeo_17','Yeo_7','Power2011','Schaefer2018_7Networks_200','Gordon',...
          'Fan_105','Arslan_1_25','Baldassano','Shen','ICA','AAL_41','Desikan','Dextrieux',...
          'Schaefer2018_7Networks_100','Schaefer2018_7Networks_300',...
          'Schaefer2018_7Networks_400','Schaefer2018_7Networks_500','Schaefer2018_7Networks_600',...
          'Schaefer2018_7Networks_700','Schaefer2018_7Networks_800','Schaefer2018_7Networks_900',...
          'Schaefer2018_7Networks_1000'}; % red

mixed = {'Glasser_2016','Fan_105','Yeo_2015'}; % blue
task = {'Yeo_2015'}; % green
resting = {'Yeo_17','Yeo_7','Power2011','Schaefer2018_7Networks_200','Gordon','Arslan_1_25','Baldassano','Shen','ICA'}; % [1 0.68 0]
anatomical = {'AAL_41','Desikan','Dextrieux'}; % [0.894 0 0.906]

% toEval = {'Schaefer2018_7Networks_100','Schaefer2018_7Networks_200','Schaefer2018_7Networks_300',...
%           'Schaefer2018_7Networks_400','Schaefer2018_7Networks_500','Schaefer2018_7Networks_600',...
%           'Schaefer2018_7Networks_700','Schaefer2018_7Networks_800','Schaefer2018_7Networks_900',...
%           'Schaefer2018_7Networks_1000'}; % red


% clusters=[];
% m=[];
% sd=[];
% %toEval = toEval;
% 
% DCBC=[];
% for k=1:length(toEval)
% 
%     T = load(sprintf('Eval_%s_Sphere_sc1sc2_unique.mat',toEval{k}));
%     
%     within = T.weightedCorr(T.bwParcel==0);
%     between = T.weightedCorr(T.bwParcel==1);
%     
%     sc_specific_within = reshape(within,[size(within,1)/4, 4]);
%     sc_specific_between = reshape(between,[size(between,1)/4, 4]);
%     
%     this_DCBC=[];
%     % 1-2 sc1 and sc2 for L hemisphere; 3-4 sc1 and sc2 for R hemisphere
%     for i=1:4
%         tmp = reshape(sc_specific_within(:,i),[35, 24]) - reshape(sc_specific_between(:,i),[35, 24]);    
%         this_DCBC = [this_DCBC; sum(tmp,'omitnan')'];
%     end
%     DCBC = [DCBC this_DCBC];
% end
% [x y] = size(DCBC);
% 
% DCBC=reshape(DCBC',[y,24,x/24]);
% mean_DCBC = nanmean(DCBC,3)';
% 
% m = mean(mean_DCBC);
% sd = std(mean_DCBC);
% anatomical = errorbar(clusters,m,sd);
% anatomical.Marker = '.';
% anatomical.MarkerSize = 10;
% anatomical.Color = [0.894 0 0.906];


% Mapping .annot label into fs_LR 32k space (Schaefer 2018)
cluster = [7,17];
hemis = {'lh','rh'};
% for p=1:length(cluster)
%     for i=1:10
%         sc1_sc2_neocortical('PARCEL:annot2labelgii',sprintf('Schaefer2018_%dParcels_%dNetworks_order',i*100, cluster(p)))
%         for h=1:2
%      
%             sc1_sc2_neocortical('PARCEL:fsaverage2FSLR',sprintf('%s.Schaefer2018_%dParcels_%dNetworks_order.label.gii',hemis{h},i*100,cluster(p)),sprintf('Schaefer2018_%dNetworks_%d.32k.%s.label.gii',cluster(p),i*100,Hem{h}),h,'32k');
%         end
%     end
% end

% evaluation of different number of clusters (Schaefer 2018)
% for p =1:length(cluster)
%     for i = 1:10
%         for h=1:2
%             A=gifti(sprintf('Schaefer2018_%dNetworks_%d.32k.%s.label.gii',cluster(p),i*100,Hem{h}));
%             parcel(:,h)=A.cdata;
%         end
%         T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'parcel',parcel,'condType','unique','bins',[0:1:35]);
%         save(sprintf('Eval_Schaefer2018_%dNetworks_%d_avrg_sc1sc2_unique.mat',cluster(p),i*100),'-struct','T');
%     end
% end

% % Making parcellations gii (group parcellations from Arslan paper)
% for h=1:2
% %     V = gifti(sprintf('Glasser_2016.32k.%s.label.gii',Hem{h}));
% %     vertIdx = find(V.cdata(:)==0);
%     for p =1:length(toEval)
% %         parcel = NaN(32492, 1);
% %         parcel(vertIdx,:) = 0;
%         A=gifti(fullfile(parDir,sprintf('%s.32k.%s.label.gii',toEval{p},Hem{h})));
%         parcel = A.cdata;
% %         
% %         for i = 1:length(A.parcels)
% %             first_non_NaN_index_of_X = find(isnan(parcel), 1);
% %             parcel(first_non_NaN_index_of_X,:) = A.parcels(i);
% %         end
%         outfilename = fullfile('group',sprintf('%s_%d_test.32k.%s.label.gii',toEval{p},length(unique(parcel))-1,Hem{h}));
%         G = surf_makeLabelGifti(parcel,'anatomicalStruct',hemname{h},'labelRGBA',[colorcube(length(unique(parcel))) ones(length(unique(parcel)),1)]);
%         save(G,outfilename);
%     end
% end

% Existing parcellations (group)
for p =1:length(toEval)
    for b = 1:length(bins)
        for h=1:2
            A=gifti(fullfile(parDir,sprintf('%s.32k.%s.label.gii',toEval{p},Hem{h})));
            parcel(:,h)=A.cdata;
        end
        T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'parcel',parcel,'distFile','distSphere_sp','bins',[0:bins(b):35]);
        save(sprintf('Eval_%s_Sphere_wbeta.mat',toEval{p}),'-struct','T');
    end
end
        
% Evaluation on Random parcellations with real functional map        
% for j = 1:length(par)
%     left = load(sprintf('nullModel_rotated_%d_L.mat', par(j)));
%     %right = load(sprintf('nullModel_rotated_%d_R.mat', par(j)));
%     %left = surf_icosahedron('D:/superCerebellum','freq',4,'res',32,'thres',0);
%     for b = 1:length(bins)
%         for i = 1:1
%             %parcel = [left.rotation(:, i) right.rotation(:, i)];
%             parcel = left.rotation(:,i);
%             %parcel = surf_icosahedron('D:/superCerebellum','freq',2,'res',32,'thres',0);
%             T=sc1_sc2_neocortical('Eval:DCBC','hem',1,'parcel',parcel,'condType','all','taskSet',1,'distFile','distSphere_sp','bins',[0:bins(b):35]);
%             %save(sprintf('Eval_Icosahedron_%d_Sphere_L_all_%d_%s.mat',par(j),i,string(bins(b))),'-struct','T');
%             save(sprintf('figure1_%d_Sphere_L_all_%d_%s.mat',par(j),i,string(bins(b))),'-struct','T');
%         end
%     end
% end

%%%% distance matrix
% distMat = zeros(256);
% N = size(distMat,1);
% for i=1:N
%     for j=1:N
%         distMat(i,j) = abs(i-j);
%     end
% end
% 
% random = normrnd(0,0.3,[32492,conNum(ts)]);

%%%% Simulation start here
% numParcel = 16;
% parcel=zeros(numParcel);
% for i = 1:numParcel
%     [row,col] = ind2sub([sqrt(numParcel) sqrt(numParcel)],i);
%     row_start = (row-1)*8+1;
%     row_end = row*8;
%     col_start = (col-1)*8+1;
%     col_end = col*8;
%     parcel(row_start:row_end,col_start:col_end)=i;
% end
% 
% for i = 1:numParcel
%     [row,col] = ind2sub([sqrt(numParcel) sqrt(numParcel)],i);
%     row_start = (row-1)*8+1;
%     row_end = row*8;
%     col_start = (col-1)*8+1;
%     col_end = col*8;
%     parcel_backup(row_start:row_end,col_start:col_end)=i+16;
% end
% 
% % shift = [2 2];
% % parcel = circshift(parcel,shift);
% % parcel_backup = circshift(parcel_backup,shift);
% % parcel(1:shift(1),1:shift(2))=parcel_backup(1:shift(1),1:shift(2));
% parcel = parcel(:);
% DCBC=[];
% for k = 1:100
%     T=sc1_sc2_neocortical('Eval:DCBC_test_square','hem',1,'parcel',parcel,'condType','all','taskSet',1,'distFile','distSphere_sp','bins',[0:1:10]);
%     save(sprintf('testEval_%d.mat',k),'-struct','T');
%     within = T.weightedCorr(T.bwParcel==0);
%     between = T.weightedCorr(T.bwParcel==1);
%     within = reshape(within,[10, 24]);
%     between = reshape(between,[10, 24]);
%     DCBC = [DCBC nanmean(within-between)'];
% end
% save('DCBC_simulation_1.mat','DCBC');

% ci_all=[];
% h_all=[];
% load('DCBC_simulation_1.mat')
% for i =1:100
%     [h,p,ci,stats] = ttest(DCBC(:,i),0,'Alpha',0.05);
%     ci_all = [ci_all ci];
%     h_all = [h_all h];
% end
% ci_all=ci_all';
% 
% sig=[11,12,23,28,45,48,50];

par = [42, 162, 362, 642, 1002];
%par = 642;
bins = [2.5, 35];
%bins = 2.5;
% 
% %Eval of Random functional map on ONE random parcellation       
% for j = 1:length(par)
%     left = gifti(sprintf('Icosahedron-%d.32k.L.label.gii', par(j)));
%     %left = surf_icosahedron('D:/superCerebellum','freq',2,'res',32,'thres',0);
%     for b = 1:length(bins)
%         DCBC=[];
% 
%         %parcel = [left.rotation(:, i) right.rotation(:, i)];
%         parcel = left.cdata;
%         T=sc1_sc2_neocortical('Eval:DCBC_1','hem',1,'parcel',parcel,'condType','all','taskSet',1,'distFile','distSphere_sp','bins',[0:bins(b):35]);
%         save(sprintf('Simulation_RandomMap_Icosahedron_%d_Sphere_L_all_%s.mat',par(j),string(bins(b))),'-struct','T');
%         %within = T.weightedCorr(T.bwParcel==0);
%         %between = T.weightedCorr(T.bwParcel==1);
%         %within = reshape(within,[35/bins(b), 100]);
%         %between = reshape(between,[35/bins(b), 100]);
%         %tmp = within-between;
%         %DCBC = [DCBC sum(tmp,'omitnan')'];
%         %save(sprintf('rand_DCBC_%d_%s.mat',par(j),string(bins(b))),'DCBC');
%     end
% end


% G_DCBC_uw = [];
% G_std_uw = [];
% 
% G_DCBC_w = [];
% G_std_w = [];
% 
% for b = 1:length(bins)
%     for j = 1:length(par)
%         T = load(fullfile(sprintf('Simulation_RandomMap_Icosahedron_%d_Sphere_L_all_%s.mat',par(j),string(bins(b)))));
%         within = T.weightedCorr(T.bwParcel==0);
%         between = T.weightedCorr(T.bwParcel==1);
%         within = reshape(within,[35/bins(b), 100]);
%         between = reshape(between,[35/bins(b), 100]);
%         tmp = within-between;
%         if size(tmp,1) ~= 1
%             DCBC = sum(tmp,'omitnan')';
%         else
%             DCBC = tmp';
%         end
%         G_DCBC_w = [G_DCBC_w DCBC];
%         G_std_w = [G_std_w std(DCBC)];
%         
%         within = T.corr(T.bwParcel==0);
%         between = T.corr(T.bwParcel==1);
%         within = reshape(within,[35/bins(b), 100]);
%         between = reshape(between,[35/bins(b), 100]);
%         tmp = within-between;
%         if size(tmp,1) ~= 1
%             DCBC = mean(tmp,'omitnan')';
%         else
%             DCBC = tmp';
%         end
%         G_DCBC_uw = [G_DCBC_uw DCBC];
%         G_std_uw = [G_std_uw std(DCBC)];
%     end
%     %figure
%     %boxplot(G_DCBC);
% end
% 
% data={G_DCBC_uw,G_DCBC_w};
% boxplotGroup(data, 'PrimaryLabels', {'uw', 'weighted'},'SecondaryLabels',{'0.1', '0.2', '0.5', '1', '2.5'}, 'InterGroupSpace', 1);
% boxplot(G_DCBC)


%%%%% t-test random parcellation
% ci_all=[];
% h_all=[];
% DCBC=sc1_sc2_neocortical('EVAL:plotSingle','toPlot','Icosahedron_42','condType','all');
% 
% % for i = 1:24
% %     [h,p,ci,stats] = ttest(DCBC(i,:),0,'Alpha',0.05);
% %     ci_all = [ci_all ci];
% %     h_all = [h_all h];
% % end
% 
% SD = std(DCBC(:));
% for i =1:100
% %    data = [];
% %     for j = 1:24
% %         y = datasample(DCBC(j,:),2);
% %         data = [data y];
% %     end
%     %[h,p,ci,stats] = ztest(data',0,SD,'Alpha',0.01);
%     [h,p,ci,stats] = ttest(DCBC(:,i),0,'Alpha',0.01);
%     ci_all = [ci_all ci];
%     h_all = [h_all h];
% end
% 
% ci_all=ci_all';
% N_in = ci_all(:,1)<=0 & ci_all(:,2)>=0;
% In_sample = DCBC(:,N_in);
% Out_sample = DCBC(:,N_in==0);
% sum(h_1_all)




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


