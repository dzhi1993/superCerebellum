%%%%%% Evaluation of group-level parcellations by Silhoutte coefficient and homogeneity
baseDir    = 'D:/data';
rootDir    = 'D:/data';
wbDir      = fullfile(baseDir,'sc1','surfaceWB');
fsDir      = fullfile(baseDir,'sc1','surfaceFreesurfer');
atlasDir   = 'D:\data\Atlas_templates\standard_mesh';
anatomicalDir = fullfile(baseDir,'sc1','anatomicals');
regDir     = fullfile(baseDir,'sc1','RegionOfInterest');
resDir     =  'D:\python_workspace\brain_parcellation_project\agglomerative clustering\cortex_clustering_results';
resDir2    =  'D:\python_workspace\brain_parcellation_project\agglomerative clustering\subject_clustering_results';
glmDir     = 'GLM_firstlevel_4';
studyDir  = {'sc1','sc2'};
Hem       = {'L','R'};
hemname   = {'CortexLeft','CortexRight'};
fshem     = {'lh','rh'};
subj_name = {'s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11',...
    's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23','s24',...
    's25','s26','s27','s28','s29','s30','s31'};
returnSubjs=[2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31];

% toPlot = {'Glasser_2016','Yeo_2015','Yeo_17','Yeo_7','Power2011','Gordon',...
%           'Fan_105','Arslan_1_25','Baldassano','Shen','ICA','AAL_41','Desikan','Dextrieux'};    
% toPlot = {'Schaefer2018_7Networks_100','Schaefer2018_7Networks_200','Schaefer2018_7Networks_300',...
%           'Schaefer2018_7Networks_400'};
toPlot = {'Schaefer2018_7Networks_100','Schaefer2018_7Networks_200','Schaefer2018_7Networks_300',...
          'Schaefer2018_7Networks_400','Schaefer2018_7Networks_500','Schaefer2018_7Networks_600',...
          'Schaefer2018_7Networks_700','Schaefer2018_7Networks_800','Schaefer2018_7Networks_900',...
          'Schaefer2018_7Networks_1000'}; % red
Icos = {'Icosahedron-42','Icosahedron-162','Icosahedron-362','Icosahedron-642','Icosahedron-1002'};
      
mixed = {'Glasser_2016'}; % blue
task = {'Yeo_2015'}; % green
resting = {'Yeo_17','Yeo_7','Power2011','Schaefer2018_7Networks_200','Gordon','Fan_105','Arslan_1_25','Baldassano','Shen','ICA'}; % [1 0.68 0]
anatomical = {'AAL_41','Desikan','Dextrieux'}; % [0.894 0 0.906]

x=[];
m=[];
sd=[];
%toPlot = anatomical;

%%%%%%% Load homogeneity 
% for k=1:length(toPlot)
%     for h=1:2
%         %load parcellation for this hemisphere
%         par=gifti(sprintf('%s.32k.%s.label.gii',toPlot{k},Hem{h}));
%         parcel(:,h)=par.cdata;
%     end
%     
%     parcel(isnan(parcel))=0;
%     numCluster = size(unique(parcel(:,1)),1)+size(unique(parcel(:,2)),1)-2;
%     load(sprintf('homogeneity_%s.mat',toPlot{k}));
%     x = [x numCluster];
%     m = [m G];
%     sd = [sd std(G)];
% end
% errorbar(x,m,sd);
% mixed.Marker = '.';
% mixed.MarkerSize = 10;
% mixed.Color = 'blue';

%%%%%%% Load Silhoutte coefficient 
for k=1:length(toPlot)
    for h=1:2
        %load parcellation for this hemisphere
        par=gifti(sprintf('%s.32k.%s.label.gii',toPlot{k},Hem{h}));
        parcel(:,h)=par.cdata;
    end
    
    parcel(isnan(parcel))=0;
    numCluster = size(unique(parcel(:,1)),1)+size(unique(parcel(:,2)),1)-2;
    load(sprintf('SC_%s.mat',toPlot{k}));
    x = [x numCluster];
    G = mean(G,2);
    m = [m G];
    sd = [sd std(G)];
end
toPlot = errorbar(x,m,sd);
toPlot.Marker = '.';
toPlot.MarkerSize = 10;
toPlot.Color = 'blue';

%%%%%% Homogeneity simulation
% H=[];
% for k=1:length(Icos)
%     fprintf('%s',Icos{k});
%     % reset global homogeneity for 24 subjects
%     global_homo = [];
%     for i=1:100
%         % reset global homogeneity
%         %global_homo = [];
%         fprintf('.');
%         %load parcellation for this hemisphere
%         par=gifti(sprintf('%s.32k.%s.label.gii',Icos{k},Hem{1}));
%         parcel=par.cdata;
%         parcel(isnan(parcel))=0;
% 
%         outfile='tmp.func.gii';
%         smoothfile='smooth_tmp.func.gii';
%         random = [];
%         for num_cons = 1:61
%             random = [random normrnd(0,1,[32492,1])];
%         end
%         G=surf_makeFuncGifti(single(random));
%         save(G,outfile);
%         com = sprintf('wb_command -metric-smoothing %s tmp.func.gii %d %s -fix-zeros','fs_LR.32k.L.sphere.surf.gii',12,smoothfile);
%         system(com);
%         delete('tmp.func.gii');
% 
%         A = gifti(smoothfile);
%         Data = A.cdata;
%         delete('smooth_tmp.func.gii'); 
% 
%         Data = bsxfun(@minus,Data,mean(Data,2));
%         Data = single(Data);
%         CORR = corr(Data');
% 
%         clear Data;
%         homo = homogeneity(parcel, CORR);
%         global_homo = [global_homo; nanmean(homo)];
%         clear CORR;
%     end
%     fprintf('\n');
%     H=[H global_homo];
% end
% save(sprintf('homogeneity_Icos_%s.mat',Hem{1}), 'H');


%%%%%% Silhouette coefficient simulation
S = [];
for k=1:length(Icos)
    
    % reset global homogeneity for 24 subjects
    fprintf('Evaluating %s \n',Icos{k});
    global_SC = [];
    for h=1:1
        %load parcellation for this hemisphere
        par=gifti(sprintf('%s.32k.%s.label.gii',Icos{k},Hem{h}));
        parcel=par.cdata;
        parcel(isnan(parcel))=0;
        mw_idx = find(parcel==0);

        surf = gifti(sprintf('fs_LR.32k.%s.sphere.surf.gii',Hem{h}));
        adj = compute_vertex_nhood(surf.vertices, surf.faces);
        
        % Making adjacency parcellation matrix
        neigh = zeros(max(parcel));
        for c=1:max(parcel)
            [row, col] = find(adj(parcel==c,:));
            neigh_par = unique(parcel(unique(col)));
            neigh_par = neigh_par(neigh_par~=c & neigh_par~=0);
            neigh(neigh_par,c) = 1;
        end
        
        for i=1:100
            fprintf('.');
        
            outfile='tmp.func.gii';
            smoothfile='smooth_tmp.func.gii';
            random = [];
            for num_cons = 1:61
                random = [random normrnd(0,1,[32492,1])];
            end
            G=surf_makeFuncGifti(single(random));
            save(G,outfile);
            com = sprintf('wb_command -metric-smoothing %s tmp.func.gii %d %s -fix-zeros','fs_LR.32k.L.sphere.surf.gii',12,smoothfile);
            system(com);
            delete('tmp.func.gii');

            A = gifti(smoothfile);
            Data = A.cdata;
            delete('smooth_tmp.func.gii'); 

            Data = bsxfun(@minus,Data,mean(Data,2));
            Data = single(Data);
            CORR = corr(Data');

            clear Data;

            %remove vertices that have nan Data
%             [rows, columns] = find(isnan(Data));
%             parcel(unique(rows))=0;

            SC = silhouette_coef(parcel, 1-CORR, neigh);
            SC(mw_idx) = nan;
            SC = nanmean(SC);
            global_SC = [global_SC; SC];
            clear rows columns Data CORR;
        end   
    end
    S = [S global_SC];
    fprintf('\n');
end
save('SC_Icosahedrons.mat', 'S');

%%%%%% Homogeneity
% for k=1:length(toPlot)
%     % reset global homogeneity for 24 subjects
%     G = [];
%     for s=returnSubjs
%         % reset global homogeneity
%         global_homo = [];
%         for h=1:2
%             %load parcellation for this hemisphere
%             par=gifti(sprintf('%s.32k.%s.label.gii',toPlot{k},Hem{h}));
%             parcel=par.cdata;
%             parcel(isnan(parcel))=0;
% 
%             A=gifti(fullfile(wbDir,subj_name{s},sprintf('%s.%s.%s.con.%s.func.gii',subj_name{s},Hem{h},'sc1','32k')));
%             Data = [A.cdata(:,2:end-1) zeros(size(A.cdata,1),1)]; % bRemove intrstuction and add rest 
%             Data = bsxfun(@rdivide,Data,sqrt(A.cdata(:,end)));      % Noise normalize 
% 
%             Data = bsxfun(@minus,Data,mean(Data,2));
%             Data = single(Data);
%             CORR = corr(Data');
% 
%             clear Data;
%             homo = homogeneity(parcel, CORR);
%             global_homo = [global_homo; homo];
%             clear CORR;
%         end
%         aver = nanmean(global_homo);
%         G = [G; aver];
%     end
%     
%     save(sprintf('homogeneity_%s.mat',toPlot{k}), 'G');
% end


%%%%%% Silhouette coefficient
% for k=1:length(toPlot)
%     % reset global homogeneity for 24 subjects
%     G = [];
%     fprintf('Evaluating %s \n',toPlot{k});
%     for h=1:2
%         global_SC = [];
%         %load parcellation for this hemisphere
%         par=gifti(sprintf('%s.32k.%s.label.gii',toPlot{k},Hem{h}));
%         parcel=par.cdata;
%         parcel(isnan(parcel))=0;
%         mw_idx = find(parcel==0);
% 
%         surf = gifti(sprintf('fs_LR.32k.%s.sphere.surf.gii',Hem{h}));
%         adj = compute_vertex_nhood(surf.vertices, surf.faces);
%         
%         % Making adjacency parcellation matrix
%         neigh = zeros(max(parcel));
%         for c=1:max(parcel)
%             [row, col] = find(adj(parcel==c,:));
%             neigh_par = unique(parcel(unique(col)));
%             neigh_par = neigh_par(neigh_par~=c & neigh_par~=0);
%             neigh(neigh_par,c) = 1;
%         end
%         
%         for s=returnSubjs
%             fprintf('.');
%             % Load subject's true functional map and compute the correlations between vertices
%             A=gifti(fullfile(wbDir,subj_name{s},sprintf('%s.%s.%s.con.%s.func.gii',subj_name{s},Hem{h},'sc1','32k')));
%             Data = [A.cdata(:,2:end-1) zeros(size(A.cdata,1),1)]; % bRemove intrstuction and add rest 
%             Data = bsxfun(@rdivide,Data,sqrt(A.cdata(:,end)));      % Noise normalize 
%             Data = bsxfun(@minus,Data,mean(Data,2));
%             Data = single(Data);
%             CORR = corr(Data');
%             
%             %remove vertices that have nan Data
% %             [rows, columns] = find(isnan(Data));
% %             parcel(unique(rows))=0;
% 
%             SC = silhouette_coef(parcel, 1-CORR, neigh);
%             SC(mw_idx) = nan;
%             SC = nanmean(SC);
%             global_SC = [global_SC; SC];
%             clear rows columns Data CORR;
%         end
%         G = [G global_SC];
%         fprintf('\n');
%     end
%     
%     save(sprintf('SC_%s.mat',toPlot{k}), 'G');
% end



