function varargout=sc1_sc2_neocortical(what,varargin)
% Analysis of cortical data
% This is modernized from the analysis in sc1_sc2_imana, in that it uses
% the new FS_LR template

baseDir    = 'D:/data';
rootDir    = 'Z:\data\super_cerebellum_new';
wbDir      = fullfile(baseDir,'sc1','surfaceWB');
fsDir      = fullfile(baseDir,'sc1','surfaceFreesurfer');
atlasDir   = '~/Data/Atlas_templates/standard_mesh';
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

switch(what)
    case 'SURF:recon_all'                    % STEP 2.2: Calls recon_all
        % STUDY 1 ONLY
        % Calls recon-all, which performs, all of the
        % FreeSurfer cortical reconstruction process
        % example: sc1_imana('surf_freesurfer',1)
        sn=varargin{1}; % subjNum
        
        for i=sn
            freesurfer_reconall(fullfile(baseDir,'sc1','surfaceFreesurfer'),subj_name{i},fullfile(anatomicalDir,subj_name{i},['anatomical.nii']));
        end
    case 'SURF:wb_resample'
        % This reslice from the individual surface into the the fs_lr standard mesh - This replaces
        % calls to freesurfer_registerXhem, freesurfer_mapicosahedron_xhem, & caret_importfreesurfer.
        % It requires connectome wb to be installed.
        sn=returnSubjs;
        vararginoptions(varargin,{'sn'});
        for i=sn
            fprintf('reslicing %d\n',i);
            surf_resliceFS2WB(subj_name{i},fsDir,wbDir,'resolution','32k');
        end;
    case 'SURF:map_beta'         % STEP 11.5: Map con / ResMS (.nii) onto surface (.gifti)
        sn    = returnSubjs;     % subjNum
        study = [1 2];
        glm  = 4;     % glmNum
        hemis = [1 2];
        resolution = '32k';
        D=dload(fullfile(baseDir,'sc1_sc2_taskConds_GLM.txt'));
        vararginoptions(varargin,{'sn','study','glm','what','hemis','resolution'});
        
        for s=sn
            for h=hemis
                surfDir = fullfile(wbDir, subj_name{s});
                white=fullfile(surfDir,sprintf('%s.%s.white.%s.surf.gii',subj_name{s},Hem{h},resolution));
                pial=fullfile(surfDir,sprintf('%s.%s.pial.%s.surf.gii',subj_name{s},Hem{h},resolution));
                C1=gifti(white);
                C2=gifti(pial);
                
                for st = study
                    glmDir =fullfile(baseDir,studyDir{st},sprintf('GLM_firstlevel_%d',glm),subj_name{s});
                    T=load(fullfile(glmDir,'SPM_info.mat'));
                    filenames={};
                    for i=1:length(T.run);
                        filenames{i} = fullfile(glmDir,sprintf('beta_%4.4d.nii',i));
                    end;
                    filenames{i+1} = fullfile(glmDir,'ResMS.nii');
                    outfile = fullfile(surfDir,sprintf('%s.%s.%s.beta.%s.func.gii',subj_name{s},Hem{h},studyDir{st},resolution));
                    
                    G=surf_vol2surf(C1.vertices,C2.vertices,filenames,'column_names',filenames,'anatomicalStruct',hemname{h});
                    save(G,outfile);
                    
                    fprintf('mapped %s %s %s \n',subj_name{s},Hem{h},studyDir{st});
                end;
            end;
        end
    case 'SURF:map_con'         % STEP 11.5: Map con / ResMS (.nii) onto surface (.gifti)
        sn    = returnSubjs;     % subjNum
        study = [1 2];
        glm  = 4;     % glmNum
        hemis = [1 2];
        resolution = '32k';
        D=dload(fullfile(baseDir,'sc1_sc2_taskConds_GLM.txt'));
        vararginoptions(varargin,{'sn','study','glm','what','hemis','resolution'});
        
        for s=sn
            for h=hemis
                surfDir = fullfile(wbDir, subj_name{s});
                white=fullfile(surfDir,sprintf('%s.%s.white.%s.surf.gii',subj_name{s},Hem{h},resolution));
                pial=fullfile(surfDir,sprintf('%s.%s.pial.%s.surf.gii',subj_name{s},Hem{h},resolution));
                C1=gifti(white);
                C2=gifti(pial);
                
                for st = study
                    glmDir =fullfile(baseDir,studyDir{st},sprintf('GLM_firstlevel_%d',glm),subj_name{s});
                    T=getrow(D,D.StudyNum==st);
                    filenames={};
                    for i=1:length(T.condNames);
                        filenames{i} = fullfile(glmDir,sprintf('con_%s-rest.nii',T.condNames{i}));
                    end;
                    filenames{i+1} = fullfile(glmDir,'ResMS.nii');
                    T.condNames{i+1}= 'ResMS';
                    outfile = fullfile(surfDir,sprintf('%s.%s.%s.con.%s.func.gii',subj_name{s},Hem{h},studyDir{st},resolution));
                    
                    G=surf_vol2surf(C1.vertices,C2.vertices,filenames,'column_names',T.condNames,'anatomicalStruct',hemname{h});
                    save(G,outfile);
                    
                    fprintf('mapped %s %s %s \n',subj_name{s},Hem{h},studyDir{st});
                end;
            end;
        end
    case 'SURF:map_con_sess'         % Map contrasts and sessions to surface, including ResMS (.nii) prewhitening
        sn    = returnSubjs;     % subjNum
        study = [1 2];
        glm  = 4;     % glmNum
        hemis = [1 2];
        resolution = '32k';
        D=dload(fullfile(baseDir,'sc1_sc2_taskConds_GLM.txt'));
        vararginoptions(varargin,{'sn','study','glm','what','hemis','resolution'});
        
        for s=sn
            for h=hemis
                surfDir = fullfile(wbDir, subj_name{s});
                white=fullfile(surfDir,sprintf('%s.%s.white.%s.surf.gii',subj_name{s},Hem{h},resolution));
                pial=fullfile(surfDir,sprintf('%s.%s.pial.%s.surf.gii',subj_name{s},Hem{h},resolution));
                C1=gifti(white);
                C2=gifti(pial);
                
                for st = study
                    T=getrow(D,D.StudyNum==st);
                    glmDir =fullfile(baseDir,studyDir{st},sprintf('GLM_firstlevel_%d',glm),subj_name{s});
                    S= load(fullfile(glmDir,'SPM_info.mat'));
                    for se=[1 2]
                        fprintf('SN:%s Hem:%s ST:%s SE:%d:',subj_name{s},Hem{h},studyDir{st},se);
                        filename=fullfile(glmDir,'ResMS.nii');
                        ResMS=surf_vol2surf(C1.vertices,C2.vertices,filename);
                        numCond =max(S.cond);
                        Data=zeros(size(ResMS.cdata,1),numCond+1);
                        for i=1:numCond
                            betaNum = find(S.cond==i & S.sess==se);
                            for j=1:length(betaNum)
                                filenames{j}=fullfile(glmDir,sprintf('beta_%04d.nii',betaNum(j)));
                            end
                            Beta=surf_vol2surf(C1.vertices,C2.vertices,filenames);
                            Data(:,i)=mean(Beta.cdata,2)./sqrt(ResMS.cdata);
                            fprintf('%d ',i);
                        end;
                        fprintf('\n');
                        Data(:,numCond+1)=0;
                        Data=bsxfun(@minus,Data,mean(Data,2));
                        G=surf_makeFuncGifti(Data,'anatomicalStruct',hemname{h},'columnNames',[T.condNames(2:end);'Rest']);
                        outfile = fullfile(surfDir,sprintf('%s.%s.%s.wcon.sess%d.%s.func.gii',subj_name{s},Hem{h},studyDir{st},se,resolution));
                        save(G,outfile);
                    end;
                end;
            end
        end
    case 'SURF:combine_con'    %  Does univariate noise normalization (combined) and then combines the contrast files by the common baseline
        sn    = returnSubjs;     % subjNum
        glm  = 4;     % glmNum
        hemis = [1 2];
        resolution = '32k';
        T=dload(fullfile(baseDir,'sc1_sc2_taskConds.txt'));
        vararginoptions(varargin,{'sn','glm','what','hemis','resolution'});
        for s=sn
            surfDir = fullfile(wbDir, subj_name{s});
            for h=hemis
                for st=[1 2]
                    G{st}=gifti(fullfile(surfDir,sprintf('%s.%s.%s.con.%s.func.gii',subj_name{s},Hem{h},studyDir{st},resolution)));
                    N=size(G{st}.cdata,1);
                    wcon{st} = [G{st}.cdata(:,2:end-1) zeros(N,1)]; % Add rest
                    Ts = getrow(T,T.StudyNum==st);
                    wcon{st}=bsxfun(@minus,wcon{st},mean(wcon{st}(:,Ts.overlap==1),2));
                end;
                resMS=(G{1}.cdata(:,end)+G{1}.cdata(:,end))/2;
                W = [wcon{1} wcon{2}];
                W = bsxfun(@rdivide,W,resMS); % Common noise normalization
                outfile = fullfile(surfDir,sprintf('%s.%s.wcon.%s.func.gii',subj_name{s},Hem{h},resolution));
                Go=surf_makeFuncGifti(W,'columnNames',T.condNames,'anatomicalStruct',hemname{h});
                save(Go,outfile);
                fprintf('combined %s %s \n',subj_name{s},Hem{h});
            end;
        end;
    case 'SURF:groupFiles'
        hemis=[1 2];
        sn = returnSubjs;
        cd(fullfile(wbDir,'group164k'));
        for h=hemis
            inputFiles = {};
            for s=1:length(sn)
                inputFiles{s}  = fullfile(wbDir, subj_name{sn(s)},sprintf('%s.%s.wcon.func.gii',subj_name{sn(s)},Hem{h}));
                columnName{s} = subj_name{sn(s)};
            end;
            groupfile=sprintf('group.wcon.%s.func.gii',Hem{h});
            outfilenamePattern=sprintf('wcon.%%s.%s.func.gii',Hem{h});
            surf_groupGiftis(inputFiles,'groupsummary',groupfile,'outcolnames',subj_name(sn),'outfilenamePattern',outfilenamePattern);
        end;
    case 'SURF:groupSmooth'
        kernel =3;
        cd(fullfile(wbDir,'group32k'));
        for h=1:2
            fname = sprintf('group.wcon.%s.func.gii',Hem{h});
            oname = sprintf('group.swcon.%s.func.gii',Hem{h});
            A=gifti(fname);
            A.cdata(isnan(sum(A.cdata,2)),:)=0;
            save(A,'temp.func.gii');
            com = sprintf('wb_command -metric-smoothing fs_LR.32k.%s.midthickness.surf.gii temp.func.gii %d group.swcon.%s.func.gii -fix-zeros',Hem{h},kernel,Hem{h});
            system(com);
        end;
        delete('temp.func.gii');
    case 'SURF:indvSmooth'
        kernel =3;
        for sn = returnSubjs
            cd(fullfile(rootDir,'sc1','surfaceWB',subj_name{sn}));
            for h=1:2
                fname = sprintf('%s.%s.wcon.32k.func.gii',subj_name{sn},Hem{h});
                oname = sprintf('%s.%s.swcon.32k.func.gii',subj_name{sn},Hem{h});
                A=gifti(fname);
                A.cdata(isnan(sum(A.cdata,2)),:)=0;
                save(A,'temp.func.gii');
                com = sprintf('wb_command -metric-smoothing Z:/data/super_cerebellum_new/sc1/surfaceWB/group32k/fs_LR.32k.%s.midthickness.surf.gii temp.func.gii %d %s -fix-zeros',Hem{h},kernel,oname);
                system(com);
            end
            delete('temp.func.gii');
        end
    case 'SURF:resample32k'  % Resample functional data from group164 to group32
        hemis=[1 2];
        sn = [2,3,4];
        sourceDir =fullfile(wbDir,'group164k');
        targetDir =fullfile(wbDir,'group32k');
        T=dload(fullfile(baseDir,'sc1_sc2_taskConds.txt'));
        
        for h=hemis
            % wb_command -metric-resample group164K/group.wcon.L.func.gii group164K/fs_LR.164k.L.sphere.surf.gii group32K/fs_LR.32k.L.sphere.surf.gii BARYCENTRIC group32K/group.wcon.L.func.gii
            % wb_command -metric-resample group164K/group.wcon.R.func.gii group164K/fs_LR.164k.R.sphere.surf.gii group32K/fs_LR.32k.R.sphere.surf.gii BARYCENTRIC group32K/group.wcon.R.func.gii
        end;
    case 'SURF:voxel2vertex' % Tansforms voxels-based data (from regions_cortex.mat) to vertices 
        % Data should be P(voxel) x K (conditions / values)
        sn   = [];
        hem  = [];
        data = [];
        numVert = 32492; 
        vararginoptions(varargin,{'data','hem','sn'});
        
        load(fullfile(rootDir,'sc1','RegionOfInterest','data',subj_name{sn},sprintf('regions_cortex.mat'))); % 'regions' are defined in 'ROI_define'
        
        % Figure out mapping from Nodes to voxels in region 
        N = length(R{hem}.linvoxidxs); 
        MAP = nan(size(R{hem}.location2linvoxindxs)); % Make a structure of vertices (without medial wall to indices in ROI structure)
        for i=1:N
            MAP(R{hem}.location2linvoxindxs==R{hem}.linvoxidxs(i))=i;
        end

        vData = nan(numVert,size(data,2)); % Make outout data structure 
        for i=1:length(R{hem}.location)
            %vData(R{hem}.location(i),:)=nanmean(data(rmmissing(MAP(i,:)),:),1);
            vData(R{hem}.location(i),:)=mode(data(rmmissing(MAP(i,:)),:)) + 1; % for parcels
        end
        varargout = {vData};
    case 'SURF:getAllData' % returns all cortical data - zero centered
        A=gifti(fullfile(wbDir,'group32k','Icosahedron-362.32k.L.label.gii')); % Determine medial wall
        indx = find(A.cdata>0);
        Data=[];
        for h=1:2
            C=gifti(fullfile(wbDir,'group32k',sprintf('group.wcon.%s.func.gii',Hem{h})));
            D.indx=indx;
            D.data = C.cdata(indx,:);
            D.hem = ones(size(D.data,1),1)*h;
            D.parcel = A.cdata(indx,1);
            Data=addstruct(Data,D);
        end;
        Data.data = bsxfun(@minus,Data.data ,mean(Data.data,2));
        Data = getrow(Data,~isnan(sum(Data.data,2)));
        varargout={Data};
    case 'SURF:randomSmoothCon'
        sn = returnSubjs;     % subjNum
        hem = [1 2];
        verticesNum = 32492;
        conNum = 61;
        mu = 0; % the default mean for generating random gaussian numbers
        sigma = 1; % the default sigma for generating random gaussian numbers
        R = [-2 2]; % the default range if type is uniformly
        kernel = 3;
        type = 'gaussian'; % default random map fits gaussian distribution
        resolution = '32k';
        vararginoptions(varargin,{'sn','mu','sigma','range','hem','verticesNum','conNum','type','resolution'});       
        
        for s=sn
            for h=hem
                outfile = fullfile(rootDir,'sc1','surfaceWB',subj_name{s},sprintf('%s.%s.srcon.%s.func.gii',subj_name{s},Hem{h},resolution));   
                switch type
                    case 'gaussian' % if the random numbers fit gaussian distribution
                        random = normrnd(mu,sigma,[verticesNum,conNum]);
                    case 'uniform' % if the random numbers are uniformly distributed
                        for i=1:conNum
                            col = min(R) + rand(verticesNum, 1)*range(R);
                            random(:,i) = col;
                        end
                end
                
                G = surf_makeFuncGifti(single(random));
                save(G,'temp.func.gii');
                com = sprintf('wb_command -metric-smoothing Z:/data/super_cerebellum_new/sc1/surfaceWB/group32k/fs_LR.32k.%s.midthickness.surf.gii temp.func.gii %d %s -fix-zeros',Hem{h},kernel,outfile);
                system(com);
            end
            delete('temp.func.gii');
        end
    case 'PARCEL:annot2labelgii' % Make an annotation file (freesurfer) into a Gifti
        filename = varargin{1};
        for i=1:2
            name = [fshem{i} '.' filename '.annot'];
            [v,label,colorT]=read_annotation(name);
            values=colorT.table(:,5);
            newlabel = zeros(size(label));
            for j=1:length(values)
                newlabel(label==values(j))=j-1;  % New label values starting from 0
            end;
            RGB = colorT.table(:,1:3)/256; % Scaled between 0 and 1
            G=surf_makeLabelGifti(newlabel,'anatomicalStruct',hemname{i},'labelNames',colorT.struct_names,'labelRGBA',[RGB ones(size(RGB,1),1)]);
            save(G,[fshem{i} '.' filename '.label.gii']);
        end;
    case 'PARCEL:fsaverage2FSLR' % Transform anything from fsaverage to fs_LR
        infilename = varargin{1}; % '~/Data/Atlas_templates/CorticalParcellations/Yeo2011/Yeo_JNeurophysiol11_FreeSurfer/fsaverage/label/lh.Yeo2011_17Networks_N1000.label.gii'
        outfilename= varargin{2}; % '~/Data/Atlas_templates/FS_LR_164/Yeo_JNeurophysiol11_17Networks.164k.L.label.gii';
        hem        = varargin{3};
        surf       = varargin{4}; % '164k','32k'
        inSphere   = fullfile(atlasDir,['fs_' Hem{hem}],['fsaverage.' Hem{hem} '.sphere.164k_fs_' Hem{hem} '.surf.gii']);
        outSphere  = fullfile(atlasDir,'resample_fsaverage',['fs_LR-deformed_to-fsaverage.' Hem{hem} '.sphere.' surf '_fs_LR.surf.gii']);
        
        % Convert surface to Gifti
        system(['wb_command -label-resample ' infilename ' ' inSphere ' ' outSphere ' ADAP_BARY_AREA ' outfilename ' -area-surfs ' inSphere ' ' outSphere]);
    case 'PARCEL:Yeo2015'        % Make the probabilistic model from Yeo 2015 into a parcellation
        hem=[1 2];
        for h=hem
            infilename = fullfile(wbDir,'group32k',sprintf('Yeo_CerCor2015_12Comp.32k.%s.func.gii',Hem{h}));
            outfilename = fullfile(wbDir,'group32k',sprintf('Yeo_CerCor2015_12Comp.32k.%s.label.gii',Hem{h}));
            A=gifti(infilename);
            [~,data]= max(A.cdata,[],2);
            data(sum(A.cdata,2)==0,1)=0;
            G = surf_makeLabelGifti(data,'anatomicalStruct',hemname{h},'labelRGBA',[colorcube(13) ones(13,1)]);
            save(G,outfilename);
        end;
    case 'PARCEL:Spec'        % Make the label.gii for Spectral clustering results
    % hem=[1 2];
    clusters = 7:25;
    taskSet = 2;
    vararginoptions(varargin,{'clusters'});
    %for s = returnSubjs
        for ts = taskSet
            for k = 1:length(clusters) % The number of clusters are range from 7 to 30
                infilename = fullfile(resDir,'group',sprintf('sc%d',ts),sprintf('ac_cosine_single_sc%d_%d.mat',ts,clusters(k)));
                outfilename1 = fullfile(resDir,'group',sprintf('sc%d',ts),'labelGii',sprintf('group_ac_%d.32k.sc%d.%s.label.gii',clusters(k),ts,Hem{1}));
                outfilename2 = fullfile(resDir,'group',sprintf('sc%d',ts),'labelGii',sprintf('group_ac_%d.32k.sc%d.%s.label.gii',clusters(k),ts,Hem{2}));
                load(infilename);
                G1 = surf_makeLabelGifti(parcels_L,'anatomicalStruct',hemname{1},'labelRGBA',[colorcube(clusters(k)+1) ones(clusters(k)+1,1)]);
                G2 = surf_makeLabelGifti(parcels_R,'anatomicalStruct',hemname{2},'labelRGBA',[colorcube(clusters(k)+1) ones(clusters(k)+1,1)]);
                save(G1,outfilename1);
                save(G2,outfilename2);
            end
        end
    %end
    case 'PARCEL:SpecIdv'        % Make the label.gii for Spectral clustering results
    % hem=[1 2];
    sn = returnSubjs;
    clusters = [];
    vararginoptions(varargin,{'clusters', 'sn'});
    for s = sn
        for k = 1:length(clusters) % The number of clusters are range from 7 to 30
            infilename = fullfile(resDir,subj_name{s},sprintf('spec_cosine_%d.mat',clusters(k)));
            outfilename1 = fullfile(resDir,subj_name{s},'labelGii',sprintf('Spec_%d.32k.%s.label.gii',clusters(k), Hem{1}));
            outfilename2 = fullfile(resDir,subj_name{s},'labelGii',sprintf('Spec_%d.32k.%s.label.gii',clusters(k), Hem{2}));
            load(infilename);
            G1 = surf_makeLabelGifti(parcels_L,'anatomicalStruct',hemname{1},'labelRGBA',[colorcube(clusters(k)+1) ones(clusters(k)+1,1)]);
            G2 = surf_makeLabelGifti(parcels_R,'anatomicalStruct',hemname{2},'labelRGBA',[colorcube(clusters(k)+1) ones(clusters(k)+1,1)]);
            save(G1,outfilename1);
            save(G2,outfilename2);
        end
    end
    case 'DCBC:computeDistances' % Compute individual Dijkstra distances between vertices
        sn=returnSubjs;
        hem = [1 2];
        resolution = '32k';
        maxradius = 40;
        
        vararginoptions(varargin,{'sn','hem','resolution','maxradius'});
        for s=sn
            for h=hem
                surfDir = fullfile(wbDir, subj_name{s});
                white=fullfile(surfDir,sprintf('%s.%s.white.%s.surf.gii',subj_name{s},Hem{h},resolution));
                pial=fullfile(surfDir,sprintf('%s.%s.pial.%s.surf.gii',subj_name{s},Hem{h},resolution));
                C1=gifti(white);
                C2=gifti(pial);
                vertices = double((C1.vertices' + C2.vertices' )/2); % Midgray surface
                faces    = double(C1.faces');
                numVert=size(vertices,2);
                n2f=surfing_nodeidxs2faceidxs(faces);
                D=inf([numVert numVert],'single');
                for i=1:numVert;
                    [subV, subF, subIndx,vidxs, fidxs]=surfing_subsurface(vertices, faces, i, maxradius, n2f); % construct correct sub surface
                    D(vidxs,i)=single(surfing_dijkstradist(subV,subF,subIndx,maxradius));
                    if (mod(i,100)==0)
                        fprintf('.');
                    end
                end;
                fprintf('\n');
                save(fullfile(surfDir,sprintf('distances.%s.mat',Hem{h})),'D','-v7.3');
            end;
        end;
    case 'DCBC:avrgdistances'    % Average individual distances
        sn=[2,3,4,6,8,9,10,12,14];
        hem = [1 2];
        resolution = '32k';
        vararginoptions(varargin,{'sn','hem','resolution','maxradius'});
        avrgD=single(zeros(32492,32492));
        N=length(sn);
        for h=hem
            for s=1:length(sn)
                fprintf('.');
                surfDir = fullfile(wbDir, subj_name{sn(s)});
                load(fullfile(surfDir,sprintf('distances.%s.mat',Hem{h})));
                D(isinf(D))=45;
                w=single((s-1)/N);
                avrgD = w.*avrgD+(1-w)*D;
            end;
            fprintf('\n');
            avrgDs=sparse(double(avrgD));
            save(fullfile(wbDir,'group32k',sprintf('distances_sp.%s.mat',Hem{h})),'avgrDs');
        end;
    case 'DCBC:sphericalDist'    % Quick fix: Compute distances on the Sphere
        hem = [1 2];
        resolution = '32k';
        A=gifti(fullfile(wbDir,'group32k','fs_LR.32k.L.sphere.surf.gii'));
        C = double(A.vertices');
        Dist = surfing_eucldist(C,C);
        Dist(Dist>50)=0;
        Dist=sparse(Dist);
        save(fullfile(wbDir,'group32k',sprintf('distanceSp_sp.mat')),'Dist');
    case 'Eval:DCBC'             % Get the DCBC evaluation
        sn=returnSubjs;
        hem = [1 2];
        resolution = '32k';
        taskSet = [1 2];
        condType = 'unique'; % Evaluate on all or only unique conditions?
        bins = [0:5:40];     % Spatial bins in mm
        parcel = [];         % N*2 matrix for both hemispheres
        RR=[];
        distFile = 'distAvrg_sp';
        icoRes = 2562;
        rate = 5;            % Indicate how many small bins in each of the bins
        vararginoptions(varargin,{'sn','hem','bins','parcel','condType','taskSet','resolution','distFile','icoRes','rate'});
        D=dload(fullfile(baseDir,'sc1_sc2_taskConds.txt'));
        numBins = numel(bins)-1;
        for h=hem
            load(fullfile(wbDir,'group32k',distFile));
            
            % Now find the pairs that we care about to safe memory 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % CHANGE: Exclude only medial wall! 
            % The node indices of medial wall are defined by taking the union of seven existing parcellations, including
            % 'Glasser','Yeo17','Yeo7','Power2011','Yeo2015','Desikan', and 'Dextrieux', which stored as external 
            % .mat files "medialWallIndex_%hem.mat" for both hems.
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % mw = load(fullfile(wbDir, sprintf('group%s', resolution), sprintf('medialWallIndex_%s.mat', Hem{h})));
            mw = gifti(fullfile(wbDir, sprintf('group%s', resolution), sprintf('Icosahedron-%d.%s.%s.label.gii',icoRes,resolution,Hem{h})));
            
            % Here, we use it to find the union with the label-0 of Icosahedron-(icoRes) as our final medial wall
            % vertIdx is the indices that we want (without medial wall)
            vertIdx = setdiff(1:size(avrgDs), find(mw.cdata(:,1)==0)); 
            avrgDs = avrgDs(vertIdx,vertIdx);
            par    = parcel(vertIdx,h);
            % END CHANGE 
            
            [row,col,avrgD]=find(avrgDs);
            inSp = sub2ind(size(avrgDs),row,col);
            sameReg=(bsxfun(@ne,par',par)+1); % 1 same region, 2 different region
            sameReg=sameReg(inSp);
            clear avrgDs par;
            for ts = taskSet
                switch condType
                    case 'unique'
                        % if funcMap - only evaluate unique tasks in sc1 or sc2
                        condIdx=find(D.overlap == 0 & D.StudyNum == ts); % get index for unique tasks
                    case 'all'
                        condIdx=find(D.StudyNum == ts);
                end
                for s=sn
                    % CHANGE: This should be re-written to start from the wcon data
                    % Start 
                    A=gifti(fullfile(rootDir,'sc1','surfaceWB',subj_name{s},sprintf('%s.%s.wcon.%s.func.gii',subj_name{s},Hem{h},resolution)));
                    
                    % End CHANGE 
                    Data = A.cdata(vertIdx,condIdx); % Take the right subset
                    Data = bsxfun(@minus,Data,mean(Data,2));
                    Data = single(Data');
                    [K,P]=size(Data);
                    clear A;
                    
                    SD = sqrt(sum(Data.^2)/K);
                    
                    VAR = (SD'*SD);
                    COV = Data'*Data/K;
                    %COR = corrcoef(Data); % Try different way to calculate the corrlation coefficient
                    fprintf('%d',s);
                    for i=1:numBins
                        for bw=[1 2] % 1-within, 2-between
                            fprintf('.');
                            %in = i+(bw-1)*numBins;
                            in = 2*(i-1)+bw;
                            inBin = avrgD>bins(i) & avrgD<=bins(i+1) & sameReg==bw;
                            R.SN(in,1)      = s;
                            R.hem(in,1)     = h;
                            R.studyNum(in,1) = ts;
                            R.N(in,1)       = sum(inBin(:));
                            R.avrDist(in,1) = mean(avrgD(inBin));
                            R.bwParcel(in,1)= bw-1;
                            R.bin(in,1)     = i;
                            R.distmin(in,1) = bins(i);
                            R.distmax(in,1) = bins(i+1);
                                        
                            R.meanCOR(in,1) = full(nanmean( COV(inSp(inBin)) ./ sqrt(VAR(inSp(inBin))) ));
                            R.meanCOV(in,1) = full(nanmean(COV(inSp(inBin))));
                            R.meanVAR(in,1) = full(nanmean(VAR(inSp(inBin))));
                            %R.meanCOR(in,1) = full(nanmean(COR(inSp(inBin)))); % Add the mean corrcoef in this bin
                        end
                        %%%%%%%%%%%%%%% Changes to eliminate the positive bias %%%%%%%%%%%%%%%
                        miniBin=(bins(i+1)-bins(i))/rate;
                        corr_W=[];
                        corr_B=[];        
                        ratio_W=[];
                        ratio_B=[];
                        for j=1:rate
                            inMiniBin_W = avrgD>(bins(i)+(j-1)*miniBin) & avrgD<=(bins(i)+j*miniBin) & sameReg==1; % within
                            inMiniBin_B = avrgD>(bins(i)+(j-1)*miniBin) & avrgD<=(bins(i)+j*miniBin) & sameReg==2; % between
                            inBin_W = avrgD>bins(i) & avrgD<=bins(i+1) & sameReg==1;
                            inBin_B = avrgD>bins(i) & avrgD<=bins(i+1) & sameReg==2;
                            N_W = sum(inBin_W(:));
                            N_B = sum(inBin_B(:));
                            %this_N = sum(inMiniBin_W(:))+sum(inMiniBin_B(:));
                            %this_w_ratio = (sum(inMiniBin_B(:))/N_B)/((sum(inMiniBin_B(:))/N_B + sum(inMiniBin_W(:))/N_W)/2);
                            %this_b_ratio = (sum(inMiniBin_W(:))/N_W)/((sum(inMiniBin_B(:))/N_B + sum(inMiniBin_W(:))/N_W)/2);
                            
                            this_w_ratio = 1/((sum(inMiniBin_W(:))/N_W)/((sum(inMiniBin_B(:))/N_B + sum(inMiniBin_W(:))/N_W)/2));
                            this_b_ratio = 1/((sum(inMiniBin_B(:))/N_B)/((sum(inMiniBin_B(:))/N_B + sum(inMiniBin_W(:))/N_W)/2));
                            
                            this_corrW = this_w_ratio*full(nanmean( COV(inSp(inMiniBin_W)) ./ sqrt(VAR(inSp(inMiniBin_W))) ));
                            this_corrB = this_b_ratio*full(nanmean( COV(inSp(inMiniBin_B)) ./ sqrt(VAR(inSp(inMiniBin_B))) ));
                            corr_W = [corr_W this_corrW];
                            corr_B = [corr_B this_corrB];
                            
                            ratio_W = [ratio_W this_w_ratio];
                            ratio_B = [ratio_B this_b_ratio];
                        end
                        R.adjustMeanCOR(2*(i-1)+1,1) = nansum(corr_W)/nansum(ratio_W);
                        R.adjustMeanCOR(2*(i-1)+2,1) = nansum(corr_B)/nansum(ratio_B);
                    end
                    clear VAR COV;
                    
                    R.corr=R.meanCOV./sqrt(R.meanVAR);
                    fn = fieldnames(R);
    
                    for idx = 1:numel(fn)
                        fni = string(fn(idx));
                        field = R.(fni);
                        odd_sub = field(1:2:end,:);
                        even_sub = field(2:2:end,:);
                        R.(fni) = [odd_sub; even_sub];
                        clear odd_sub even_sub
                    end
                    
                    fprintf('\n');
                    RR = addstruct(RR,R);
                    clear R;
                end
            end
        end
        varargout={RR};
    case 'EVAL:doEval'           % Recipe for producing the DCBC evaluation results
        clusters = 7:30;
        sn = returnSubjs;
        taskSet = [1 2];
        evalSet = [2 1];
        condType = 'unique';
        type = 'group';
        vararginoptions(varargin,{'clusters','condType','type','sn','taskSet'});
        
%         % Yeo 7
%         fprintf('Evaluating Yeo 7 ... \n');
%         for h=1:2
%             A=gifti(fullfile(rootDir,'sc1','surfaceWB','group32k',sprintf('Yeo_JNeurophysiol11_7Networks.32k.%s.label.gii',Hem{h})));
%             parcel(:,h)=A.cdata;
%         end
%         T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'parcel',parcel,'condType','unique','taskSet',1,'distFile','distSphere_sp');
%         save(sprintf('Eval_Yeo7_Sphere_sc1_%s.mat',condType),'-struct','T');
%         
%         % Yeo 17
%         fprintf('Evaluating Yeo 17 ... \n');
%         for h=1:2
%             A=gifti(fullfile(rootDir,'sc1','surfaceWB','group32k',sprintf('Yeo_JNeurophysiol11_17Networks.32k.%s.label.gii',Hem{h})));
%             parcel(:,h)=A.cdata;
%         end
%         T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'parcel',parcel,'condType','unique','taskSet',1,'distFile','distSphere_sp');
%         save(sprintf('Eval_Yeo17_Sphere_sc1_%s.mat',condType),'-struct','T');
%         
%         % Yeo 2015
%         fprintf('Evaluating Yeo 2015 ... \n');
%         for h=1:2
%             A=gifti(fullfile(rootDir,'sc1','surfaceWB','group32k',sprintf('Yeo_CerCor2015_12Comp.32k.%s.label.gii',Hem{h})));
%             parcel(:,h)=A.cdata;
%         end
%         T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'parcel',parcel,'condType','unique','taskSet',1,'distFile','distSphere_sp');
%         save(sprintf('Eval_Yeo2015_Sphere_sc1_%s.mat',condType),'-struct','T');
%         
%         % Glasser 2016
%         fprintf('Evaluating Glasser 2016 ... \n');
%         for h=1:2
%             A=gifti(fullfile(rootDir,'sc1','surfaceWB','group32k',sprintf('Glasser_2016.32k.%s.label.gii',Hem{h})));
%             parcel(:,h)=A.cdata;
%         end
%         parcel(isnan(parcel))=0;
%         T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'parcel',parcel,'condType','unique','taskSet',1,'distFile','distSphere_sp');
%         save(sprintf('Eval_Glasser_Sphere_sc1_%s.mat',condType),'-struct','T');
        
        
% %         for h=1:2
% %             A=gifti(sprintf('Icosahedron-42.32k.%s.label.gii',Hem{h}));
% %             parcel(:,h)=A.cdata;
% %         end;
% %         T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'parcel',parcel,'condType','all','distFile','distSphere_sp');
% %         save('Eval_Icosahedron42_Sphere_all.mat','-struct','T');
% % 
% %         for h=1:2
% %             A=gifti(sprintf('Icosahedron-162.32k.%s.label.gii',Hem{h}));
% %             parcel(:,h)=A.cdata;
% %         end;
% %         T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'parcel',parcel,'condType','all','distFile','distSphere_sp');
% %         save('Eval_Icosahedron162_Sphere_all.mat','-struct','T');
% % 
% %         for h=1:2
% %             A=gifti(sprintf('Icosahedron-362.32k.%s.label.gii',Hem{h}));
% %             parcel(:,h)=A.cdata;
% %         end;
% %         T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'parcel',parcel,'condType','all','distFile','distSphere_sp');
% %         save('Eval_Icosahedron362_Sphere_all.mat','-struct','T');
        
        % Power 2011
        fprintf('Evaluating Power 2011 ... \n');
        for h=1:2
            A=gifti(fullfile(rootDir,'sc1','surfaceWB','group32k',sprintf('Power2011.32k.%s.label.gii',Hem{h})));
            parcel(:,h)=A.cdata;
        end
        T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'parcel',parcel,'condType',condType,'distFile','distSphere_sp');
        save(sprintf('Eval_Power2011_Sphere_sc1sc2_%s_randomMap.mat',condType),'-struct','T');
%         
%         % Desikan
%         fprintf('Evaluating Desikan ... \n');
%         for h=1:2
%             A=gifti(fullfile(rootDir,'sc1','surfaceWB','group32k',sprintf('Desikan.32k.%s.label.gii',Hem{h})));
%             parcel(:,h)=A.cdata;
%         end
%         T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'parcel',parcel,'condType','unique','taskSet',1,'distFile','distSphere_sp');
%         save(sprintf('Eval_Desikan_Sphere_sc1_%s.mat',condType),'-struct','T');
%         
%         % Dextrieux
%         fprintf('Evaluating Dextrieux ... \n');
%         for h=1:2
%             A=gifti(fullfile(rootDir,'sc1','surfaceWB','group32k',sprintf('Dextrieux.32k.%s.label.gii',Hem{h})));
%             parcel(:,h)=A.cdata;
%         end;
%         T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'parcel',parcel,'condType','unique','taskSet',1,'distFile','distSphere_sp');
%         save(sprintf('Eval_Dextrieux_Sphere_sc1_%s.mat',condType),'-struct','T');
        
%         if strcmp(type,'group')
%             % group Spectral clustering 
%             fprintf('Start evaluating agglomerative clustering results ... \n');
%             for ts = taskSet
%                 for k=1:length(clusters)
%                     for h=1:2
%                         infile = fullfile(resDir,'group',sprintf('sc%d',ts),'labelGii', sprintf('group_ac_%d.32k.sc%d.%s.label.gii',clusters(k),ts,Hem{h}));
%                         A=gifti(infile);
%                         parcel(:,h)=A.cdata;
%                     end
%                     T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'parcel',parcel,'condType','unique','taskSet',evalSet(ts),'distFile','distSphere_sp');
%                     save(fullfile(resDir,'group',sprintf('sc%d',ts),'Eval',sprintf('Eval_ac_Sphere_sc%d_unique_%d.mat',ts,clusters(k))),'-struct','T');
%                 end
%             end
%         else
%             % individual Spectral clustering 
%             for ts = taskSet
%                 fprintf('Start evaluating experiment %d on experiment %d ... \n', ts, evalSet(ts));
%                 for s = sn
%                     % fprintf('Start evaluating subject %d ... \n', s);
%                     for k=1:length(clusters)
%                         fprintf('Evaluating subjuect %d, k=%d ... \n',s,clusters(k));
%                         for h=1:2
%                             infile = fullfile(resDir2,sprintf('sc%d',ts),subj_name{s},'labelGii',sprintf('%s_Spec_%d.32k.sc%d.%s.label.gii',subj_name{s},clusters(k),ts,Hem{h}));
%                             A=gifti(infile);
%                             parcel(:,h)=A.cdata;
%                         end
%                         % T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'sn',s,'parcel',parcel,'condType','all','distFile','distSphere_sp');
%                         T=sc1_sc2_neocortical('Eval:DCBC','hem',[1 2],'parcel',parcel,'sn',s,'condType','unique','taskSet',evalSet(ts),'distFile','distSphere_sp');
%                         save(fullfile(resDir2,sprintf('sc%d',ts),subj_name{s},'Eval',sprintf('Eval_Spec_Sphere_unique_%d.mat', clusters(k))),'-struct','T');
%                     end
%                 end
%             end
%         end
    case 'EVAL:plotSingle'       % Plots a single evaluation
        toPlot = 'Power2011';
        condType='all';
        clusters = 7;
        CAT.linecolor={'k','r'};
        CAT.linestyle={'-','-'};
        CAT.linewidth=2;
        CAT.markertype='none';
        CAT.errorcolor={'k','r'};
        vararginoptions(varargin,{'toPlot','condType','clusters'});
        
        if strcmp(toPlot,'Spec')
            T=load(fullfile(wbDir, 'group32k', sprintf('Eval_%s_Sphere_%s_%d.mat', toPlot,condType,clusters)));
        else
            T=load(['Eval_' toPlot '_Sphere_' condType '.mat']);
        end
        D=tapply(T,{'bin','SN','distmin','distmax','bwParcel'},{'corr'},{'avrDist'});
        D.binC = (D.distmin+D.distmax)/2;
        lineplot(D.binC,D.corr,'split',D.bwParcel,'CAT',CAT);
        set(gca,'XLim',[0 40],'YLim',[-0.01 0.165],'XTick',[5:5:35]);
        drawline(0,'dir','horz');
        set(gcf,'PaperPosition',[2 2 3 3.7]);
        wysiwyg;
        keyboard;
    case 'EVAL:plotEval'         % Comparision plot of different parcellations
        g1 = [0.5 0.5 0.5]; % Gray 1
        % toPlot={'Glasser','Yeo17','Yeo7','Power2011','Yeo2015','Desikan','Dextrieux','Icosahedron362', 'Spec'};
        %toPlot={'Yeo7','Yeo17','Power2011','Yeo2015','Glasser','AcGroup'};
        toPlot={'Icosahedron'};
        % toPlot={'SpecGroup'};
        sn=returnSubjs;
        clusters=17;
        taskSet=[1 2]; % Either 1 or 2, to specify which dataset used to train
        evalSet=[2 1];
        CAT.linecolor={'r','b','b','b','k','g','g',g1};
        CAT.linestyle={'-',':',':','--','-','-',':',':'};
        CAT.linewidth=2;
        CAT.markertype='none';
        CAT.errorcolor={'r','b','b','b','r','g','g',g1};
        condType='all';
        vararginoptions(varargin,{'toPlot','condType', 'clusters','sn','taskSet'});
        T=[];
        for i=1:length(toPlot)
            if strcmp(toPlot{i},'Spec')
                % D=load(fullfile(wbDir,'group32k', sprintf('Eval_%s_Sphere_%s_%d.mat',toPlot{i},condType, clusters)));
                % D=load(fullfile(resDir2,'sc1',subj_name{sn},'Eval',sprintf('Eval_%s_Sphere_unique_%d.mat',toPlot{i},clusters)));
                % D=load(fullfile(resDir,subj_name{sn},'Eval',sprintf('Eval_%s_Sphere_all_%d.mat',toPlot{i},clusters)));
            elseif strcmp(toPlot{i},'Icosahedron')
                D.SN=[];
                D.hem=[];
                D.studyNum=[];
                D.N=[];
                D.avrDist=[];
                D.bwParcel=[];
                D.bin=[];
                D.distmin=[];
                D.distmax=[];
                D.meanVAR=[];
                D.meanCOV=[];
                D.corr=[];
                for ts = taskSet
                    for s = sn
                        temp=load(fullfile(resDir2,sprintf('sc%d',ts),subj_name{s},'Eval',sprintf('Eval_Spec_Sphere_unique_%d.mat',clusters)));
                        D.SN=[D.SN;temp.SN];
                        D.hem=[D.hem;temp.hem];
                        D.studyNum=[D.studyNum;temp.studyNum];
                        D.N=[D.N;temp.N];
                        D.avrDist=[D.avrDist;temp.avrDist];
                        D.bwParcel=[D.bwParcel;temp.bwParcel];
                        D.bin=[D.bin;temp.bin];
                        D.distmin=[D.distmin;temp.distmin];
                        D.distmax=[D.distmax;temp.distmax];
                        D.meanVAR=[D.meanVAR;temp.meanVAR];
                        D.meanCOV=[D.meanCOV;temp.meanCOV];
                        D.corr=[D.corr;temp.corr];
                    end
                end
            elseif strcmp(toPlot{i},'AcGroup')
                D1=load(fullfile(resDir,'group',sprintf('sc%d',1),'Eval',sprintf('Eval_ac_Sphere_sc1_unique_%d.mat',clusters)));
                D2=load(fullfile(resDir,'group',sprintf('sc%d',2),'Eval',sprintf('Eval_ac_Sphere_sc2_unique_%d.mat',clusters)));
                D.SN=[D1.SN;D2.SN];
                D.hem=[D1.hem;D2.hem];
                D.studyNum=[D1.studyNum;D2.studyNum];
                D.N=[D1.N;D2.N];
                D.avrDist=[D1.avrDist;D2.avrDist];
                D.bwParcel=[D1.bwParcel;D2.bwParcel];
                D.bin=[D1.bin;D2.bin];
                D.distmin=[D1.distmin;D2.distmin];
                D.distmax=[D1.distmax;D2.distmax];
                D.meanVAR=[D1.meanVAR;D2.meanVAR];
                D.meanCOV=[D1.meanCOV;D2.meanCOV];
                D.corr=[D1.corr;D2.corr];
            else
                D1=load(fullfile(sprintf('Eval_%s_Sphere_sc%d_%s.mat',toPlot{i},1,condType)));
                D2=load(fullfile(sprintf('Eval_%s_Sphere_sc%d_%s.mat',toPlot{i},2,condType)));
                D.SN=[D1.SN;D2.SN];
                D.hem=[D1.hem;D2.hem];
                D.studyNum=[D1.studyNum;D2.studyNum];
                D.N=[D1.N;D2.N];
                D.avrDist=[D1.avrDist;D2.avrDist];
                D.bwParcel=[D1.bwParcel;D2.bwParcel];
                D.bin=[D1.bin;D2.bin];
                D.distmin=[D1.distmin;D2.distmin];
                D.distmax=[D1.distmax;D2.distmax];
                D.meanVAR=[D1.meanVAR;D2.meanVAR];
                D.meanCOV=[D1.meanCOV;D2.meanCOV];
                D.corr=[D1.corr;D2.corr];
            end
            TT=tapply(D,{'bin','SN','distmin','distmax'},{'corr','mean','name','corrB','subset',D.bwParcel==0},...
                {'corr','mean','name','corrW','subset',D.bwParcel==1});
            TT.parcel=ones(length(TT.SN),1)*i;
            T=addstruct(T,TT);
        end
        T.DCBC=T.corrB-T.corrW;
        T.binC = (T.distmin+T.distmax)/2;
        lineplot(T.binC,T.DCBC,'CAT',CAT,'split',T.parcel,'leg',toPlot);
        set(gca,'XLim',[0 40],'YLim',[-0.01 0.06],'XTick',[5:5:35]);
        set(gcf,'PaperPosition',[2 2 3.3 3]);
        wysiwyg;
    case 'EVAL:plotIcosahedron'         % Comparision plot of different parcellations
        g1 = [0.5 0.5 0.5]; % Gray 1
        %toPlot={'Icosahedron_42','Icosahedron_162','Icosahedron_362','Icosahedron_642','Icosahedron_1002'};
        toPlot={'Icosahedron_42'};
        CAT.linecolor={'r','b','b','b','g','k','k',g1};
        CAT.linestyle={'-','-',':','--','-','-',':','-'};
        CAT.linewidth=2;
        CAT.markertype='none';
        CAT.errorcolor={'r','b','b','b','g','k','k',g1};
        condType='all';
        vararginoptions(varargin,{'toPlot','condType'});
        T=[];
        for i=1:length(toPlot)
            D.SN=[];
            D.hem=[];
            D.studyNum=[];
            D.N=[];
            D.avrDist=[];
            D.bwParcel=[];
            D.bin=[];
            D.distmin=[];
            D.distmax=[];
            D.meanVAR=[];
            D.meanCOV=[];
            D.meanCOR=[];
            D.corr=[];
            D.adjustMeanCOR=[];
            for s = 1:8
                temp=load(sprintf('Eval_%s_Sphere_L_%s_%d_rate5.mat',toPlot{i},condType,s));
                D.SN=[D.SN;temp.SN];
                D.hem=[D.hem;temp.hem];
                D.studyNum=[D.studyNum;temp.studyNum];
                D.N=[D.N;temp.N];
                D.avrDist=[D.avrDist;temp.avrDist];
                D.bwParcel=[D.bwParcel;temp.bwParcel];
                D.bin=[D.bin;temp.bin];
                D.distmin=[D.distmin;temp.distmin];
                D.distmax=[D.distmax;temp.distmax];
                D.meanVAR=[D.meanVAR;temp.meanVAR];
                D.meanCOV=[D.meanCOV;temp.meanCOV];
                D.meanCOR=[D.meanCOR;temp.meanCOR];
                D.corr=[D.corr;temp.corr];
                D.adjustMeanCOR=[D.adjustMeanCOR;temp.adjustMeanCOR];
            end

            TT=tapply(D,{'bin','SN','distmin','distmax'},{'adjustMeanCOR','mean','name','corrW','subset',D.bwParcel==0},...
                {'adjustMeanCOR','mean','name','corrB','subset',D.bwParcel==1}); % Change the name of bwParcel; 0-within, 1-between
            TT.parcel=ones(length(TT.SN),1)*i;
            T=addstruct(T,TT);
        end
        T.DCBC=T.corrW-T.corrB; % corrB is within, corrW is between in old version
        T.binC = (T.distmin+T.distmax)/2;    
        lineplot(T.binC,T.DCBC,'CAT',CAT,'split',T.parcel,'errorcolor','b');
        
%         lineplot(T.binC,T.N_W,'CAT',CAT,'split',T.parcel,'linecolor','r','errorcolor','k');
%         hold on
%         lineplot(T.binC,T.N_B,'CAT',CAT,'split',T.parcel,'linecolor','b','errorcolor','k');
        
        set(gca,'XLim',[0 40],'YLim',[-0.01 0.06],'XTick', [5:5:35]);
        set(gcf,'PaperPosition',[2 2 3.3 3]);
        wysiwyg;
    case 'EVAL:plotSpatial'         % Comparision plot of different parcellations
        g1 = [0.5 0.5 0.5]; % Gray 1
        toPlot={'Icosahedron_42'};
        CAT.linecolor={'r','b','b','b','g','k','k',g1};
        CAT.linestyle={'-','-',':','--','-','-',':','-'};
        CAT.linewidth=2;
        CAT.markertype='none';
        CAT.errorcolor={'r','b','b','b','g','k','k',g1};
        condType='all';
        vararginoptions(varargin,{'toPlot','condType'});
        T=[];
        for i=1:length(toPlot)
            D=load(sprintf('Eval_%s_Sphere_%s.mat',toPlot{i},condType));
            TT=tapply(D,{'bin','SN','distmin','distmax'},{'adjustMeanCOR','mean','name','corrW','subset',D.bwParcel==0},...
                {'adjustMeanCOR','mean','name','corrB','subset',D.bwParcel==1}); % Change the name of bwParcel; 0-within, 1-between
            TT.parcel=ones(length(TT.SN),1)*i;
            T=addstruct(T,TT);
        end
        T.DCBC=T.corrB-T.corrW;
        T.binC = (T.distmin+T.distmax)/2;
        lineplot(T.binC,T.DCBC,'CAT',CAT,'split',T.parcel,'leg',toPlot);
        set(gca,'XLim',[0 40],'YLim',[-0.01 0.06],'XTick',[5:5:35]);
        set(gcf,'PaperPosition',[2 2 3.3 3]);
        wysiwyg;
    case 'ROI:defineROI'
        sn = returnSubjs;
        vararginoptions(varargin,{'sn'});
        for s=sn
            R=[];
            idx=1;
            mask = fullfile(baseDir,'sc1','GLM_firstlevel_4',subj_name{s},'mask.nii'); % mask from original GLM
            surfDir = fullfile(wbDir,'group32k');
            subjDir = fullfile(wbDir,subj_name{s});
            for h=1:2
                G = gifti(fullfile(surfDir,'Icosahedron-42.32k.L.label.gii'));
                R{h}.type     = 'surf_nodes'; % workbench version
                R{h}.location = find(G.cdata(:,1)>0);
                R{h}.white    = fullfile(subjDir,sprintf('%s.%s.white.32k.surf.gii',subj_name{s},Hem{h}));
                R{h}.pial     = fullfile(subjDir,sprintf('%s.%s.pial.32k.surf.gii',subj_name{s},Hem{h}));
                R{h}.linedef  = [5,0,1];
                R{h}.image    = mask;
                R{h}.name     = sprintf('cortex_%s.%s',subj_name{s},Hem{h});
            end
            R = region_calcregions(R);
            save(fullfile(regDir,'data',subj_name{s},'regions_cortex.mat'),'R');
            
            fprintf('\nHemisphereic region has been defined for %s \n',subj_name{s});
        end;
    case 'ROI:betas'  % extract betas for the ROI/study at hand. Results should be indentical to the
        % ROI:betas case in sc1_sc2_imana, but for speed reasons we do it
        % here directly from the beta_*.nii files, rather than going back
        % to the time series.
        sn=returnSubjs; % subjNum
        study=1; % studyNum
        glm='4'; % glmNum
        roi='cortex';  % 'cortical_lobes','whole_brain','yeo','desikan','cerebellum','yeo_cerebellum'
        
        vararginoptions(varargin,{'sn','study','glm','roi'});
        
        for s=sn
            glmDir = sprintf('GLM_firstlevel_%s',glm);
            glmDirSubj=fullfile(baseDir,studyDir{study},glmDir,subj_name{s});
            T=load(fullfile(glmDirSubj,'SPM_info.mat'));
            
            % load data
            load(fullfile(regDir,'data',subj_name{s},sprintf('regions_%s.mat',roi))); % 'regions' are defined in 'ROI_define'
            
            % Get the raw data files
            fname={};
            for i=1:length(T.SN)
                fname{i}=fullfile(glmDirSubj,sprintf('beta_%4.4d.nii',i));
            end;
            fname{end+1}=fullfile(glmDirSubj,'ResMS.nii');
            V=spm_vol(char(fname));
            
            D = region_getdata(V,R);  % Data is N x P
            
            for r=1:numel(R), % R is the output 'regions' structure from 'ROI_define'
                % Get betas (univariately prewhitened)
                B{r}.resMS   = D{r}(end,:);
                B{r}.betasNW = D{r}(1:end-1,:); % no noise normalisation
                B{r}.betasUW = bsxfun(@rdivide,B{r}.betasNW,sqrt(B{r}.resMS)); % univariate noise normalisation
                B{r}.region  = r;
                B{r}.regName = R{r}.name;
                B{r}.SN      = s;
            end
            
            % Save output for each subject
            outfile = sprintf('betas_%s.mat',roi);
            save(fullfile(baseDir,studyDir{study},'RegionOfInterest',sprintf('glm%s',glm),subj_name{s},outfile),'B');
            fprintf('betas computed and saved for %s for study%d \n',subj_name{s},study);
        end
    case 'CLUSTER:spectral'
        K = 10; % Number of clusters
        normalisation = 3;
        vararginoptions(varargin,{'K'});
        D=sc1_sc2_neocortical('SURF:getAllData');
        T=tapply(D,{'hem','parcel'},{'data'}); % Condense
        A=bsxfun(@rdivide,T.data,sqrt(sum(T.data.^2,2))); % Normalize
        Ang = 1-A*A';
        W=exp(-(Ang.^2)*3); % Gaussian affinity matrix
        cl = SpectralClustering(W,K,normalisation);
        [~,T.cl]=max(cl,[],2);
        for i=1:K
            Centroids(i,:)=mean(T.data(T.cl==i,:),1); % Get the centroids
        end;
        nodes=bsxfun(@rdivide,D.data,sqrt(sum(D.data.^2,2)));
        NodeAng=1-nodes*Centroids';
        [~,D.cl]=min(NodeAng,[],2);
        for h=1:2
            Data = zeros(32492,1);
            Data(D.indx(D.hem==h))=D.cl(D.hem==h);
            GC=surf_makeLabelGifti(Data,'anatomicalStruct',hemname{h},'labelRGBA',[zeros(1,4);[colorcube(K) ones(K,1)]]);
            save(GC,fullfile(wbDir,'group32k',sprintf('specCluster.%d.%s.label.gii',K,Hem{h})));
        end;
        con = [10 15 21];
        subplot(1,2,1);
        sc1_sc2_neocortical('CLUSTER:visualize',D.data,D.cl,Centroids,'threshold',0.9,'con',con);
        subplot(1,2,2);
        sc1_sc2_neocortical('CLUSTER:visualize',T.data,T.cl,Centroids,'threshold',0,'con',con);
        keyboard;
    case 'CLUSTER:visualize'
        X=varargin{1};
        cl=varargin{2};
        centroids=varargin{3};
        threshold=0.7; % Length threshold
        color_dots=1;
        plot_lines=1;
        K=max(cl);
        color=colorcube(K+1);
        con=[ 50    49    40]; % unidrnd(N,1,3); % [4 8 24]; %
        
        vararginoptions(varargin(4:end),{'threshold','plot_lines','color_dots','con'});
        
        L = sqrt(sum(X(:,:).^2,2));
        th  = quantile(L,threshold);
        D=dload(fullfile(baseDir,'sc1_sc2_taskConds.txt'));
        Y=X(:,con);
        for i=1:K
            if (color_dots==1)
                scatterplot3(X(:,con(1)),X(:,con(2)),X(:,con(3)),'markercolor',color(i,:),'markerfill',color(i,:),'subset',cl==i & L>th);
            elseif(color_dots==2)
                scatterplot3(X(:,con(1)),X(:,con(2)),X(:,con(3)),'markercolor',[0.6 0.6 0.6],'markerfill',[0.6 0.6 0.6],'subset',cl==i & L>th);
            end;
            hold on;
            if (plot_lines)
                a=quiver3(0,0,0,centroids(i,con(1)),centroids(i,con(2)),centroids(i,con(3)),0);
                set(a,'LineWidth',3,'Color',color(i,:));
            end;
        end;
        hold off;
        l=abs(minmax(minmax(X(:,con))));
        xlabel(D.condNames{con(1)});
        ylabel(D.condNames{con(2)});
        zlabel(D.condNames{con(3)});
        set(gca,'XLim',[-l l],'YLim',[-l l],'ZLim',[-l l]);
        axis equal;
        set(gcf,'PaperPosition',[2 2 8 6]);
        wysiwyg;
    case 'GRAD:compute'             % Computes local gradient
        cd(fullfile(wbDir,'group32k'));
        normalize=0; % Normalize each location before computing the gradient?
        vararginoptions(varargin,{'normalize'})
        for h=1:2
            sname = sprintf('fs_LR.32k.%s.midthickness.surf.gii',Hem{h});
            fname = sprintf('group.swcon.%s.func.gii',Hem{h});
            if (normalize)
                A = gifti(fname);
                X = A.cdata;
                X = bsxfun(@rdivide,X,sqrt(sum(X.^2,2)));
                B = surf_makeFuncGifti(X,'anatomicalStruct',hemname{h});
                fname   = sprintf('group.nwcon.%s.func.gii',Hem{h});
                save(B,fname);
            end
            oname = sprintf('group.grad.%s.func.gii',Hem{h});
            vname = sprintf('group.vec.%s.func.gii',Hem{h});
            com = sprintf('wb_command -metric-gradient %s %s %s -vectors %s',sname,fname,oname,vname);
            system(com);
        end;
    case 'GRAD:compare'
        cd(fullfile(wbDir,'group32k'));
        for h=1:2
            fname = sprintf('group.vec.%s.func.gii',Hem{h});
            oname = sprintf('group.mgrad.%s.func.gii',Hem{h});
            A=gifti(fname);
            [N,Q]=size(A.cdata);
            V=reshape(A.cdata,N,3,Q/3);
            A=sqrt(sum(V.^2,2));
            A=mean(A,3);
            V=permute(V,[3 2 1]);
            for i=1:size(V,3)
                B(i,1:2)=sqrt(eigs(double(V(:,:,i)'*V(:,:,i)),2));
                if (mod(i,1000)==0)
                    fprintf('.');
                end
            end;
            fprintf('\n');
            K=surf_makeFuncGifti([A B(:,1) B(:,1)-B(:,2) B(:,1)./sum(B,2)],'columnName',{'avrgGrad','sqEig1','sqEigD','sqEigN'},'anatomicalStruct',hemname{h});
            save(K,oname);
        end;
end;