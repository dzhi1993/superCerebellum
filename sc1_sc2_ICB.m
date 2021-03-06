function varargout=sc1_sc2_ICB(what,varargin)

% Directories
baseDir          = '/Users/maedbhking/Documents/Cerebellum_Cognition';
% baseDir            = '/Volumes/MotorControl/data/super_cerebellum_new';
% baseDir          = '/Users/jdiedrichsen/Data/super_cerebellum_new';

atlasDir='/Users/maedbhking/Documents/Atlas_templates/';

studyDir{1}     =fullfile(baseDir,'sc1');
studyDir{2}     =fullfile(baseDir,'sc2');
IBCDir          =fullfile(baseDir,'ibc');
studyStr        = {'SC1','SC2','SC12'};
behavDir        ='/data';
suitDir         ='/suit';
caretDir        ='/surfaceCaret';
regDir          ='/RegionOfInterest/';
encodeDir       ='/encoding';
contrastDir     ='/contrasts';
anatDir         ='/anatomical';

funcRunNum = [51,66];  % first and last behavioural run numbers (16 runs per subject)

run = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16'};

MDTB_subjs = {'s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11',...
    's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23','s24',...
    's25','s26','s27','s28','s29','s30','s31'};

MDTB_goodSubjs=[2,3,4,6,8,9,10,12,14,15,17:22,24:31];

IBC_subjs = {'s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s14'};
IBC_goodSubjs=[1,2,4:9,12:14];

loc_AC = {[-81,-127,-144],... %s01
    [-81,-120,-144],...       %s02
    [],...                    %s03
    [-79,-124,-155],...       %s04
    [-77,-123,-162],...       %s05
    [-81,-125,-156],...       %s06
    [-82,-126,-153],...       %s07
    [-80,-113,-161],...       %s08
    [-83,-120,-157],...       %s09
    [],...                    %s10
    [-83,-124,162],...        %s11
    [-82,-127,-158],...       %s12
    [-79,-129,-155],...       %s13
    [-80,-119,-159],...       %s14
    };

hem={'lh','rh'};
hemName={'LeftHem','RightHem'};


switch what
    
    case 'ANAT:reslice_LPI'                  % STEP 1.2: Reslice anatomical image within LPI coordinate systems
        % STUDY 1 ONLY
        sn  = varargin{1}; % subjNum
        % example: sc1_sc2_imana('ANAT:reslice_LPI',1)
        
        subjs=length(sn);
        
        for s=1:subjs,
            
            % (1) Reslice anatomical image to set it within LPI co-ordinate frames
            source  = fullfile(IBCDir,anatDir,IBC_subjs{sn(s)},['anatomical_raw','.nii']);
            dest    = fullfile(IBCDir,anatDir,IBC_subjs{sn(s)},['anatomical','.nii']);
            spmj_reslice_LPI(source,'name', dest);
            
            % (2) In the resliced image, set translation to zero
            V               = spm_vol(dest);
            dat             = spm_read_vols(V);
            V.mat(1:3,4)    = [0 0 0];
            spm_write_vol(V,dat);
            display 'Manually retrieve the location of the anterior commissure (x,y,z) before continuing'
        end
    case 'ANAT:centre_AC'                    % STEP 1.3: Re-centre AC
        % STUDY 1 ONLY
        % Set origin of anatomical to anterior commissure (must provide
        % coordinates in section (4)).
        % example: sc1_imana('ANAT:centre_AC',1)
        sn=varargin{1}; % subjNum
        
        subjs=length(sn);
        for s=1:subjs,
            img    = fullfile(IBCDir,anatDir,IBC_subjs{sn(s)},['anatomical','.nii']);
            V               = spm_vol(img);
            dat             = spm_read_vols(V);
            V.mat(1:3,4)    = loc_AC{sn(s)};
            spm_write_vol(V,dat);
            fprintf('Done for % \ns',IBC_subjs{sn(s)})
        end
        
    case 'SUIT:run_all'
        sn=varargin{1};
        
        sc1_sc2_ICB('SUIT:isolate_segment',sn)
        sc1_sc2_ICB('SUIT:make_maskImage',sn)
        sc1_sc2_ICB('SUIT:corr_cereb_cortex_mask',sn)
        sc1_sc2_ICB('SUIT:normalise_dartel',sn,'grey')
        sc1_sc2_ICB('SUIT:make_mask',sn,'grey')
        %         sc1_sc2_ICB('SUIT:reslice',sn)
    case 'SUIT:isolate_segment'              % STEP 9.2:Segment cerebellum into grey and white matter
        sn=varargin{1};
        %         spm fmri
        for s=sn,
            suitSubjDir = fullfile(IBCDir,suitDir,'anatomicals',IBC_subjs{s});dircheck(suitSubjDir);
            source=fullfile(IBCDir,anatDir,IBC_subjs{s},'anatomical.nii');
            dest=fullfile(suitSubjDir,'anatomical.nii');
            copyfile(source,dest);
            cd(fullfile(suitSubjDir));
            suit_isolate_seg({fullfile(suitSubjDir,'anatomical.nii')},'keeptempfiles',1);
        end
    case 'SUIT:make_maskImage'               % STEP 3.7:Make mask images (noskull and grey_only)
        % Make maskImage meanepi
        % example: sc1_sc2_imana('FUNC:make_maskImage',1)
        sn=varargin{1}; % subjNum
        
        for s=1:length(sn),
            
            % get example func image and mask
            funcImage=dir(fullfile(IBCDir,'contrasts',IBC_subjs{sn(s)})); % get example func image and mask.
            
            nam{1}  = fullfile(IBCDir,'contrasts',IBC_subjs{sn(s)}, funcImage(20).name); % contrast image
            spm_imcalc(nam, fullfile(IBCDir,'contrasts',IBC_subjs{sn(s)},'mask_gray.nii'), 'i1~=0')
        end
    case 'SUIT:corr_cereb_cortex_mask'       % STEP 9.4:
        sn=varargin{1};
        % STUDY 1
        
        subjs=length(sn);
        suitAnatDir=fullfile(IBCDir,'suit','anatomicals');
        
        for s=1:subjs,
            
            cortexGrey= fullfile(suitAnatDir,IBC_subjs{sn(s)},'c3anatomical.nii');
            cerebGrey = fullfile(suitAnatDir,IBC_subjs{sn(s)},'c1anatomical.nii');
            bufferVox = fullfile(suitAnatDir,IBC_subjs{sn(s)},'buffer_voxels.nii');
            
            % isolate overlapping voxels
            spm_imcalc({cortexGrey,cerebGrey},bufferVox,'(i1.*i2)')
            
            % mask buffer
            spm_imcalc({bufferVox},bufferVox,'i1>0')
            
            cerebGrey2 = fullfile(suitAnatDir,IBC_subjs{sn(s)},'cereb_prob_corr_grey.nii');
            cortexGrey2= fullfile(suitAnatDir,IBC_subjs{sn(s)},'cortical_mask_grey_corr.nii');
            
            % remove buffer from cerebellum
            spm_imcalc({cerebGrey,bufferVox},cerebGrey2,'i1-i2')
            
            % remove buffer from cortex
            spm_imcalc({cortexGrey,bufferVox},cortexGrey2,'i1-i2')
        end
    case 'SUIT:normalise_dartel'             % STEP 9.5: Normalise the cerebellum into the SUIT template.
        % STUDY 1
        % Normalise an individual cerebellum into the SUIT atlas template
        % Dartel normalises the tissue segmentation maps produced by suit_isolate
        % to the SUIT template
        % !! Make sure that you're choosing the correct isolation mask
        % (corr OR corr1 OR corr2 etc)!!
        % if you are running multiple subjs - change to 'job.subjND(s)."'
        % example: sc1_sc2_imana('SUIT:normalise_dartel',1,'grey')
        sn=varargin{1}; %subjNum
        type=varargin{2}; % 'grey' or 'whole' cerebellar mask
        
        for s=1:length(sn),
            cd(fullfile(IBCDir,'suit','anatomicals',IBC_subjs{sn(s)}));
            job.subjND.gray      = {'c_anatomical_seg1.nii'};
            job.subjND.white     = {'c_anatomical_seg2.nii'};
            switch type,
                case 'grey'
                    job.subjND.isolation= {'cereb_prob_corr_grey.nii'};
                case 'whole'
                    job.subjND.isolation= {'cereb_prob_corr.nii'};
            end
            suit_normalize_dartel(job);
        end
        
        % 'spm_dartel_warp' code was changed to look in the working
        % directory for 'u_a_anatomical_segment1.nii' file - previously it
        % was giving a 'file2mat' error because it mistakenly believed that
        % this file had been created
    case 'SUIT:make_mask'                    % STEP 9.7: Make cerebellar mask using SUIT
        sn=varargin{1}; % subjNum
        type=varargin{2}; % 'grey' or 'whole'
        
        subjs=length(sn);
        
        for s=1:subjs,
            mask = fullfile(IBCDir,'contrasts',IBC_subjs{sn(s)},'mask_gray.nii'); % mask for functional image
            switch type
                case 'grey'
                    suit  = fullfile(IBCDir,'suit','anatomicals',IBC_subjs{sn(s)},'cereb_prob_corr_grey.nii'); % cerebellar mask grey (corrected)
                    omask = fullfile(IBCDir,'suit','anatomicals',IBC_subjs{sn(s)},'maskbrainSUITGrey.nii'); % output mask image - grey matter
                case 'whole'
                    suit  = fullfile(IBCDir,'suit','anatomicals',IBC_subjs{sn(s)},'cereb_prob_corr.nii'); % cerebellar mask (corrected)
                    omask = fullfile(IBCDir,'suit','anatomicals',IBC_subjs{sn(s)},'maskbrainSUIT.nii'); % output mask image
            end
            cd(fullfile(IBCDir,'suit','anatomicals',IBC_subjs{sn(s)}));
            spm_imcalc({mask,suit},omask,'i1>0 & i2>0.7',{});
        end
    case 'SUIT:reslice'                      % STEP 9.8: Reslice the contrast images from first-level GLM
        % Reslices the functional data (betas, contrast images or ResMS)
        % from the first-level GLM using deformation from
        % 'suit_normalise_dartel'.
        % example: sc1_sc2_imana('SUIT:reslice',1,1,4,'betas','cereb_prob_corr_grey')
        % make sure that you reslice into 2mm^3 resolution
        sn=varargin{1}; % subjNum
        
        for s=1:length(sn),
            images='sess';
            source=dir(fullfile(IBCDir,'contrasts',IBC_subjs{sn(s)},sprintf('*%s*',images))); % images to be resliced
            job.subj.affineTr = {fullfile(IBCDir,'suit','anatomicals',IBC_subjs{sn(s)},'Affine_c_anatomical_seg1.mat')};
            job.subj.flowfield= {fullfile(IBCDir,'suit','anatomicals',IBC_subjs{sn(s)},'u_a_c_anatomical_seg1.nii')};
            job.subj.mask     = {fullfile(IBCDir,'suit','anatomicals',IBC_subjs{sn(s)},'cereb_prob_corr_grey.nii')};
            job.vox           = [2 2 2];
            cd(fullfile(IBCDir,'contrasts',IBC_subjs{sn(s)}))
            job.subj.resample = {source.name};
            dircheck(fullfile(IBCDir,'suit','contrasts',IBC_subjs{sn(s)}));
            %             cd(fullfile(IBCDir,'suit','contrasts',IBC_subjs{sn(s)}));
            suit_reslice_dartel(job);
            temp=dir('*wd*');
            for f=1:length(temp),
                movefile(fullfile(IBCDir,'contrasts',IBC_subjs{sn(s)},temp(f).name),fullfile(IBCDir,'suit','contrasts',IBC_subjs{sn(s)},temp(f).name));
            end
            
            fprintf('contrasts have been resliced into suit space for %s \n\n',IBC_subjs{sn(s)})
        end
        
    case 'PREP:ICB_info'
        sn=varargin{1};
        
        % get all possible task conditions
        idx=1;
        for s=1:length(sn),
            conName=dir(fullfile(IBCDir,'contrasts',IBC_subjs{sn(s)},'*sess*'));
            for c=1:length(conName),
                condName{idx,1}=conName(c).name(8:end-4);
                idx=idx+1;
            end
        end
        D.condNames=unique(condName);
        D.condNum=[1:length(D.condNames)]';
        
        % organise across subjs
        S=[];
        for s=1:length(sn),
            conName=dir(fullfile(IBCDir,'contrasts',IBC_subjs{sn(s)},'*sess*'));
            for c=1:length(conName),
                condName=conName(c).name(8:end-4);
                T.condNum(c,1)=D.condNum(strcmp(D.condNames,condName));
                T.condNames{c,1}=condName;
                tmp=str2double(conName(c).name(5:6));
                T.sessNum(c,1)=tmp;
            end
            T.SN=repmat(sn(s),length(T.condNames),1);
            S=addstruct(S,T);
            clear T conName tmp
        end
        
        varargout={S};
    case 'PREP:avrgMask_cereb'               % STEP 11.3:
        sn=varargin{1};
        step=varargin{2}; % 'reslice' or 'mask'
        % don't include s01 and s04 in the 'mask' step
        
        subjs=length(sn);
        
        switch step,
            case 'reslice'
                for s=1:subjs,
                    cd(fullfile(IBCDir,'suit','anatomicals',IBC_subjs{sn(s)}))
                    % normalise cerebellar grey into suit
                    job.subj.affineTr = {fullfile(IBCDir,'suit','anatomicals',IBC_subjs{sn(s)},'Affine_c_anatomical_seg1.mat')};
                    job.subj.flowfield= {fullfile(IBCDir,'suit','anatomicals',IBC_subjs{sn(s)},'u_a_c_anatomical_seg1.nii')};
                    job.subj.mask     = {fullfile(IBCDir,'suit','anatomicals',IBC_subjs{sn(s)},'cereb_prob_corr_grey.nii')};
                    job.vox           = [2 2 2];
                    job.subj.resample = {'c1anatomical.nii'};
                    suit_reslice_dartel(job);
                end
            case 'mask'
                
                for s=1:subjs,
                    nam{s}=fullfile(IBCDir,'suit','anatomicals',IBC_subjs{sn(s)},'wdc1anatomical.nii');
                end
                opt.dmtx = 1;
                cd(fullfile(IBCDir,'suit','anatomicals'));
                spm_imcalc(nam,'cerebellarGreySUIT.nii','mean(X)',opt);
                fprintf('averaged cerebellar grey mask in SUIT space has been computed \n')
        end
<<<<<<< HEAD
    case 'PREP:cereb:voxels_old'                 % STEP 11.6: Get UW cerebellar data (voxels)
=======
    case 'PREP:cereb:voxels'                 % STEP 11.6: Get UW cerebellar data (voxels)
>>>>>>> b59eb197fba4166de28734ba6da7640df273ff17
        sn=varargin{1};
        
        P=24076; % # of cerebellar voxels
        Q=117; % # of (unique) task contrasts
        
        % Load over all grey matter mask
        V=spm_vol(fullfile(IBCDir,'suit','anatomicals','cerebellarGreySUIT.nii'));
        
        % load PREP info
        T=sc1_sc2_ICB('PREP:ICB_info',IBC_goodSubjs);
        
        X=spm_read_vols(V);
        grey_threshold = 0.1; % grey matter threshold
        volIndx=find(X>grey_threshold);
        [i,j,k]= ind2sub(size(X),volIndx');
        
        S=[];
        for s=1:length(sn),
            
            Y=getrow(T,T.SN==sn(s));
            
            % load in normalised contrasts
            % univariately pre-whiten cerebellar voxels
            nam={};
            idx=1;
            for c=1:Q,
                condIdx=Y.condNum==c;
                numCon=numel(condIdx(condIdx~=0));
                for n=1:numCon,
                    sessIdx=Y.sessNum(condIdx);
                    CI=Y.condNames(condIdx);
                    nam{1}=fullfile(IBCDir,'suit','contrasts',IBC_subjs{sn(s)},sprintf('wdsess%2.2d-%s.nii',sessIdx(n),CI{n}));
                    Vi=spm_vol(nam{1});
                    C1(idx,:)=spm_sample_vol(Vi,i,j,k,0);
                    idx=idx+1;
                end
            end
            
            % make zero values nan
            C1(C1==0)=nan;
            
            % write out new structure ('Y_info')
            Y.data=C1;
            Y.nonZeroInd=repmat(volIndx',size(C1,1),1);
            
            outName=fullfile(IBCDir,'suit','contrasts','cereb_avrgDataStruct.mat');
            fprintf('cerebellar voxels computed for %s \n',IBC_subjs{sn(s)});
            clear C1
            
            S=addstruct(S,Y);
            clear Y
        end
        volIndx=volIndx';
        save(outName,'S','volIndx','V');
<<<<<<< HEAD

=======
        
>>>>>>> b59eb197fba4166de28734ba6da7640df273ff17
    case 'ACTIVITY:map2surf'
        sn=varargin{1}; % 'group' or <subjNum>
        
        % group or individual ?
        if ~strcmp(sn,'group'),
            outDir=fullfile(studyDir{2},caretDir,sprintf('x%s',IBC_subjs{sn}),'cerebellum'); dircheck(outDir)
        else
            outDir=fullfile(studyDir{2},caretDir,'suit_flat','glm4');
        end
        
        load(fullfile(IBCDir,'suit','contrasts','cereb_avrgDataStruct.mat'));
        
        SN=unique(S.SN);
        CN=unique(S.condNum);
        
        % set up volume info
        Yy=zeros(length(CN),length(SN),V.dim(1)*V.dim(2)*V.dim(3));
        C{1}.dim=V.dim;
        C{1}.mat=V.mat;
        
        % loop over subjs
        for s=1:length(SN),
            for c=1:length(CN),
                condNames{c}=char(unique(S.condNames(S.condNum==CN(c))));
                idx=S.SN==SN(s) & S.condNum==CN(c);
                B(c,s,:)=nanmean(S.data(idx,:),1);
            end
        end
        
        % subtract baseline - do we have a baseline here ??
        %         baseline=nanmean(B,1);
        %         B=bsxfun(@minus,B,baseline);
        
        % z score the activity patterns
        B=zscore(B);
        
        Yy(:,:,volIndx)=B;
        Yy=permute(Yy,[2 1 3]);
        
        indices=nanmean(Yy,1);
        indices=reshape(indices,[size(indices,2),size(indices,3)]);
        
        % map vol2surf
        indices=reshape(indices,[size(indices,1) V.dim(1),V.dim(2),V.dim(3)]);
        for i=1:size(indices,1),
            data=reshape(indices(i,:,:,:),[C{1}.dim]);
            C{i}.dat=data;
        end
        P=caret_suit_map2surf(C,'space','SUIT','stats','nanmean','column_names',condNames);  % MK created caret_suit_map2surf to allow for output to be used as input to caret_save
        
        % save out metric
        if strcmp(sn,'group'),
            outName='IBC_contrasts';
        else
            outName=sprintf('%s_IBC_contrasts',IBC_subjs{sn});
        end
        caret_save(fullfile(outDir,sprintf('%s.metric',outName)),P);
    case 'ACTIVITY:vol2surf'
        % this function takes any labelled volume (already in SUIT space)
        % and plots to the surface
        sn=varargin{1};
        inputMap=varargin{2}; % some options are 'Buckner_7Networks','SC1_9cluster','lob10', 'Cole_10Networks', 'SC2_90cluster' etc
        
        mapDir=fullfile(IBCDir,'suit','contrasts',IBC_subjs{sn},sprintf('wdsess%s.nii',inputMap));
        
        Vo=spm_vol(fullfile(mapDir));
        Vi=spm_read_vols(Vo);
        Vv{1}.dat=Vi;
        Vv{1}.dim=Vo.dim;
        Vv{1}.mat=Vo.mat;
        
        M=caret_suit_map2surf(Vv,'space','SUIT');
        
        suit_plotflatmap(M.data)
    case 'EVAL:ICB'% Evaluate group Map on IBC data
        sn=varargin{1}; % 'group' or <subjNum>
        mapType=varargin{2}; % options are 'lob10','lob26','Buckner_17Networks','Buckner_7Networks', 'Cole_10Networks','SC<studyNum>_<num>cluster'
        
        % load in func data to test (e.g. if map is sc1; func data should
        % be sc2)
        load(fullfile(IBCDir,'suit','contrasts','cereb_avrgDataStruct.mat'));
        T=S;
        
        % evaluating the group or the individual ?
        if strcmp(sn,'group'),
            % load in map
            mapName=fullfile(studyDir{2},encodeDir,'glm4',sprintf('groupEval_%s',mapType),'map.nii');
            outName=fullfile(studyDir{2},encodeDir,'glm4',sprintf('groupEval_%s',mapType),'spatialBoundfunc_ICB.mat');
            sn=unique(T.SN)';
        else
            mapName=fullfile(studyDir{2},encodeDir,'glm4',MDTB_subjs{sn},sprintf('map_%s.nii',mapType));
            outName=fullfile(studyDir{2},encodeDir,'glm4',MDTB_subjs{sn},sprintf('%s_spatialBoundfunc_ICB.mat',mapType));
        end
        
        % Now get the parcellation sampled into the same space
        [i,j,k]=ind2sub(V.dim,volIndx);
        [x,y,z]=spmj_affine_transform(i,j,k,V.mat);
        VA= spm_vol(mapName);
        [i1,j1,k1]=spmj_affine_transform(x,y,z,inv(VA.mat));
        Parcel = spm_sample_vol(VA,i1,j1,k1,0);
        % Divide the voxel pairs into all the spatial bins that we want
        fprintf('parcels\n');
        voxIn = Parcel>0;
        XYZ= [x;y;z];
        RR=[];
        [BIN,R]=mva_spatialCorrBin(XYZ(:,voxIn),'Parcel',Parcel(1,voxIn));
        clear XYZ i k l x y z i1 j1 k1 VA Parcel; % Free memory
        % Now calculate the estimation of the correlation for each subject
        for s=sn,
            D=T.data(find(T.SN==s),voxIn);
            fprintf('%d cross\n',s);
            R.SN = ones(length(R.N),1)*s;
            R.corr=mva_spatialCorr(D,BIN);
            R.crossval = zeros(length(R.corr),1);
            RR = addstruct(RR,R);
        end;
        save(outName,'-struct','RR');
    case 'EVAL:PLOT:CURVES'
        mapType=varargin{1}; % options are 'lob10','lob26','bucknerRest','SC<studyNum>_<num>cluster', or 'SC<studyNum>_POV<num>'
        
        vararginoptions({varargin{2:end}},{'CAT','sn'}); % option if doing individual map analysis
        
        T=load(fullfile(studyDir{2},encodeDir,'glm4',sprintf('groupEval_%s',mapType),'spatialBoundfunc_ICB.mat'));
        
        % distances are diff across evals so need to get dist per bin:
        for b=1:length(unique(T.bin)),
            dist=mode(round(T.dist(T.bin==b)));
            idx=find(T.bin==b);
            T.dist(idx,1)=dist;
        end

        if exist('CAT'),
            xyplot(T.dist,T.corr,T.dist,'split',T.bwParcel,'subset',T.crossval==0 & T.dist<=35,'CAT',CAT,'leg',{'within','between'},'leglocation','SouthEast');
        else
            xyplot(T.dist,T.corr,T.dist,'split',T.bwParcel,'subset',T.crossval==0 & T.dist<=35,'leg',{'within','between'},'leglocation','SouthEast');
        end
    case 'EVAL:STATS:CURVES'
        mapType=varargin{1}; % options are 'lob10','lob26','bucknerRest','SC<studyNum>_<num>cluster', or 'SC<studyNum>_POV<num>'
        crossval=0;
        
        T=load(fullfile(studyDir{2},encodeDir,'glm4',sprintf('groupEval_%s',mapType),'spatialBoundfunc_ICB.mat'));
        
        % do stats (over all bins) for group only
        C=getrow(T,T.crossval==crossval & T.dist<=35); % only crossval and dist<35
        S=tapply(C,{'bwParcel','SN'},{'corr'});
        fprintf('overall \n')
        ttest(S.corr(S.bwParcel==0), S.corr(S.bwParcel==1),2,'paired');
        
        % calculate effect size
        Group1=S.corr(S.bwParcel==0);
        Group2=S.corr(S.bwParcel==1);
        
        num=((Group1-1)*std(Group1)^2 + (Group2-1)*std(Group2)^2);
        denom=Group1+Group2-2;
        
        pooledSTD= sqrt(mean(num)/mean(denom));
        
        ES_pooled=(mean(Group1)-mean(Group2))/pooledSTD;
        
        fprintf('Effect size for within and between for %s is %2.2f when denom is pooled std  \n',mapType,ES_pooled);
        
        % summary stats
        x1=nanmean(S.corr(S.bwParcel==0));x2=nanmean(S.corr(S.bwParcel==1));
        SEM1=std(S.corr(S.bwParcel==0))/sqrt(length(T.SN));SEM2=std(S.bwParcel==1)/sqrt(length(T.SN));
        fprintf('average within corr is %2.2f; CI:%2.2f-%2.2f \n average between corr is %2.2f; CI:%2.2f-%2.2f \n',...
            nanmean(S.corr(S.bwParcel==0)),x1-(1.96*SEM1),x1+(1.96*SEM1),nanmean(S.corr(S.bwParcel==1)),...
            x2-(1.96*SEM2),x2+(1.96*SEM2));
        
    case 'AXES:group_curves' % make separate graphs for 'lob10','Buckner_7Networks','Buckner_17Networks','Cole_10Networks','SC12_10cluster'
        toPlot=varargin{1}; % 'SC12_10cluster'
        
        % Aesthetics
        CAT.markertype='none';
        CAT.errorwidth=.5;
        CAT.linecolor={'r','k'};
        CAT.errorcolor={'r','k'};
        CAT.linewidth={2, 2};
        CAT.linestyle={'-','-'};
        
        sc1_sc2_ICB('EVAL:PLOT:CURVES',toPlot,'CAT',CAT);
        
        % Labelling
        set(gca,'YLim',[0 0.3],'XLim',[0 35],'FontSize',14,'xtick',[0:5:35],'XTickLabel',{'0','','','','','','','35'}); %
        xlabel('Spatial Distances (mm)');
        ylabel('Activity Correlation (R)');
        %         title(plotName);
        set(gcf,'units','centimeters','position',[5,5,15,15])
        %         axis('auto')
        % do stats
        %         sc1_sc2_ICB('EVAL:STATS:CURVES',toPlot)
end


% Local functions
function dircheck(dir)
if ~exist(dir,'dir');
    warning('%s doesn''t exist. Creating one now. You''re welcome! \n',dir);
    mkdir(dir);
end