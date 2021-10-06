%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script for DCBC evaluation on volume space parcellations
%
% Author: Da Zhi, Oct. 5th, 2021
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
scDir           = 'D:\python_workspace\brain_parcellation_project\Spectral clustering';
resDir          = 'D:\superCerebellum\cross_validation';
glmDir          = 'D:\data\sc2\encoding\glm7';

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
resolution = 2; % in mm

toEval = {'choi_7','choi_17','parkes_3','pauli_3','pauli_5','pauli_7','pauli_17','tian_1','tian_2','tian_3','tian_4'};
bins = 5;

%% Compute the distance matrix by given mask file.%%
mask = niftiread(fullfile(glmDir,'spect','striatum_mask_2mm.nii'));
load(fullfile(glmDir,'striatum_avrgDataStruct.mat'),'volIndx');

[x, y, z] = ind2sub(size(mask),volIndx);
dist = zeros(length(volIndx), length(volIndx));
for i = 1:length(volIndx)
    for j = 1:length(volIndx)
        dist(i,j) = sqrt((x(i)-x(j))^2 + (y(i)-y(j))^2 + (z(i)-z(j))^2) * resolution;
    end
end

save('dist.mat','dist');

%% Do evaluation of a given parcellation %%
for i=1:length(toEval)
    for b = 1:length(bins)
        parcel = niftiread(fullfile(glmDir,'spect',sprintf('masked_par_%s.nii.gz',toEval{i})));
        parcel = round(parcel(volIndx));
        T=sc1_sc2_neocortical('Eval:DCBC_volume','parcel',parcel,'sess','average','distFile',dist,'bins',0:bins(b):90);
        save(fullfile(sprintf('Eval_%s_volume_MDTB_all_bin=%d.mat',toEval{i},bins(b))),'-struct','T');
        fprintf('Done %s \n',toEval{i});
    end
end 

%% Get DCBC values for all evaluated parcellations and plot %%
DCBC_all = [];
for i=1:length(toEval)
    for b = 1:length(bins)
        DCBC = sc1_sc2_neocortical('EVAL:getDCBC','toPlot',toEval{i},'bin_width',bins(b),'bins',90/bins(b));
    end
    DCBC_all = [DCBC_all DCBC];
end 
errorbar(1:11,mean(DCBC_all),std(DCBC_all),'LineStyle','none','MarkerSize',10,'Marker','.');
yline(0,'--','LineWidth',0.5,'Color',[0 0 0]);
set(gca,'xtick',[1:11],'xticklabel',toEval);
set(gcf,'PaperPosition',[0 0 8 4]);
wysiwyg

%% Plot DCBC curves (un-weighted) %%
T=load(sprintf('Eval_%s_volume_MDTB_all_bin=%d.mat',toEval{1},bins));
D=tapply(T,{'bin','SN','distmin','distmax','bwParcel'},{'corr'});
CAT.linecolor={'k','r'};
CAT.linestyle={'-','-'};
CAT.linewidth=2;
CAT.markertype='none';
D.binC = (D.distmin+D.distmax)/2;
%D.DCBC = D.corrW-D.corrB;
lineplot(D.binC,D.corr,'split',D.bwParcel,'CAT',CAT,'errorwidth',1);
set(gca,'XLim',[0 80],'YLim',[-0.05 0.4],'XTick',0:bins:80);
drawline(0,'dir','horz','color',[0.5 0.5 0.5]);
set(gcf,'PaperPosition',[2 2 3 3.7]);
ylabel('Voxel-to-voxel correlation')
xlabel('Spatial distance (mm)')
wysiwyg;

