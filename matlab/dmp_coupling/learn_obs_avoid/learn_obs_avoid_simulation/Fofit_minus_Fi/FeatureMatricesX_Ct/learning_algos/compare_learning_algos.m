clear all;
close all;
clc;

addpath('../../../../utilities/');

%% Data Loading

load('../X_groundTruth_stacked.mat');
load('../Ct_groundTruth_stacked.mat');
load('../X_observed_Yt_misaligned_stacked.mat');
load('../Ct_observed_Yt_misaligned_stacked.mat');
load('../X_observed_Yt_aligned_stacked.mat');
load('../Ct_observed_Yt_aligned_stacked.mat');

is_loading_data     = 1;

% end of Data Loading

%% ARD

tic
disp(['Performing ARD (ground truth):']);
[ w_gT_ard, nmse_gT_ard, Ct_gT_fit_ard ] = learnUsingARD( X_groundTruth_stacked, Ct_groundTruth_stacked );
toc

tic
disp(['Performing ARD (misaligned y(t)):']);
[ w_misalign_ard, nmse_misalign_ard, Ct_misalign_fit_ard ] = learnUsingARD( X_observed_Yt_misaligned_stacked, Ct_observed_Yt_misaligned_stacked );
toc

tic
disp(['Performing ARD (aligned y(t)):']);
[ w_align_ard, nmse_align_ard, Ct_align_fit_ard ] = learnUsingARD( X_observed_Yt_aligned_stacked, Ct_observed_Yt_aligned_stacked );
toc

% end of ARD

%% LASSO

if (~is_loading_data)
    tic
    disp(['Performing LASSO (ground truth):']);
    [ w_gT_lasso, nmse_gT_lasso, Ct_gT_fit_lasso ] = learnUsingLASSO( X_groundTruth_stacked, Ct_groundTruth_stacked );
    toc

    tic
    disp(['Performing LASSO (misaligned y(t)):']);
    [ w_misalign_lasso, nmse_misalign_lasso, Ct_misalign_fit_lasso ] = learnUsingLASSO( X_observed_Yt_misaligned_stacked, Ct_observed_Yt_misaligned_stacked );
    toc

    tic
    disp(['Performing LASSO (aligned y(t)):']);
    [ w_align_lasso, nmse_align_lasso, Ct_align_fit_lasso ] = learnUsingLASSO( X_observed_Yt_aligned_stacked, Ct_observed_Yt_aligned_stacked );
    toc
else
    load('lasso.mat');
end

% end of LASSO

%% VBLS

if (~is_loading_data)
    tic
    disp(['Performing VBLS (ground truth):']);
    [ w_gT_vbls, nmse_gT_vbls, Ct_gT_fit_vbls ] = learnUsingVBLS( X_groundTruth_stacked, Ct_groundTruth_stacked );
    toc

    tic
    disp(['Performing VBLS (misaligned y(t)):']);
    [ w_misalign_vbls, nmse_misalign_vbls, Ct_misalign_fit_vbls ] = learnUsingVBLS( X_observed_Yt_misaligned_stacked, Ct_observed_Yt_misaligned_stacked );
    toc

    tic
    disp(['Performing VBLS (aligned y(t)):']);
    [ w_align_vbls, nmse_align_vbls, Ct_align_fit_vbls ] = learnUsingVBLS( X_observed_Yt_aligned_stacked, Ct_observed_Yt_aligned_stacked );
    toc
else
    load('vbls.mat');
end

% end of VBLS

%% Normalized Mean Square (NMSE) Display

disp(['-------']);
disp(['ARD:']);
disp(['nmse (ground truth, with ARD)        = ', num2str(nmse_gT_ard)]);
disp(['nmse (misaligned y(t), with ARD)     = ', num2str(nmse_misalign_ard)]);
disp(['nmse (aligned y(t), with ARD)        = ', num2str(nmse_align_ard)]);
disp(['-------']);
disp(['LASSO:']);
disp(['nmse (ground truth, with LASSO)      = ', num2str(nmse_gT_lasso)]);
disp(['nmse (misaligned y(t), with LASSO)   = ', num2str(nmse_misalign_lasso)]);
disp(['nmse (aligned y(t), with LASSO)      = ', num2str(nmse_align_lasso)]);
disp(['-------']);
disp(['VBLS:']);
disp(['nmse (ground truth, with VBLS)       = ', num2str(nmse_gT_vbls)]);
disp(['nmse (misaligned y(t), with VBLS)    = ', num2str(nmse_misalign_vbls)]);
disp(['nmse (aligned y(t), with VBLS)       = ', num2str(nmse_align_vbls)]);

% end of Normalized Mean Square (NMSE) Display

%% Plotting

figure;
for d=1:2
    subplot(3,2,0+d); hold on; plot(w_gT_ard(:,d),'g'); plot(w_gT_lasso(:,d),'b'); plot(w_gT_vbls(:,d),'r'); title(['w (ground truth) d=',num2str(d)]); legend('ARD','LASSO','VBLS'); hold off;
    subplot(3,2,2+d); hold on; plot(w_misalign_ard(:,d),'g'); plot(w_misalign_lasso(:,d),'b'); plot(w_misalign_vbls(:,d),'r'); title(['w (mis-aligned y(t)) d=',num2str(d)]); legend('ARD','LASSO','VBLS'); hold off;
    subplot(3,2,4+d); hold on; plot(w_align_ard(:,d),'g'); plot(w_align_lasso(:,d),'b'); plot(w_align_vbls(:,d),'r'); title(['w (aligned y(t)) d=',num2str(d)]); legend('ARD','LASSO','VBLS'); hold off;
end

figure;
for d=1:2
    subplot(3,2,0+d); hold on; plot([1:45],Ct_groundTruth_stacked([1:45],d),'g'); plot([1:45],Ct_gT_fit_ard([1:45],d),'b'); title(['Ct (ground truth) d=',num2str(d)]); legend('ground truth','ARD'); hold off;
    subplot(3,2,2+d); hold on; plot([1:45],Ct_observed_Yt_misaligned_stacked([1:45],d),'g'); plot([1:45],Ct_misalign_fit_ard([1:45],d),'b'); title(['Ct (mis-aligned y(t)) d=',num2str(d)]); legend('ground truth','ARD'); hold off;
    subplot(3,2,4+d); hold on; plot([1:45],Ct_observed_Yt_aligned_stacked([1:45],d),'g'); plot([1:45],Ct_align_fit_ard([1:45],d),'b'); title(['Ct (aligned y(t)) d=',num2str(d)]); legend('ground truth','ARD'); hold off;
end

figure;
for d=1:2
    subplot(3,2,0+d); hold on; plot([1:45],Ct_groundTruth_stacked([1:45],d),'g'); plot([1:45],Ct_gT_fit_lasso([1:45],d),'b'); title(['Ct (ground truth) d=',num2str(d)]); legend('ground truth','LASSO'); hold off;
    subplot(3,2,2+d); hold on; plot([1:45],Ct_observed_Yt_misaligned_stacked([1:45],d),'g'); plot([1:45],Ct_misalign_fit_lasso([1:45],d),'b'); title(['Ct (mis-aligned y(t)) d=',num2str(d)]); legend('mis-aligned y(t)','LASSO'); hold off;
    subplot(3,2,4+d); hold on; plot([1:45],Ct_observed_Yt_aligned_stacked([1:45],d),'g'); plot([1:45],Ct_align_fit_lasso([1:45],d),'b'); title(['Ct (aligned y(t)) d=',num2str(d)]); legend('aligned y(t)','LASSO'); hold off;
end

figure;
for d=1:2
    subplot(3,2,0+d); hold on; plot([1:45],Ct_groundTruth_stacked([1:45],d),'g'); plot([1:45],Ct_gT_fit_vbls([1:45],d),'b'); title(['Ct (ground truth) d=',num2str(d)]); legend('ground truth','VBLS'); hold off;
    subplot(3,2,2+d); hold on; plot([1:45],Ct_observed_Yt_misaligned_stacked([1:45],d),'g'); plot([1:45],Ct_misalign_fit_vbls([1:45],d),'b'); title(['Ct (mis-aligned y(t)) d=',num2str(d)]); legend('mis-aligned y(t)','VBLS'); hold off;
    subplot(3,2,4+d); hold on; plot([1:45],Ct_observed_Yt_aligned_stacked([1:45],d),'g'); plot([1:45],Ct_align_fit_vbls([1:45],d),'b'); title(['Ct (aligned y(t)) d=',num2str(d)]); legend('aligned y(t)','VBLS'); hold off;
end

% end of Plotting