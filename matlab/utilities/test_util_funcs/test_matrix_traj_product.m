clear all;
close all;
clc;

addpath('../');

average_computation_speed   = zeros(3,1);
sum_of_errors_C1            = 0.0;
sum_of_errors_C2            = 0.0;

N_trials    = 1000;

for n_trial=1:N_trials
    if (mod(n_trial,50) == 0)
        fprintf('n_trial = %d/%d\n',n_trial,N_trials);
    end

    A           = rand(4,1000,7);
    B           = rand(7,1000,20);
    
    %% Using manual/iterative matrix products:

    tic;
    I           = size(A,1);
    J           = size(A,2);
    K           = size(B,3);
    L           = size(A,3);
    C_manual    = zeros(I,J,K);
    for j=1:J
        A_2D            = reshape(A(:,j,:),I,L,1);
        B_2D            = reshape(B(:,j,:),L,K,1);
        C_manual(:,j,:) = reshape((A_2D*B_2D),I,1,K);
    end
    average_computation_speed(1,1)  = average_computation_speed(1,1) + toc;

    %% Using tensor product method 1:

    tic;
    C1  = computeMatrixTrajectoryProduct(A, B, 1);
    average_computation_speed(2,1)  = average_computation_speed(2,1) + toc;
    sum_of_errors_C1    = sum_of_errors_C1 + sum(sum(sum(C1 - C_manual)));

    %% Using tensor product method 2:

    tic;
    C2  = computeMatrixTrajectoryProduct(A, B, 2);
    average_computation_speed(3,1)  = average_computation_speed(3,1) + toc;
    sum_of_errors_C2    = sum_of_errors_C2 + sum(sum(sum(C2 - C_manual)));
end

average_computation_speed   = average_computation_speed/N_trials;

%% Results:

fprintf('Ave. comp. time using manual/iterative matrix products = %f\n', average_computation_speed(1,1));
fprintf('\n');

fprintf('Ave. comp. time using tensor product method 1          = %f\n', average_computation_speed(2,1));
fprintf('Sum of Differences (C1) = %f\n', sum_of_errors_C1);
fprintf('\n');

fprintf('Ave. comp. time using tensor product method 2          = %f\n', average_computation_speed(3,1));
fprintf('Sum of Differences (C2) = %f\n', sum_of_errors_C2);
fprintf('\n');