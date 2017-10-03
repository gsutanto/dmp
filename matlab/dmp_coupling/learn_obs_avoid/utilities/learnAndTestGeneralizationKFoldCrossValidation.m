function [ nmse_train_cell, nmse_test_cell, w_cell, w_per_dim_cell, max_w_cell, min_w_cell, sparsity_w_cell ] = learnAndTestGeneralizationKFoldCrossValidation( X, Ct, learning_methods_cell, k_fold_array )
    D                               = size(Ct,2);
    N_learning_method               = size(learning_methods_cell,2);

    nmse_train_cell                 = cell(size(k_fold_array,2),N_learning_method);
    nmse_test_cell                  = cell(size(k_fold_array,2),N_learning_method);
    w_cell                          = cell(size(k_fold_array,2),N_learning_method);
    w_per_dim_cell                  = cell(size(k_fold_array,2),N_learning_method);
    max_w_cell                      = cell(size(k_fold_array,2),N_learning_method);
    min_w_cell                      = cell(size(k_fold_array,2),N_learning_method);
    sparsity_w_cell                 = cell(size(k_fold_array,2),N_learning_method);
    
    all_idx                         = [1:size(X,1)];

    for f_idx=1:size(k_fold_array,2)
        for l_idx=1:N_learning_method
            nmse_train_cell{f_idx,l_idx}= zeros(k_fold_array(1,f_idx),D);
            nmse_test_cell{f_idx,l_idx} = zeros(k_fold_array(1,f_idx),D);
            w_cell{f_idx,l_idx}         = cell(k_fold_array(1,f_idx),1);
            w_per_dim_cell{f_idx,l_idx} = cell(1,D);
            for d=1:D
                w_per_dim_cell{f_idx,l_idx}{1,d}    = zeros(size(X,2), k_fold_array(1,f_idx));
            end
            max_w_cell{f_idx,l_idx}     = zeros(k_fold_array(1,f_idx),D);
            min_w_cell{f_idx,l_idx}     = zeros(k_fold_array(1,f_idx),D);
            sparsity_w_cell{f_idx,l_idx}= zeros(k_fold_array(1,f_idx),D);
        end
        
        rest_idx_rand_permutated    = randperm(size(X,1));
        for j=1:k_fold_array(1,f_idx)
            disp(['performing generalization test: k-fold cross validation ',num2str(f_idx),' out of ',num2str(size(k_fold_array,2)),' cross-validations',...
                  ', ',num2str(j),'-th fold out of ',num2str(k_fold_array(1,f_idx))]);

            if (j ~= k_fold_array(1,f_idx))
                test_idx            = rest_idx_rand_permutated(1:round(size(X,1)/k_fold_array(1,f_idx)));
            else % if (j == k_fold_array(1,f_idx)) => the last one take the rest
                test_idx            = rest_idx_rand_permutated;
            end
            rest_idx_rand_permutated= rest_idx_rand_permutated(length(test_idx)+1:end);
            train_idx               = setdiff(all_idx,test_idx);
            
            % Bug Checking:
            supposed_all_idx        = union(train_idx, test_idx);
            supposed_null_set       = setdiff(all_idx, supposed_all_idx);
            if (isempty(supposed_null_set) ~= 1)
                disp('BUG: indexing bug; NOT all data samples are covered');
                return;
            end

            disp('--------------------------------------------------------');
            for l_idx=1:N_learning_method
                disp(learning_methods_cell{1,l_idx});
                w_cell{f_idx,l_idx}{j,1}    = zeros(size(X,2), D);

                for d=1:D
                    disp([' => dim: ', num2str(d)]);
                    
                    tic
                    if (strcmp(learning_methods_cell{1,l_idx},'ARD') == 1)
                        [w_ard_d,r_ard_idx] = ARD( X(train_idx,:), Ct(train_idx,d), 0);
                        w_cell{f_idx,l_idx}{j,1}(r_ard_idx,d)   = w_ard_d;
                    elseif (strcmp(learning_methods_cell{1,l_idx},'LASSO') == 1)
                        alpha       = 1;
                        num_lambda  = 3;
                        [w_lasso_d]	= lasso( X(train_idx,:), Ct(train_idx,d), 'Alpha', alpha, 'NumLambda', num_lambda );
                        w_cell{f_idx,l_idx}{j,1}(:,d)       = w_lasso_d(:,1);   % pick the column of w_lasso_d corresponding to the smallest regularization constant
                    end
                    toc
                    
                    w_per_dim_cell{f_idx,l_idx}{1,d}(:,j)   = w_cell{f_idx,l_idx}{j,1}(:,d);
                    max_w_cell{f_idx,l_idx}(j,d)            = max(w_cell{f_idx,l_idx}{j,1}(:,d));
                    min_w_cell{f_idx,l_idx}(j,d)            = min(w_cell{f_idx,l_idx}{j,1}(:,d));
                    sparsity_w_cell{f_idx,l_idx}(j,d)       = length(find(w_cell{f_idx,l_idx}{j,1}(:,d) ~= 0));
                end

                [ mse_train, nmse_train_cell{f_idx,l_idx}(j,:), Ct_fit_train ] = computeNMSE( X(train_idx,:), w_cell{f_idx,l_idx}{j,1}, Ct(train_idx,:) );
                [ mse_test, nmse_test_cell{f_idx,l_idx}(j,:), Ct_fit_test ] = computeNMSE( X(test_idx,:), w_cell{f_idx,l_idx}{j,1}, Ct(test_idx,:) );
                disp(['   nmse_train = ',num2str(nmse_train_cell{f_idx,l_idx}(j,:))]);
                disp(['   nmse_test  = ',num2str(nmse_test_cell{f_idx,l_idx}(j,:))]);
                disp(['   max_w      = ',num2str(max_w_cell{f_idx,l_idx}(j,:))]);
                disp(['   min_w      = ',num2str(min_w_cell{f_idx,l_idx}(j,:))]);
                disp(['   sparsity_w = ',num2str(sparsity_w_cell{f_idx,l_idx}(j,:))]);
                disp('--------------------------------------------------------');
            end
            disp('--------------------------------------------------------');
        end
    end
end

