from __future__ import print_function
import numpy as np
import random
import scipy.io as sio
import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../neural_nets/feedforward/pmnn/'))
import utilities as py_util
from PMNN import PMNN

class RLTactileFbPMNNSupervisedTraining:
    def __init__(self):
        # Seed the random variables generator:
        random.seed(38)
        np.random.seed(38)
        
        self.initial_model_parent_dir_path = '../models/'
        
        # Create directories if not currently exist:
        self.rl_model_output_dir_path = './models/'
        py_util.createDirIfNotExist(self.rl_model_output_dir_path)
        
        self.N_primitives = 3
        self.N_NN_reinit_trials = 3
        self.batch_size = 64
        
        self.TF_max_train_iters = np.loadtxt(self.initial_model_parent_dir_path+'TF_max_train_iters.txt', dtype=np.int, ndmin=0) + 1
        self.regular_NN_hidden_layer_topology = list(np.loadtxt(self.initial_model_parent_dir_path+'regular_NN_hidden_layer_topology.txt', dtype=np.int, ndmin=1))
        self.regular_NN_hidden_layer_activation_func_list = list(np.loadtxt(self.initial_model_parent_dir_path+'regular_NN_hidden_layer_activation_func_list.txt', dtype=np.str, ndmin=1)) * len(self.regular_NN_hidden_layer_topology)
        
        # Dropouts:
        self.tf_train_dropout_keep_prob = 0.5
        
        # L2 Regularization Constant
        self.beta = 0.0
        
        self.logs_path = "/tmp/pmnn/"
        
        self.NN_name = 'my_PMNN'
        
        self.fraction_train_dataset = 0.85
        self.fraction_test_dataset  = 0.075
        
        self.expected_D_output = 6
        
        self.chunk_size = 1
        
        self.is_performing_weighted_training = True
        self.is_only_optimizing_roll_Ct = True
        
        # Initial Learning Rate
        self.init_learning_rate = 0.001
        
        self.DeltaS_demo = [None] * self.N_primitives
        self.Ct_target_demo = [None] * self.N_primitives
        self.normalized_phase_kernels_demo = [None] * self.N_primitives
        self.data_point_priority_demo = [None] * self.N_primitives
        for n_prim in range(1, self.N_primitives):
            # load dataset:
            self.DeltaS_demo[n_prim] = sio.loadmat('../scraping/prim_'+str(n_prim+1)+'_X_raw_scraping.mat', struct_as_record=True)['X'].astype(np.float32)
            self.Ct_target_demo[n_prim] = sio.loadmat('../scraping/prim_'+str(n_prim+1)+'_Ct_target_scraping.mat', struct_as_record=True)['Ct_target'].astype(np.float32)
            self.normalized_phase_kernels_demo[n_prim] = sio.loadmat('../scraping/prim_'+str(n_prim+1)+'_normalized_phase_PSI_mult_phase_V_scraping.mat', struct_as_record=True)['normalized_phase_PSI_mult_phase_V'].astype(np.float32)
            self.data_point_priority_demo[n_prim] = sio.loadmat('../scraping/prim_'+str(n_prim+1)+'_data_point_priority_scraping.mat', struct_as_record=True)['data_point_priority'].astype(np.float32)
            
            print('DeltaS_demo[%d].shape                   = ' % (n_prim), self.DeltaS_demo[n_prim].shape)
            print('Ct_target_demo[%d].shape                = ' % (n_prim), self.Ct_target_demo[n_prim].shape)
            print('normalized_phase_kernels_demo[%d].shape = ' % (n_prim), self.normalized_phase_kernels_demo[n_prim].shape)
            print('data_point_priority_demo[%d].shape      = ' % (n_prim), self.data_point_priority_demo[n_prim].shape)
            
            N_data = self.Ct_target_demo[n_prim].shape[0]
            D_input = self.DeltaS_demo[n_prim].shape[1]
            D_output = self.Ct_target_demo[n_prim].shape[1]
            print('N_data   =', N_data)
            print('D_input  =', D_input)
            print('D_output =', D_output)
            assert (D_output == self.expected_D_output)
            
            N_phaseLWR_kernels = self.normalized_phase_kernels_demo[n_prim].shape[1]
            
            # Permutation with Chunks (for Stochastic Gradient Descent (SGD))
            data_idx_chunks = list(py_util.chunks(range(N_data), self.chunk_size))
            N_chunks = len(data_idx_chunks)
            
            N_train_chunks = np.round(self.fraction_train_dataset * N_chunks).astype(int)
            N_test_chunks = np.round(self.fraction_test_dataset * N_chunks).astype(int)
            N_valid_chunks = N_chunks - N_train_chunks - N_test_chunks
            
            chunk_permutation = np.random.permutation(N_chunks)
            chunk_idx_train = np.sort(chunk_permutation[0:N_train_chunks], 0)
            chunk_idx_valid = np.sort(chunk_permutation[N_train_chunks:(N_train_chunks+N_valid_chunks)], 0)
            chunk_idx_test = np.sort(chunk_permutation[(N_train_chunks+N_valid_chunks):N_chunks], 0)
            idx_train_dataset = np.concatenate([data_idx_chunks[i] for i in chunk_idx_train])
            idx_valid_dataset = np.concatenate([data_idx_chunks[i] for i in chunk_idx_valid])
            idx_test_dataset = np.concatenate([data_idx_chunks[i] for i in chunk_idx_test])
            # Training Dataset Index is Permuted for Stochastic Gradient Descent (SGD)
            permuted_idx_train_dataset = idx_train_dataset[np.random.permutation(len(idx_train_dataset))]
            assert (((set(permuted_idx_train_dataset).union(set(idx_valid_dataset))).union(set(idx_test_dataset))) == set(np.arange(N_data))), "NOT all data is utilized!"
            
            X_train = self.DeltaS_demo[n_prim][permuted_idx_train_dataset,:]
            nPSI_train = self.normalized_phase_kernels_demo[n_prim][permuted_idx_train_dataset,:]
            Ctt_train = self.Ct_target_demo[n_prim][permuted_idx_train_dataset,:]
            W_train = self.data_point_priority_demo[n_prim][permuted_idx_train_dataset,:]
            
            X_valid = self.DeltaS_demo[n_prim][idx_valid_dataset,:]
            nPSI_valid = self.normalized_phase_kernels_demo[n_prim][idx_valid_dataset,:]
            Ctt_valid = self.Ct_target_demo[n_prim][idx_valid_dataset,:]
            W_valid = self.data_point_priority_demo[n_prim][idx_valid_dataset,:]
            
            X_test = self.DeltaS_demo[n_prim][idx_test_dataset,:]
            nPSI_test = self.normalized_phase_kernels_demo[n_prim][idx_test_dataset,:]
            Ctt_test = self.Ct_target_demo[n_prim][idx_test_dataset,:]
            W_test = self.data_point_priority_demo[n_prim][idx_test_dataset,:]
            
            N_train_dataset = X_train.shape[0]
            N_valid_dataset = X_valid.shape[0]
            N_test_dataset = X_test.shape[0]
            print('N_train_dataset =', N_train_dataset)
            print('N_valid_dataset =', N_valid_dataset)
            print('N_test_dataset  =', N_test_dataset)
            
            # Build the complete graph for feeding inputs, training, and saving checkpoints.
            ff_nn_graph = tf.Graph()
            with ff_nn_graph.as_default():
                # Input data. For the training data, we use a placeholder that will be fed
                # at run time with a training minibatch.
                tf_train_X_batch = tf.placeholder(tf.float32, shape=[self.batch_size, D_input], name="tf_train_X_batch_placeholder")
                tf_train_nPSI_batch = tf.placeholder(tf.float32, shape=[self.batch_size, N_phaseLWR_kernels], name="tf_train_nPSI_batch_placeholder")
                tf_train_W_batch = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name="tf_train_W_batch_placeholder")
                tf_train_Ctt_batch = tf.placeholder(tf.float32, shape=[self.batch_size, D_output], name="tf_train_Ctt_batch_placeholder")
                tf_train_X = tf.constant(X_train, name="tf_train_X_constant")
                tf_train_nPSI = tf.constant(nPSI_train, name="tf_train_nPSI_constant")
                tf_valid_X = tf.constant(X_valid, name="tf_valid_X_constant")
                tf_valid_nPSI = tf.constant(nPSI_valid, name="tf_valid_nPSI_constant")
                tf_test_X = tf.constant(X_test, name="tf_test_X_constant")
                tf_test_nPSI = tf.constant(nPSI_test, name="tf_test_nPSI_constant")
                
                pmnn = PMNN(self.NN_name, D_input, 
                            self.regular_NN_hidden_layer_topology, self.regular_NN_hidden_layer_activation_func_list, 
                            N_phaseLWR_kernels, D_output, "", is_using_phase_kernel_modulation=True)
            
                # Build the Prediction Graph (that computes predictions from the inference model).
                train_batch_prediction = pmnn.performNeuralNetworkPrediction(tf_train_X_batch, tf_train_nPSI_batch, self.tf_train_dropout_keep_prob)
                
                train_op_dim = [None] * D_output
                loss_dim = [None] * D_output
                
                # Build the Training Graph (that calculate and apply gradients), per output dimension.
                for d_output in range(D_output):
                    if (self.is_performing_weighted_training):
                        [train_op_dim[d_output], 
                         loss_dim[d_output]
                         ] = pmnn.performNeuralNetworkWeightedTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, self.init_learning_rate, self.beta, d_output, tf_train_W_batch)
                    else:
                        [train_op_dim[d_output], 
                         loss_dim[d_output]
                         ]= pmnn.performNeuralNetworkTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, self.init_learning_rate, self.beta, d_output)
                    
                    # Create a summary:
                    tf.summary.scalar("loss_dim_%d"%(d_output), loss_dim[d_output])
            
                # merge all summaries into a single "operation" which we can execute in a session
                summary_op = tf.summary.merge_all()
            
                # Predictions for the training, validation, and test data.
                train_prediction = pmnn.performNeuralNetworkPrediction(tf_train_X, tf_train_nPSI, 1.0)
                valid_prediction = pmnn.performNeuralNetworkPrediction(tf_valid_X, tf_valid_nPSI, 1.0)
                test_prediction  = pmnn.performNeuralNetworkPrediction(tf_test_X, tf_test_nPSI, 1.0)
            
            # Run training for TF_max_train_iters and save checkpoint at the end.
            with tf.Session(graph=ff_nn_graph) as session:
                for n_NN_reinit_trial in range(self.N_NN_reinit_trials):
                    print ("n_NN_reinit_trial = ", n_NN_reinit_trial)
                    
                    # Run the Op to initialize the variables.
                    tf.global_variables_initializer().run()
                    print("Initialized")
                    
                    if (pmnn.num_params < N_train_dataset):
                        print("OK: pmnn.num_params=%d < %d=N_train_dataset" % (pmnn.num_params, N_train_dataset))
                    else:
                        print("WARNING: pmnn.num_params=%d >= %d=N_train_dataset" % (pmnn.num_params, N_train_dataset))
                    
                    # create log writer object
                    writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())
                
                    # Start the training loop.
                    for step in range(self.TF_max_train_iters):
                        # Read a batch of input dataset and labels.
                        # Pick an offset within the training data, which has been randomized.
                        # Note: we could use better randomization across epochs.
                        offset = (step * self.batch_size) % (Ctt_train.shape[0] - self.batch_size)
                        # Generate a minibatch.
                        batch_X = X_train[offset:(offset + self.batch_size), :]
                        batch_nPSI = nPSI_train[offset:(offset + self.batch_size), :]
                        batch_W = W_train[offset:(offset + self.batch_size), :]
                        batch_Ctt = Ctt_train[offset:(offset + self.batch_size), :]
                        # Prepare a dictionary telling the session where to feed the minibatch.
                        # The key of the dictionary is the placeholder node of the graph to be fed,
                        # and the value is the numpy array to feed to it.
                        feed_dict = {tf_train_X_batch : batch_X, tf_train_nPSI_batch : batch_nPSI, tf_train_W_batch : batch_W, tf_train_Ctt_batch : batch_Ctt}
                
                        # Run one step of the model.  The return values are the activations
                        # from the `train_op` (which is discarded) and the `loss` Op.  To
                        # inspect the values of your Ops or variables, you may include them
                        # in the list passed to sess.run() and the value tensors will be
                        # returned in the tuple from the call.
                        if (self.is_only_optimizing_roll_Ct):
                            [_, 
                             tr_batch_prediction, summary
                             ] = session.run([train_op_dim[4], 
                                              train_batch_prediction, summary_op
                                              ], feed_dict=feed_dict)
                        else:
                            [_, _, _, _, _, _, 
                             tr_batch_prediction, summary
                             ] = session.run([train_op_dim[0], train_op_dim[1], train_op_dim[2], train_op_dim[3], train_op_dim[4], train_op_dim[5], 
                                              train_batch_prediction, summary_op
                                              ], feed_dict=feed_dict)
                        
                        # write log
                        writer.add_summary(summary, step)
                        
                        if (step % 1000 == 0):
                            print("Minibatch NMSE :", py_util.computeNMSE(tr_batch_prediction, batch_Ctt))
                        if ((step % 5000 == 0) or ((step > 0) and ((step == np.power(10,(np.floor(np.log10(step)))).astype(np.int32)) or (step == 5 * np.power(10,(np.floor(np.log10(step/5)))).astype(np.int32))))):
                            nmse = {}
                            print("")
                            if ((self.is_performing_weighted_training) and (step % 5000 == 0) and (step > 0)):
                                wnmse_train = py_util.computeWNMSE(train_prediction.eval(), Ctt_train, W_train)
                                wnmse_valid = py_util.computeWNMSE(valid_prediction.eval(), Ctt_valid, W_valid)
                                wnmse_test = py_util.computeWNMSE(test_prediction.eval(), Ctt_test, W_test)
                                print("Training            WNMSE: ", wnmse_train)
                                print("Validation          WNMSE: ", wnmse_valid)
                                print("Test                WNMSE: ", wnmse_test)
                                nmse["wnmse_train"] = wnmse_train
                                nmse["wnmse_valid"] = wnmse_valid
                                nmse["wnmse_test"] = wnmse_test
                            nmse_train = py_util.computeNMSE(train_prediction.eval(), Ctt_train)
                            nmse_valid = py_util.computeNMSE(valid_prediction.eval(), Ctt_valid)
                            nmse_test = py_util.computeNMSE(test_prediction.eval(), Ctt_test)
                            var_ground_truth_Ctt_train = np.var(Ctt_train, axis=0)
                            print("Training             NMSE: ", nmse_train)
                            print("Validation           NMSE: ", nmse_valid)
                            print("Test                 NMSE: ", nmse_test)
                            print("Training         Variance: ", var_ground_truth_Ctt_train)
                            print("")
                            
                            NN_model_params = pmnn.saveNeuralNetworkToMATLABMatFile()
                            sio.savemat((self.rl_model_output_dir_path+'prim_'+str(n_prim+1)+'_params_reinit_'+str(n_NN_reinit_trial)+'_step_%07d'%step+'.mat'), NN_model_params)
                            nmse["nmse_train"] = nmse_train
                            nmse["nmse_valid"] = nmse_valid
                            nmse["nmse_test"] = nmse_test
                            sio.savemat((self.rl_model_output_dir_path+'prim_'+str(n_prim+1)+'_nmse_reinit_'+str(n_NN_reinit_trial)+'_step_%07d'%step+'.mat'), nmse)
                            var_ground_truth = {}
                            var_ground_truth["var_ground_truth_Ctt_train"] = var_ground_truth_Ctt_train
                            sio.savemat((self.rl_model_output_dir_path+'prim_'+str(n_prim+1)+'_var_ground_truth.mat'), var_ground_truth)
                    print("")
                    if (self.is_performing_weighted_training):
                        print("Final Training            WNMSE: ", wnmse_train)
                        print("Final Validation          WNMSE: ", wnmse_valid)
                        print("Final Test                WNMSE: ", wnmse_test)
                    print("Final Training             NMSE: ", nmse_train)
                    print("Final Validation           NMSE: ", nmse_valid)
                    print("Final Test                 NMSE: ", nmse_test)
                    print("")