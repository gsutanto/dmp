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
        
        self.expected_D_input = 45
        self.expected_N_phaseLWR_kernels = 25
        self.expected_D_output = 6
        
        self.chunk_size = 1
        
        self.is_performing_weighted_training = True
        self.is_only_optimizing_roll_Ct = True
        
        # Initial Learning Rate
        self.init_learning_rate = 0.001
        
        self.N_data_demo = [None] * self.N_primitives
        self.N_data_rlit = [None] * self.N_primitives
        
        # demonstration (demo) dataset
        self.DeltaS_demo = [None] * self.N_primitives
        self.Ct_target_demo = [None] * self.N_primitives
        self.normalized_phase_kernels_demo = [None] * self.N_primitives
        self.data_point_priority_demo = [None] * self.N_primitives
        
        self.DeltaS_demo_train = [None] * self.N_primitives
        self.nPSI_demo_train = [None] * self.N_primitives
        self.Ctt_demo_train = [None] * self.N_primitives
        self.W_demo_train = [None] * self.N_primitives
        
        self.DeltaS_demo_valid = [None] * self.N_primitives
        self.nPSI_demo_valid = [None] * self.N_primitives
        self.Ctt_demo_valid = [None] * self.N_primitives
        self.W_demo_valid = [None] * self.N_primitives
        
        self.DeltaS_demo_test = [None] * self.N_primitives
        self.nPSI_demo_test = [None] * self.N_primitives
        self.Ctt_demo_test = [None] * self.N_primitives
        self.W_demo_test = [None] * self.N_primitives
        
        # RL iteration (rlit) dataset
        self.DeltaS_rlit = [None] * self.N_primitives
        self.nPSI_rlit = [None] * self.N_primitives
        self.Ctt_rlit = [None] * self.N_primitives
        self.W_rlit = [None] * self.N_primitives
        
        self.DeltaS_rlit_train = [None] * self.N_primitives
        self.nPSI_rlit_train = [None] * self.N_primitives
        self.Ctt_rlit_train = [None] * self.N_primitives
        self.W_rlit_train = [None] * self.N_primitives
        
        self.DeltaS_rlit_valid = [None] * self.N_primitives
        self.nPSI_rlit_valid = [None] * self.N_primitives
        self.Ctt_rlit_valid = [None] * self.N_primitives
        self.W_rlit_valid = [None] * self.N_primitives
        
        self.DeltaS_rlit_test = [None] * self.N_primitives
        self.nPSI_rlit_test = [None] * self.N_primitives
        self.Ctt_rlit_test = [None] * self.N_primitives
        self.W_rlit_test = [None] * self.N_primitives
        for n_prim in range(1, self.N_primitives):
            # load dataset:
            self.DeltaS_demo[n_prim] = sio.loadmat('../scraping/prim_'+str(n_prim+1)+'_X_raw_scraping.mat', struct_as_record=True)['X'].astype(np.float32)
            self.Ct_target_demo[n_prim] = sio.loadmat('../scraping/prim_'+str(n_prim+1)+'_Ct_target_scraping.mat', struct_as_record=True)['Ct_target'].astype(np.float32)
            self.normalized_phase_kernels_demo[n_prim] = sio.loadmat('../scraping/prim_'+str(n_prim+1)+'_normalized_phase_PSI_mult_phase_V_scraping.mat', struct_as_record=True)['normalized_phase_PSI_mult_phase_V'].astype(np.float32)
            self.data_point_priority_demo[n_prim] = sio.loadmat('../scraping/prim_'+str(n_prim+1)+'_data_point_priority_scraping.mat', struct_as_record=True)['data_point_priority'].astype(np.float32)
            
            print('DeltaS_demo[%d].shape                   = ' % (n_prim) + str(self.DeltaS_demo[n_prim].shape))
            print('Ct_target_demo[%d].shape                = ' % (n_prim) + str(self.Ct_target_demo[n_prim].shape))
            print('normalized_phase_kernels_demo[%d].shape = ' % (n_prim) + str(self.normalized_phase_kernels_demo[n_prim].shape))
            print('data_point_priority_demo[%d].shape      = ' % (n_prim) + str(self.data_point_priority_demo[n_prim].shape))
            
            self.N_data_demo[n_prim] = self.Ct_target_demo[n_prim].shape[0]
            print('N_data_demo[%d]   = ' % (n_prim) + str(self.N_data_demo[n_prim]))
            print('D_input_demo[%d]  = ' % (n_prim) + str(self.DeltaS_demo[n_prim].shape[1]))
            print('D_output_demo[%d] = ' % (n_prim) + str(self.Ct_target_demo[n_prim].shape[1]))
            assert (self.DeltaS_demo[n_prim].shape[1]    == self.expected_D_input)
            assert (self.Ct_target_demo[n_prim].shape[1] == self.expected_D_output)
            assert (self.normalized_phase_kernels_demo[n_prim].shape[1] == self.expected_N_phaseLWR_kernels)
            
            # Permutation with Chunks (for Stochastic Gradient Descent (SGD))
            demo_data_idx_chunks = list(py_util.chunks(range(self.N_data_demo[n_prim]), self.chunk_size))
            N_demo_chunks = len(demo_data_idx_chunks)
            
            N_demo_train_chunks = np.round(self.fraction_train_dataset * N_demo_chunks).astype(int)
            N_demo_test_chunks  = np.round(self.fraction_test_dataset * N_demo_chunks).astype(int)
            N_demo_valid_chunks = N_demo_chunks - N_demo_train_chunks - N_demo_test_chunks
            assert(N_demo_train_chunks >  0)
            assert(N_demo_test_chunks  >= 0)
            assert(N_demo_valid_chunks >= 0)
            
            demo_chunk_permutation = np.random.permutation(N_demo_chunks)
            demo_chunk_idx_train = np.sort(demo_chunk_permutation[0:N_demo_train_chunks], 0)
            demo_chunk_idx_valid = np.sort(demo_chunk_permutation[N_demo_train_chunks:(N_demo_train_chunks+N_demo_valid_chunks)], 0)
            demo_chunk_idx_test  = np.sort(demo_chunk_permutation[(N_demo_train_chunks+N_demo_valid_chunks):N_demo_chunks], 0)
            idx_train_demo_dataset = np.concatenate([demo_data_idx_chunks[i] for i in demo_chunk_idx_train])
            idx_valid_demo_dataset = np.concatenate([demo_data_idx_chunks[i] for i in demo_chunk_idx_valid])
            idx_test_demo_dataset  = np.concatenate([demo_data_idx_chunks[i] for i in demo_chunk_idx_test])
            
            self.DeltaS_demo_train[n_prim] = self.DeltaS_demo[n_prim][idx_train_demo_dataset,:]
            self.nPSI_demo_train[n_prim] = self.normalized_phase_kernels_demo[n_prim][idx_train_demo_dataset,:]
            self.Ctt_demo_train[n_prim] = self.Ct_target_demo[n_prim][idx_train_demo_dataset,:]
            self.W_demo_train[n_prim] = self.data_point_priority_demo[n_prim][idx_train_demo_dataset,:]
            
            self.DeltaS_demo_valid[n_prim] = self.DeltaS_demo[n_prim][idx_valid_demo_dataset,:]
            self.nPSI_demo_valid[n_prim] = self.normalized_phase_kernels_demo[n_prim][idx_valid_demo_dataset,:]
            self.Ctt_demo_valid[n_prim] = self.Ct_target_demo[n_prim][idx_valid_demo_dataset,:]
            self.W_demo_valid[n_prim] = self.data_point_priority_demo[n_prim][idx_valid_demo_dataset,:]
            
            self.DeltaS_demo_test[n_prim] = self.DeltaS_demo[n_prim][idx_test_demo_dataset,:]
            self.nPSI_demo_test[n_prim] = self.normalized_phase_kernels_demo[n_prim][idx_test_demo_dataset,:]
            self.Ctt_demo_test[n_prim] = self.Ct_target_demo[n_prim][idx_test_demo_dataset,:]
            self.W_demo_test[n_prim] = self.data_point_priority_demo[n_prim][idx_test_demo_dataset,:]
    
    def trainPMNNWithAdditionalRLIterDatasetInitializedAtPath(self, rl_data, 
                                                              prim_tbi, # prim-to-be-improved
                                                              iterations_list, 
                                                              initial_pmnn_params_dirpath # should be cpp_models dirpath
                                                              ):
        DeltaS_rlit_list = list()
        nPSI_rlit_list = list()
        Ctt_rlit_list = list()
        W_rlit_list = list()
        for it in iterations_list:
            for n_trial in range(len(rl_data[prim_tbi][it][prim_tbi])):
                DeltaS_rlit_list.append(rl_data[prim_tbi][it][prim_tbi][n_trial]['DeltaS'])
                nPSI_rlit_list.append(rl_data[prim_tbi][it][prim_tbi][n_trial]['normalized_phase_PSI_mult_phase_V'])
                Ctt_rlit_list.append(rl_data[prim_tbi][it][prim_tbi][n_trial]['Ct_target'])
                W_rlit_list.append(rl_data[prim_tbi][it][prim_tbi][n_trial]['data_point_priority'])
        
        self.DeltaS_rlit[prim_tbi] = np.vstack(DeltaS_rlit_list)
        self.nPSI_rlit[prim_tbi] = np.vstack(nPSI_rlit_list)
        self.Ctt_rlit[prim_tbi] = np.vstack(Ctt_rlit_list)
        self.W_rlit[prim_tbi] = np.vstack(W_rlit_list)
        
        print('DeltaS_rlit[%d].shape                   = ' % (prim_tbi) + str(self.DeltaS_rlit[prim_tbi].shape))
        print('Ct_target_rlit[%d].shape                = ' % (prim_tbi) + str(self.Ct_target_rlit[prim_tbi].shape))
        print('normalized_phase_kernels_rlit[%d].shape = ' % (prim_tbi) + str(self.normalized_phase_kernels_rlit[prim_tbi].shape))
        print('data_point_priority_rlit[%d].shape      = ' % (prim_tbi) + str(self.data_point_priority_rlit[prim_tbi].shape))
        
        self.N_data_rlit[prim_tbi] = self.Ct_target_rlit[prim_tbi].shape[0]
        print('N_data_rlit[%d]   = ' % (prim_tbi) + str(self.N_data_rlit[prim_tbi]))
        assert (self.DeltaS_rlit[prim_tbi].shape[1]    == self.expected_D_input)
        assert (self.Ct_target_rlit[prim_tbi].shape[1] == self.expected_D_output)
        assert (self.normalized_phase_kernels_rlit[prim_tbi].shape[1] == self.expected_N_phaseLWR_kernels)
        
        # Permutation with Chunks (for Stochastic Gradient Descent (SGD))
        rlit_data_idx_chunks = list(py_util.chunks(range(self.N_data_rlit[prim_tbi]), self.chunk_size))
        N_rlit_chunks = len(rlit_data_idx_chunks)
        
        N_rlit_train_chunks = np.round(self.fraction_train_dataset * N_rlit_chunks).astype(int)
        N_rlit_test_chunks  = np.round(self.fraction_test_dataset * N_rlit_chunks).astype(int)
        N_rlit_valid_chunks = N_rlit_chunks - N_rlit_train_chunks - N_rlit_test_chunks
        assert(N_rlit_train_chunks >  0)
        assert(N_rlit_test_chunks  >= 0)
        assert(N_rlit_valid_chunks >= 0)
        
        rlit_chunk_permutation = np.random.permutation(N_rlit_chunks)
        rlit_chunk_idx_train = np.sort(rlit_chunk_permutation[0:N_rlit_train_chunks], 0)
        rlit_chunk_idx_valid = np.sort(rlit_chunk_permutation[N_rlit_train_chunks:(N_rlit_train_chunks+N_rlit_valid_chunks)], 0)
        rlit_chunk_idx_test  = np.sort(rlit_chunk_permutation[(N_rlit_train_chunks+N_rlit_valid_chunks):N_rlit_chunks], 0)
        idx_train_rlit_dataset = np.concatenate([rlit_data_idx_chunks[i] for i in rlit_chunk_idx_train])
        idx_valid_rlit_dataset = np.concatenate([rlit_data_idx_chunks[i] for i in rlit_chunk_idx_valid])
        idx_test_rlit_dataset  = np.concatenate([rlit_data_idx_chunks[i] for i in rlit_chunk_idx_test])
        
        self.DeltaS_rlit_train[prim_tbi] = self.DeltaS_rlit[prim_tbi][idx_train_rlit_dataset,:]
        self.nPSI_rlit_train[prim_tbi] = self.normalized_phase_kernels_rlit[prim_tbi][idx_train_rlit_dataset,:]
        self.Ctt_rlit_train[prim_tbi] = self.Ct_target_rlit[prim_tbi][idx_train_rlit_dataset,:]
        self.W_rlit_train[prim_tbi] = self.data_point_priority_rlit[prim_tbi][idx_train_rlit_dataset,:]
        
        self.DeltaS_rlit_valid[prim_tbi] = self.DeltaS_rlit[prim_tbi][idx_valid_rlit_dataset,:]
        self.nPSI_rlit_valid[prim_tbi] = self.normalized_phase_kernels_rlit[prim_tbi][idx_valid_rlit_dataset,:]
        self.Ctt_rlit_valid[prim_tbi] = self.Ct_target_rlit[prim_tbi][idx_valid_rlit_dataset,:]
        self.W_rlit_valid[prim_tbi] = self.data_point_priority_rlit[prim_tbi][idx_valid_rlit_dataset,:]
        
        self.DeltaS_rlit_test[prim_tbi] = self.DeltaS_rlit[prim_tbi][idx_test_rlit_dataset,:]
        self.nPSI_rlit_test[prim_tbi] = self.normalized_phase_kernels_rlit[prim_tbi][idx_test_rlit_dataset,:]
        self.Ctt_rlit_test[prim_tbi] = self.Ct_target_rlit[prim_tbi][idx_test_rlit_dataset,:]
        self.W_rlit_test[prim_tbi] = self.data_point_priority_rlit[prim_tbi][idx_test_rlit_dataset,:]
        
        # combine demo and rlit dataset:
        self.DeltaS_train = np.vstack([self.DeltaS_demo_train[prim_tbi], self.DeltaS_rlit_train[prim_tbi]])
        self.nPSI_train = np.vstack([self.nPSI_demo_train[prim_tbi], self.nPSI_rlit_train[prim_tbi]])
        self.Ctt_train = np.vstack([self.Ctt_demo_train[prim_tbi], self.Ctt_rlit_train[prim_tbi]])
        self.W_train = np.vstack([self.W_demo_train[prim_tbi], self.W_rlit_train[prim_tbi]])
        
        self.DeltaS_valid = np.vstack([self.DeltaS_demo_valid[prim_tbi], self.DeltaS_rlit_valid[prim_tbi]])
        self.nPSI_valid = np.vstack([self.nPSI_demo_valid[prim_tbi], self.nPSI_rlit_valid[prim_tbi]])
        self.Ctt_valid = np.vstack([self.Ctt_demo_valid[prim_tbi], self.Ctt_rlit_valid[prim_tbi]])
        self.W_valid = np.vstack([self.W_demo_valid[prim_tbi], self.W_rlit_valid[prim_tbi]])
        
        self.DeltaS_test = np.vstack([self.DeltaS_demo_test[prim_tbi], self.DeltaS_rlit_test[prim_tbi]])
        self.nPSI_test = np.vstack([self.nPSI_demo_test[prim_tbi], self.nPSI_rlit_test[prim_tbi]])
        self.Ctt_test = np.vstack([self.Ctt_demo_test[prim_tbi], self.Ctt_rlit_test[prim_tbi]])
        self.W_test = np.vstack([self.W_demo_test[prim_tbi], self.W_rlit_test[prim_tbi]])
        
        # Training Dataset Index is Permuted for Stochastic Gradient Descent (SGD)
        permuted_idx_train_dataset = np.random.permutation(self.DeltaS_train.shape[0])
        self.DeltaS_train = self.DeltaS_train[permuted_idx_train_dataset,:]
        self.nPSI_train = self.nPSI_train[permuted_idx_train_dataset,:]
        self.Ctt_train = self.Ctt_train[permuted_idx_train_dataset,:]
        self.W_train = self.W_train[permuted_idx_train_dataset,:]
        
        N_train_dataset = self.DeltaS_train.shape[0]
        N_valid_dataset = self.DeltaS_valid.shape[0]
        N_test_dataset = self.DeltaS_test.shape[0]
        print('N_train_dataset = %d' % N_train_dataset)
        print('N_valid_dataset = %d' % N_valid_dataset)
        print('N_test_dataset  = %d' % N_test_dataset)
            
        # Build the complete graph for feeding inputs, training, and saving checkpoints.
        self.pmnn_graph = tf.Graph()
        with self.pmnn_graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.
            self.tf_train_DeltaS_batch = tf.placeholder(tf.float32, shape=[self.batch_size, self.expected_D_input], name="tf_train_DeltaS_batch_placeholder")
            self.tf_train_nPSI_batch = tf.placeholder(tf.float32, shape=[self.batch_size, self.expected_N_phaseLWR_kernels], name="tf_train_nPSI_batch_placeholder")
            self.tf_train_W_batch = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name="tf_train_W_batch_placeholder")
            self.tf_train_Ctt_batch = tf.placeholder(tf.float32, shape=[self.batch_size, self.expected_D_output], name="tf_train_Ctt_batch_placeholder")
            self.tf_train_DeltaS = tf.placeholder(tf.float32, shape=[None, self.expected_D_input], name="tf_train_DeltaS_placeholder")
            self.tf_train_nPSI = tf.placeholder(tf.float32, shape=[None, self.expected_N_phaseLWR_kernels], name="tf_train_nPSI_placeholder")
            self.tf_valid_DeltaS = tf.placeholder(tf.float32, shape=[None, self.expected_D_input], name="tf_valid_DeltaS_placeholder")
            self.tf_valid_nPSI = tf.placeholder(tf.float32, shape=[None, self.expected_N_phaseLWR_kernels], name="tf_valid_nPSI_placeholder")
            self.tf_test_DeltaS = tf.placeholder(tf.float32, shape=[None, self.expected_D_input], name="tf_test_DeltaS_placeholder")
            self.tf_test_nPSI = tf.placeholder(tf.float32, shape=[None, self.expected_N_phaseLWR_kernels], name="tf_test_nPSI_placeholder")
            
            self.pmnn = PMNN(name=self.NN_name, D_input=self.expected_D_input, 
                             regular_hidden_layer_topology=self.regular_NN_hidden_layer_topology, 
                             regular_hidden_layer_activation_func_list=self.regular_NN_hidden_layer_activation_func_list, 
                             N_phaseLWR_kernels=self.expected_N_phaseLWR_kernels, 
                             D_output=self.expected_D_output, 
                             path=initial_pmnn_params_dirpath + "/prim%d/" % prim_tbi, 
                             is_using_phase_kernel_modulation=True)
        
            # Build the Prediction Graph (that computes predictions from the inference model).
            self.train_batch_prediction = self.pmnn.performNeuralNetworkPrediction(self.tf_train_DeltaS_batch, 
                                                                                   self.tf_train_nPSI_batch, 
                                                                                   self.tf_train_dropout_keep_prob)
            
            self.train_op_dim = [None] * self.expected_D_output
            self.loss_dim = [None] * self.expected_D_output
            # Build the Training Graph (that calculate and apply gradients), per output dimension.
            for self.d_output in range(self.expected_D_output):
                if (self.is_performing_weighted_training):
                    [self.train_op_dim[self.d_output], 
                     self.loss_dim[self.d_output]
                     ] = self.pmnn.performNeuralNetworkWeightedTrainingPerDimOut(self.train_batch_prediction, 
                                                                                 self.tf_train_Ctt_batch, 
                                                                                 self.init_learning_rate, 
                                                                                 self.beta, self.d_output, 
                                                                                 self.tf_train_W_batch)
                else:
                    [self.train_op_dim[self.d_output], 
                     self.loss_dim[self.d_output]
                     ]= self.pmnn.performNeuralNetworkTrainingPerDimOut(self.train_batch_prediction, 
                                                                        self.tf_train_Ctt_batch, 
                                                                        self.init_learning_rate, 
                                                                        self.beta, self.d_output)
                
                # Create a summary:
                tf.summary.scalar("loss_dim_%d"%(self.d_output), self.loss_dim[self.d_output])
        
            # merge all summaries into a single "operation" which we can execute in a session
            self.summary_op = tf.summary.merge_all()
        
            # Predictions for the training, validation, and test data.
            self.train_prediction = self.pmnn.performNeuralNetworkPrediction(self.tf_train_DeltaS, self.tf_train_nPSI, 1.0)
            self.valid_prediction = self.pmnn.performNeuralNetworkPrediction(self.tf_valid_DeltaS, self.tf_valid_nPSI, 1.0)
            self.test_prediction  = self.pmnn.performNeuralNetworkPrediction(self.tf_test_DeltaS, self.tf_test_nPSI, 1.0)
        
        # Run training for TF_max_train_iters and save checkpoint at the end.
        with tf.Session(graph=self.pmnn_graph) as session:
            # Run the Op to initialize the variables.
            tf.global_variables_initializer().run()
            print("Initialized")
            
            if (self.pmnn.num_params < N_train_dataset):
                print("OK: self.pmnn.num_params=%d < %d=N_train_dataset" % (self.pmnn.num_params, N_train_dataset))
            else:
                print("WARNING: self.pmnn.num_params=%d >= %d=N_train_dataset" % (self.pmnn.num_params, N_train_dataset))
            
            # create log writer object
            self.writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())
        
            # Start the training loop.
            for step in range(self.TF_max_train_iters):
                # Read a batch of input dataset and labels.
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * self.batch_size) % (N_train_dataset - self.batch_size)
                # Generate a minibatch.
                batch_X = self.X_train[offset:(offset + self.batch_size), :]
                batch_nPSI = self.nPSI_train[offset:(offset + self.batch_size), :]
                batch_W = self.W_train[offset:(offset + self.batch_size), :]
                batch_Ctt = self.Ctt_train[offset:(offset + self.batch_size), :]
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict_train = {self.tf_train_DeltaS_batch : batch_X, 
                                   self.tf_train_nPSI_batch : batch_nPSI, 
                                   self.tf_train_W_batch : batch_W, 
                                   self.tf_train_Ctt_batch : batch_Ctt}
        
                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.  To
                # inspect the values of your Ops or variables, you may include them
                # in the list passed to sess.run() and the value tensors will be
                # returned in the tuple from the call.
                if (self.is_only_optimizing_roll_Ct):
                    [_, 
                     tr_batch_prediction, summary
                     ] = session.run([self.train_op_dim[4], 
                                      self.train_batch_prediction, self.summary_op
                                      ], feed_dict=feed_dict_train)
                else:
                    [_, _, _, _, _, _, 
                     tr_batch_prediction, summary
                     ] = session.run([self.train_op_dim[0], self.train_op_dim[1], self.train_op_dim[2], 
                                      self.train_op_dim[3], self.train_op_dim[4], self.train_op_dim[5], 
                                      self.train_batch_prediction, self.summary_op
                                      ], feed_dict=feed_dict_train)
                
                # write log
                self.writer.add_summary(summary, step)
                
                if (step % 1000 == 0):
                    print("Minibatch NMSE at step %d:" % (step), py_util.computeNMSE(tr_batch_prediction, batch_Ctt))
                if ((step % 5000 == 0) or ((step > 0) and ((step == np.power(10,(np.floor(np.log10(step)))).astype(np.int32)) or (step == 5 * np.power(10,(np.floor(np.log10(step/5)))).astype(np.int32))))):
                    nmse = {}
                    print("")
                    
                    feed_dict_eval_rlit = {self.tf_train_DeltaS : self.DeltaS_rlit_train[prim_tbi], 
                                           self.tf_train_nPSI : self.nPSI_rlit_train[prim_tbi], 
                                           self.tf_valid_DeltaS : self.DeltaS_rlit_valid[prim_tbi], 
                                           self.tf_valid_nPSI : self.nPSI_rlit_valid[prim_tbi], 
                                           self.tf_test_DeltaS : self.DeltaS_rlit_test[prim_tbi], 
                                           self.tf_test_nPSI : self.nPSI_rlit_test[prim_tbi]}
                    [rlit_train_prediction, rlit_valid_prediction, rlit_test_prediction
                     ] = session.run([self.train_prediction, self.valid_prediction, self.test_prediction
                                      ], feed_dict=feed_dict_eval_rlit)
                    
                    feed_dict_eval_demo = {self.tf_train_DeltaS : self.DeltaS_demo_train[prim_tbi], 
                                           self.tf_train_nPSI : self.nPSI_demo_train[prim_tbi], 
                                           self.tf_valid_DeltaS : self.DeltaS_demo_valid[prim_tbi], 
                                           self.tf_valid_nPSI : self.nPSI_demo_valid[prim_tbi], 
                                           self.tf_test_DeltaS : self.DeltaS_demo_test[prim_tbi], 
                                           self.tf_test_nPSI : self.nPSI_demo_test[prim_tbi]}
                    [demo_train_prediction, demo_valid_prediction, demo_test_prediction
                     ] = session.run([self.train_prediction, self.valid_prediction, self.test_prediction
                                      ], feed_dict=feed_dict_eval_demo)
                    
                    feed_dict_eval = {self.tf_train_DeltaS : self.DeltaS_train, 
                                      self.tf_train_nPSI : self.nPSI_train, 
                                      self.tf_valid_DeltaS : self.DeltaS_valid, 
                                      self.tf_valid_nPSI : self.nPSI_valid, 
                                      self.tf_test_DeltaS : self.DeltaS_test, 
                                      self.tf_test_nPSI : self.nPSI_test}
                    [train_prediction, valid_prediction, test_prediction
                     ] = session.run([self.train_prediction, self.valid_prediction, self.test_prediction
                                      ], feed_dict=feed_dict_eval)
                    
                    if ((self.is_performing_weighted_training) and (step % 5000 == 0) and (step > 0)):
                        rlit_wnmse_train = py_util.computeWNMSE(rlit_train_prediction, self.Ctt_rlit_train[prim_tbi], self.W_rlit_train[prim_tbi])
                        rlit_wnmse_valid = py_util.computeWNMSE(rlit_valid_prediction, self.Ctt_rlit_valid[prim_tbi], self.W_rlit_valid[prim_tbi])
                        rlit_wnmse_test = py_util.computeWNMSE(rlit_test_prediction, self.Ctt_rlit_test[prim_tbi], self.W_rlit_test[prim_tbi])
                        print("RLit Training       WNMSE: ", rlit_wnmse_train)
                        print("RLit Validation     WNMSE: ", rlit_wnmse_valid)
                        print("RLit Test           WNMSE: ", rlit_wnmse_test)
                        print("")
                        nmse["rlit_wnmse_train"] = rlit_wnmse_train
                        nmse["rlit_wnmse_valid"] = rlit_wnmse_valid
                        nmse["rlit_wnmse_test"] = rlit_wnmse_test
                        
                        demo_wnmse_train = py_util.computeWNMSE(demo_train_prediction, self.Ctt_demo_train[prim_tbi], self.W_demo_train[prim_tbi])
                        demo_wnmse_valid = py_util.computeWNMSE(demo_valid_prediction, self.Ctt_demo_valid[prim_tbi], self.W_demo_valid[prim_tbi])
                        demo_wnmse_test = py_util.computeWNMSE(demo_test_prediction, self.Ctt_demo_test[prim_tbi], self.W_demo_test[prim_tbi])
                        print("Demo Training       WNMSE: ", demo_wnmse_train)
                        print("Demo Validation     WNMSE: ", demo_wnmse_valid)
                        print("Demo Test           WNMSE: ", demo_wnmse_test)
                        print("")
                        nmse["demo_wnmse_train"] = demo_wnmse_train
                        nmse["demo_wnmse_valid"] = demo_wnmse_valid
                        nmse["demo_wnmse_test"] = demo_wnmse_test
                        
                        wnmse_train = py_util.computeWNMSE(train_prediction, self.Ctt_train, self.W_train)
                        wnmse_valid = py_util.computeWNMSE(valid_prediction, self.Ctt_valid, self.W_valid)
                        wnmse_test = py_util.computeWNMSE(test_prediction, self.Ctt_test, self.W_test)
                        print("Training            WNMSE: ", wnmse_train)
                        print("Validation          WNMSE: ", wnmse_valid)
                        print("Test                WNMSE: ", wnmse_test)
                        print("")
                        nmse["wnmse_train"] = wnmse_train
                        nmse["wnmse_valid"] = wnmse_valid
                        nmse["wnmse_test"] = wnmse_test
                        
                        print("")
                    
                    rlit_nmse_train = py_util.computeNMSE(rlit_train_prediction, self.Ctt_rlit_train[prim_tbi])
                    rlit_nmse_valid = py_util.computeNMSE(rlit_valid_prediction, self.Ctt_rlit_valid[prim_tbi])
                    rlit_nmse_test = py_util.computeNMSE(rlit_test_prediction, self.Ctt_rlit_test[prim_tbi])
                    rlit_var_ground_truth_Ctt_train = np.var(self.Ctt_rlit_train[prim_tbi], axis=0)
                    print("RLit Training        NMSE: ", rlit_nmse_train)
                    print("RLit Validation      NMSE: ", rlit_nmse_valid)
                    print("RLit Test            NMSE: ", rlit_nmse_test)
                    print("RLit Training    Variance: ", rlit_var_ground_truth_Ctt_train)
                    print("")
                    
                    demo_nmse_train = py_util.computeNMSE(demo_train_prediction, self.Ctt_demo_train[prim_tbi])
                    demo_nmse_valid = py_util.computeNMSE(demo_valid_prediction, self.Ctt_demo_valid[prim_tbi])
                    demo_nmse_test = py_util.computeNMSE(demo_test_prediction, self.Ctt_demo_test[prim_tbi])
                    demo_var_ground_truth_Ctt_train = np.var(self.Ctt_demo_train[prim_tbi], axis=0)
                    print("Demo Training        NMSE: ", demo_nmse_train)
                    print("Demo Validation      NMSE: ", demo_nmse_valid)
                    print("Demo Test            NMSE: ", demo_nmse_test)
                    print("Demo Training    Variance: ", demo_var_ground_truth_Ctt_train)
                    print("")
                    
                    nmse_train = py_util.computeNMSE(train_prediction, self.Ctt_train)
                    nmse_valid = py_util.computeNMSE(valid_prediction, self.Ctt_valid)
                    nmse_test = py_util.computeNMSE(test_prediction, self.Ctt_test)
                    var_ground_truth_Ctt_train = np.var(self.Ctt_train, axis=0)
                    print("Training             NMSE: ", nmse_train)
                    print("Validation           NMSE: ", nmse_valid)
                    print("Test                 NMSE: ", nmse_test)
                    print("Training         Variance: ", var_ground_truth_Ctt_train)
                    print("")
                    
                    NN_model_params = self.pmnn.saveNeuralNetworkToMATLABMatFile()
                    sio.savemat((self.rl_model_output_dir_path+'prim_'+str(prim_tbi+1)+'_params_step_%07d'%step+'.mat'), NN_model_params)
                    nmse["nmse_train"] = nmse_train
                    nmse["nmse_valid"] = nmse_valid
                    nmse["nmse_test"] = nmse_test
                    sio.savemat((self.rl_model_output_dir_path+'prim_'+str(prim_tbi+1)+'_nmse_step_%07d'%step+'.mat'), nmse)
                    var_ground_truth = {}
                    var_ground_truth["var_ground_truth_Ctt_train"] = var_ground_truth_Ctt_train
                    sio.savemat((self.rl_model_output_dir_path+'prim_'+str(prim_tbi+1)+'_var_ground_truth.mat'), var_ground_truth)
            print("")
            if (self.is_performing_weighted_training):
                print("Final RLit Training       WNMSE: ", rlit_wnmse_train)
                print("Final RLit Validation     WNMSE: ", rlit_wnmse_valid)
                print("Final RLit Test           WNMSE: ", rlit_wnmse_test)
                print("")
                
                print("Final Demo Training       WNMSE: ", demo_wnmse_train)
                print("Final Demo Validation     WNMSE: ", demo_wnmse_valid)
                print("Final Demo Test           WNMSE: ", demo_wnmse_test)
                print("")
                
                print("Final Training            WNMSE: ", wnmse_train)
                print("Final Validation          WNMSE: ", wnmse_valid)
                print("Final Test                WNMSE: ", wnmse_test)
                print("")
            
            print("Final RLit Training        NMSE: ", rlit_nmse_train)
            print("Final RLit Validation      NMSE: ", rlit_nmse_valid)
            print("Final RLit Test            NMSE: ", rlit_nmse_test)
            print("")
            
            print("Final Demo Training        NMSE: ", demo_nmse_train)
            print("Final Demo Validation      NMSE: ", demo_nmse_valid)
            print("Final Demo Test            NMSE: ", demo_nmse_test)
            print("")
            
            print("Final Training             NMSE: ", nmse_train)
            print("Final Validation           NMSE: ", nmse_valid)
            print("Final Test                 NMSE: ", nmse_test)
            print("")