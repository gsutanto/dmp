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
import rl_tactile_fb_utils as rl_util

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
        
        self.chunk_size = np.loadtxt(self.initial_model_parent_dir_path+'chunk_size.txt', dtype=np.int, ndmin=0) + 0
        
        self.is_performing_weighted_training = True
        self.is_only_optimizing_roll_Ct = True
        
        # Initial Learning Rate
        self.init_learning_rate = 0.001
        
        self.N_data_demo = [None] * self.N_primitives
        self.N_data_rlit = [None] * self.N_primitives
        
        # demonstration (demo) dataset
        self.DeltaS_demo = [None] * self.N_primitives
        self.nPSI_demo = [None] * self.N_primitives
        self.Ctt_demo = [None] * self.N_primitives
        self.W_demo = [None] * self.N_primitives
        
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
            self.Ctt_demo[n_prim] = sio.loadmat('../scraping/prim_'+str(n_prim+1)+'_Ct_target_scraping.mat', struct_as_record=True)['Ct_target'].astype(np.float32)
            self.nPSI_demo[n_prim] = sio.loadmat('../scraping/prim_'+str(n_prim+1)+'_normalized_phase_PSI_mult_phase_V_scraping.mat', struct_as_record=True)['normalized_phase_PSI_mult_phase_V'].astype(np.float32)
            self.W_demo[n_prim] = sio.loadmat('../scraping/prim_'+str(n_prim+1)+'_data_point_priority_scraping.mat', struct_as_record=True)['data_point_priority'].astype(np.float32)
            
            [self.DeltaS_demo_train[n_prim], self.nPSI_demo_train[n_prim], self.Ctt_demo_train[n_prim], self.W_demo_train[n_prim], 
             self.DeltaS_demo_valid[n_prim], self.nPSI_demo_valid[n_prim], self.Ctt_demo_valid[n_prim], self.W_demo_valid[n_prim], 
             self.DeltaS_demo_test[n_prim],  self.nPSI_demo_test[n_prim],  self.Ctt_demo_test[n_prim],  self.W_demo_test[n_prim]
             ] = rl_util.splitDatasetIntoTrainValidTestSubDataset(self.DeltaS_demo[n_prim], self.Ctt_demo[n_prim], 
                                                                  self.nPSI_demo[n_prim], self.W_demo[n_prim], 
                                                                  "_demo", n_prim, 
                                                                  self.expected_D_input, self.expected_D_output, self.expected_N_phaseLWR_kernels, self.chunk_size, 
                                                                  self.fraction_train_dataset, self.fraction_test_dataset)
    
    def savePMNNParamsFromDictAtDirPath(self, pmnn_params_dirpath, pmnn_params):
        N_primitives = len(pmnn_params.keys())
        py_util.createDirIfNotExist(pmnn_params_dirpath)
        for n_prim in range(N_primitives):
            pmnn_prim_param_dirpath = pmnn_params_dirpath+"/prim%d/"%(n_prim+1)
            
            self.pmnn = PMNN(name=self.NN_name, D_input=self.expected_D_input, 
                             regular_hidden_layer_topology=self.regular_NN_hidden_layer_topology, 
                             regular_hidden_layer_activation_func_list=self.regular_NN_hidden_layer_activation_func_list, 
                             N_phaseLWR_kernels=self.expected_N_phaseLWR_kernels, 
                             D_output=self.expected_D_output, 
                             path="", 
                             is_using_phase_kernel_modulation=True, 
                             is_predicting_only=True, 
                             model_params_dict=pmnn_params[n_prim])
            
            self.pmnn.saveNeuralNetworkToTextFiles(pmnn_prim_param_dirpath)
        return None
    
    def loadPMNNParamsAsDictFromDirPath(self, pmnn_params_dirpath, N_primitives):
        pmnn_params = {}
        for n_prim in range(N_primitives):
            pmnn_prim_param_dirpath = pmnn_params_dirpath+"/prim%d/"%(n_prim+1)
            
            self.pmnn = PMNN(name=self.NN_name, D_input=self.expected_D_input, 
                             regular_hidden_layer_topology=self.regular_NN_hidden_layer_topology, 
                             regular_hidden_layer_activation_func_list=self.regular_NN_hidden_layer_activation_func_list, 
                             N_phaseLWR_kernels=self.expected_N_phaseLWR_kernels, 
                             D_output=self.expected_D_output, 
                             path=pmnn_prim_param_dirpath, 
                             is_using_phase_kernel_modulation=True, 
                             is_predicting_only=True)
            
            pmnn_params[n_prim] = self.pmnn.model_params
        return pmnn_params
        
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
            for n_trial in range(len(rl_data[prim_tbi][it]["additional_fb_dataset"][prim_tbi])):
                DeltaS_rlit_list.append(rl_data[prim_tbi][it]["additional_fb_dataset"][prim_tbi][n_trial]['DeltaS'])
                nPSI_rlit_list.append(rl_data[prim_tbi][it]["additional_fb_dataset"][prim_tbi][n_trial]['normalized_phase_PSI_mult_phase_V'])
                Ctt_rlit_list.append(rl_data[prim_tbi][it]["additional_fb_dataset"][prim_tbi][n_trial]['Ct_target'])
                W_rlit_list.append(rl_data[prim_tbi][it]["additional_fb_dataset"][prim_tbi][n_trial]['data_point_priority'])
        
        self.DeltaS_rlit[prim_tbi] = np.vstack(DeltaS_rlit_list)
        self.nPSI_rlit[prim_tbi] = np.vstack(nPSI_rlit_list)
        self.Ctt_rlit[prim_tbi] = np.vstack(Ctt_rlit_list)
        self.W_rlit[prim_tbi] = np.vstack(W_rlit_list)
        
        [self.DeltaS_rlit_train[prim_tbi], self.nPSI_rlit_train[prim_tbi], self.Ctt_rlit_train[prim_tbi], self.W_rlit_train[prim_tbi], 
         self.DeltaS_rlit_valid[prim_tbi], self.nPSI_rlit_valid[prim_tbi], self.Ctt_rlit_valid[prim_tbi], self.W_rlit_valid[prim_tbi], 
         self.DeltaS_rlit_test[prim_tbi],  self.nPSI_rlit_test[prim_tbi],  self.Ctt_rlit_test[prim_tbi],  self.W_rlit_test[prim_tbi]
         ] = rl_util.splitDatasetIntoTrainValidTestSubDataset(self.DeltaS_rlit[prim_tbi], self.Ctt_rlit[prim_tbi], 
                                                              self.nPSI_rlit[prim_tbi], self.W_rlit[prim_tbi], 
                                                              "_rlit", prim_tbi, 
                                                              self.expected_D_input, self.expected_D_output, self.expected_N_phaseLWR_kernels, self.chunk_size, 
                                                              self.fraction_train_dataset, self.fraction_test_dataset)
        
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
        
        # Measure proportion of rlit and demo dataset:
        self.percentage_proportion_rlit_train = self.DeltaS_rlit_train[prim_tbi].shape[0] * 100.0 / self.DeltaS_train.shape[0]
        self.percentage_proportion_rlit_valid = self.DeltaS_rlit_valid[prim_tbi].shape[0] * 100.0 / self.DeltaS_valid.shape[0]
        self.percentage_proportion_rlit_test  = self.DeltaS_rlit_test[prim_tbi].shape[0]  * 100.0 / self.DeltaS_test.shape[0]
        print('percentage_proportion_rlit_train = %f%%' % self.percentage_proportion_rlit_train)
        print('percentage_proportion_rlit_valid = %f%%' % self.percentage_proportion_rlit_valid)
        print('percentage_proportion_rlit_test  = %f%%' % self.percentage_proportion_rlit_test)
        
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
        
        self.prim_pmnn_params_dirpath = initial_pmnn_params_dirpath + "/prim%d/" % (prim_tbi+1)
            
        # Build the complete graph for feeding inputs, training, and saving checkpoints.
        self.pmnn_graph = tf.Graph()
        with self.pmnn_graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.
            self.tf_train_DeltaS_batch_ph = tf.placeholder(tf.float32, shape=[self.batch_size, self.expected_D_input], name="tf_train_DeltaS_batch_placeholder")
            self.tf_train_nPSI_batch_ph = tf.placeholder(tf.float32, shape=[self.batch_size, self.expected_N_phaseLWR_kernels], name="tf_train_nPSI_batch_placeholder")
            self.tf_train_W_batch_ph = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name="tf_train_W_batch_placeholder")
            self.tf_train_Ctt_batch_ph = tf.placeholder(tf.float32, shape=[self.batch_size, self.expected_D_output], name="tf_train_Ctt_batch_placeholder")
            self.tf_train_DeltaS_ph = tf.placeholder(tf.float32, shape=[None, self.expected_D_input], name="tf_train_DeltaS_placeholder")
            self.tf_train_nPSI_ph = tf.placeholder(tf.float32, shape=[None, self.expected_N_phaseLWR_kernels], name="tf_train_nPSI_placeholder")
            self.tf_valid_DeltaS_ph = tf.placeholder(tf.float32, shape=[None, self.expected_D_input], name="tf_valid_DeltaS_placeholder")
            self.tf_valid_nPSI_ph = tf.placeholder(tf.float32, shape=[None, self.expected_N_phaseLWR_kernels], name="tf_valid_nPSI_placeholder")
            self.tf_test_DeltaS_ph = tf.placeholder(tf.float32, shape=[None, self.expected_D_input], name="tf_test_DeltaS_placeholder")
            self.tf_test_nPSI_ph = tf.placeholder(tf.float32, shape=[None, self.expected_N_phaseLWR_kernels], name="tf_test_nPSI_placeholder")
            
            # load (initial) PMNN params from self.prim_pmnn_params_dirpath
            self.pmnn = PMNN(name=self.NN_name, D_input=self.expected_D_input, 
                             regular_hidden_layer_topology=self.regular_NN_hidden_layer_topology, 
                             regular_hidden_layer_activation_func_list=self.regular_NN_hidden_layer_activation_func_list, 
                             N_phaseLWR_kernels=self.expected_N_phaseLWR_kernels, 
                             D_output=self.expected_D_output, 
                             path=self.prim_pmnn_params_dirpath, 
                             is_using_phase_kernel_modulation=True)
        
            # Build the Prediction Graph (that computes predictions from the inference model).
            self.train_batch_prediction = self.pmnn.performNeuralNetworkPrediction(self.tf_train_DeltaS_batch_ph, 
                                                                                   self.tf_train_nPSI_batch_ph, 
                                                                                   self.tf_train_dropout_keep_prob)
            
            self.train_op_dim = [None] * self.expected_D_output
            self.loss_dim = [None] * self.expected_D_output
            # Build the Training Graph (that calculate and apply gradients), per output dimension.
            for self.d_output in range(self.expected_D_output):
                if (self.is_performing_weighted_training):
                    [self.train_op_dim[self.d_output], 
                     self.loss_dim[self.d_output]
                     ] = self.pmnn.performNeuralNetworkWeightedTrainingPerDimOut(self.train_batch_prediction, 
                                                                                 self.tf_train_Ctt_batch_ph, 
                                                                                 self.init_learning_rate, 
                                                                                 self.beta, self.d_output, 
                                                                                 self.tf_train_W_batch_ph)
                else:
                    [self.train_op_dim[self.d_output], 
                     self.loss_dim[self.d_output]
                     ]= self.pmnn.performNeuralNetworkTrainingPerDimOut(self.train_batch_prediction, 
                                                                        self.tf_train_Ctt_batch_ph, 
                                                                        self.init_learning_rate, 
                                                                        self.beta, self.d_output)
                
                # Create a summary:
                tf.summary.scalar("loss_dim_%d"%(self.d_output), self.loss_dim[self.d_output])
        
            # merge all summaries into a single "operation" which we can execute in a session
            self.summary_op = tf.summary.merge_all()
        
            # Predictions for the training, validation, and test data.
            self.train_prediction = self.pmnn.performNeuralNetworkPrediction(self.tf_train_DeltaS_ph, self.tf_train_nPSI_ph, 1.0)
            self.valid_prediction = self.pmnn.performNeuralNetworkPrediction(self.tf_valid_DeltaS_ph, self.tf_valid_nPSI_ph, 1.0)
            self.test_prediction  = self.pmnn.performNeuralNetworkPrediction(self.tf_test_DeltaS_ph, self.tf_test_nPSI_ph, 1.0)
        
        # Run training for TF_max_train_iters and save checkpoint at the end.
        with tf.Session(graph=self.pmnn_graph) as self.session:
            tf_dict = dict()
            tf_dict['session'] = self.session
            tf_dict['tf_train_DeltaS_ph'] = self.tf_train_DeltaS_ph
            tf_dict['tf_train_nPSI_ph'] = self.tf_train_nPSI_ph
            tf_dict['tf_valid_DeltaS_ph'] = self.tf_valid_DeltaS_ph
            tf_dict['tf_valid_nPSI_ph'] = self.tf_valid_nPSI_ph
            tf_dict['tf_test_DeltaS_ph'] = self.tf_test_DeltaS_ph
            tf_dict['tf_test_nPSI_ph'] = self.tf_test_nPSI_ph
            tf_dict['train_prediction'] = self.train_prediction
            tf_dict['valid_prediction'] = self.valid_prediction
            tf_dict['test_prediction'] = self.test_prediction
            
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
                batch_DeltaS = self.DeltaS_train[offset:(offset + self.batch_size), :]
                batch_nPSI = self.nPSI_train[offset:(offset + self.batch_size), :]
                batch_W = self.W_train[offset:(offset + self.batch_size), :]
                batch_Ctt = self.Ctt_train[offset:(offset + self.batch_size), :]
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict_train = {self.tf_train_DeltaS_batch_ph : batch_DeltaS, 
                                   self.tf_train_nPSI_batch_ph : batch_nPSI, 
                                   self.tf_train_W_batch_ph : batch_W, 
                                   self.tf_train_Ctt_batch_ph : batch_Ctt}
        
                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.  To
                # inspect the values of your Ops or variables, you may include them
                # in the list passed to sess.run() and the value tensors will be
                # returned in the tuple from the call.
                if (self.is_only_optimizing_roll_Ct):
                    opt_dim = 4
                    [_, 
                     tr_batch_prediction, summary
                     ] = self.session.run([self.train_op_dim[opt_dim], 
                                           self.train_batch_prediction, self.summary_op
                                           ], feed_dict=feed_dict_train)
                else:
                    opt_dim = None
                    [_, _, _, _, _, _, 
                     tr_batch_prediction, summary
                     ] = self.session.run([self.train_op_dim[0], self.train_op_dim[1], self.train_op_dim[2], 
                                           self.train_op_dim[3], self.train_op_dim[4], self.train_op_dim[5], 
                                           self.train_batch_prediction, self.summary_op
                                           ], feed_dict=feed_dict_train)
                
                # write log
                self.writer.add_summary(summary, step)
                
                if (step % 1000 == 0):
                    batch_nmse_train = py_util.computeNMSE(tr_batch_prediction, batch_Ctt)
                    if (self.is_only_optimizing_roll_Ct):
                        batch_nmse_train = batch_nmse_train[opt_dim]
                    print("Minibatch NMSE at step %d: " % (step) + str(batch_nmse_train))
                if ((step % 5000 == 0) or ((step > 0) and ((step == np.power(10,(np.floor(np.log10(step)))).astype(np.int32)) or (step == 5 * np.power(10,(np.floor(np.log10(step/5)))).astype(np.int32))))):
                    eval_info = {}
                    print("")
                    
                    [rlit_wnmse_train, rlit_wnmse_valid, rlit_wnmse_test, rlit_nmse_train, rlit_nmse_valid, rlit_nmse_test, rlit_var_ground_truth_Ctt_train
                     ] = rl_util.displayLearningEvaluation(tf_dict, 
                                                           self.DeltaS_rlit_train[prim_tbi], self.nPSI_rlit_train[prim_tbi], self.Ctt_rlit_train[prim_tbi], self.W_rlit_train[prim_tbi], 
                                                           self.DeltaS_rlit_valid[prim_tbi], self.nPSI_rlit_valid[prim_tbi], self.Ctt_rlit_valid[prim_tbi], self.W_rlit_valid[prim_tbi], 
                                                           self.DeltaS_rlit_test[prim_tbi],  self.nPSI_rlit_test[prim_tbi],  self.Ctt_rlit_test[prim_tbi],  self.W_rlit_test[prim_tbi], 
                                                           step=step, is_performing_weighted_training=self.is_performing_weighted_training, print_prefix="RLit ", disp_dim=opt_dim)
                    
                    [demo_wnmse_train, demo_wnmse_valid, demo_wnmse_test, demo_nmse_train, demo_nmse_valid, demo_nmse_test, demo_var_ground_truth_Ctt_train
                     ] = rl_util.displayLearningEvaluation(tf_dict, 
                                                           self.DeltaS_demo_train[prim_tbi], self.nPSI_demo_train[prim_tbi], self.Ctt_demo_train[prim_tbi], self.W_demo_train[prim_tbi], 
                                                           self.DeltaS_demo_valid[prim_tbi], self.nPSI_demo_valid[prim_tbi], self.Ctt_demo_valid[prim_tbi], self.W_demo_valid[prim_tbi], 
                                                           self.DeltaS_demo_test[prim_tbi],  self.nPSI_demo_test[prim_tbi],  self.Ctt_demo_test[prim_tbi],  self.W_demo_test[prim_tbi], 
                                                           step=step, is_performing_weighted_training=self.is_performing_weighted_training, print_prefix="Demo ", disp_dim=opt_dim)
                    
                    [wnmse_train, wnmse_valid, wnmse_test, nmse_train, nmse_valid, nmse_test, var_ground_truth_Ctt_train
                     ] = rl_util.displayLearningEvaluation(tf_dict, 
                                                           self.DeltaS_train, self.nPSI_train, self.Ctt_train, self.W_train, 
                                                           self.DeltaS_valid, self.nPSI_valid, self.Ctt_valid, self.W_valid, 
                                                           self.DeltaS_test,  self.nPSI_test,  self.Ctt_test,  self.W_test, 
                                                           step=step, is_performing_weighted_training=self.is_performing_weighted_training, print_prefix="", disp_dim=opt_dim)
                    
                    if (self.is_performing_weighted_training):
                        if ((rlit_wnmse_train is not None) and (rlit_wnmse_valid is not None) and (rlit_wnmse_test is not None)):
                            eval_info["rlit_wnmse_train"] = rlit_wnmse_train
                            eval_info["rlit_wnmse_valid"] = rlit_wnmse_valid
                            eval_info["rlit_wnmse_test"] = rlit_wnmse_test
                        
                        if ((demo_wnmse_train is not None) and (demo_wnmse_valid is not None) and (demo_wnmse_test is not None)):
                            eval_info["demo_wnmse_train"] = demo_wnmse_train
                            eval_info["demo_wnmse_valid"] = demo_wnmse_valid
                            eval_info["demo_wnmse_test"] = demo_wnmse_test
                        
                        if ((wnmse_train is not None) and (wnmse_valid is not None) and (wnmse_test is not None)):
                            eval_info["wnmse_train"] = wnmse_train
                            eval_info["wnmse_valid"] = wnmse_valid
                            eval_info["wnmse_test"] = wnmse_test
                    
                    eval_info["rlit_nmse_train"] = rlit_nmse_train
                    eval_info["rlit_nmse_valid"] = rlit_nmse_valid
                    eval_info["rlit_nmse_test"] = rlit_nmse_test
                    eval_info["rlit_var_ground_truth_Ctt_train"] = rlit_var_ground_truth_Ctt_train
                    
                    eval_info["demo_nmse_train"] = demo_nmse_train
                    eval_info["demo_nmse_valid"] = demo_nmse_valid
                    eval_info["demo_nmse_test"] = demo_nmse_test
                    eval_info["demo_var_ground_truth_Ctt_train"] = demo_var_ground_truth_Ctt_train
                    
                    eval_info["nmse_train"] = nmse_train
                    eval_info["nmse_valid"] = nmse_valid
                    eval_info["nmse_test"] = nmse_test
                    eval_info["var_ground_truth_Ctt_train"] = var_ground_truth_Ctt_train
                    
                    NN_model_params = self.pmnn.saveNeuralNetworkToMATLABMatFile(self.rl_model_output_dir_path+'prim_'+str(prim_tbi+1)+'_params_step_%07d'%step+'.mat')
                    sio.savemat((self.rl_model_output_dir_path+'prim_'+str(prim_tbi+1)+'_eval_info_step_%07d'%step+'.mat'), eval_info)
            print("")
            if (self.is_performing_weighted_training):
                print("Final RLit Training       WNMSE: " + str(rlit_wnmse_train))
                print("Final RLit Validation     WNMSE: " + str(rlit_wnmse_valid))
                print("Final RLit Test           WNMSE: " + str(rlit_wnmse_test))
                print("")
                
                print("Final Demo Training       WNMSE: " + str(demo_wnmse_train))
                print("Final Demo Validation     WNMSE: " + str(demo_wnmse_valid))
                print("Final Demo Test           WNMSE: " + str(demo_wnmse_test))
                print("")
                
                print("Final Training            WNMSE: " + str(wnmse_train))
                print("Final Validation          WNMSE: " + str(wnmse_valid))
                print("Final Test                WNMSE: " + str(wnmse_test))
                print("")
            
            print("Final RLit Training        NMSE: " + str(rlit_nmse_train))
            print("Final RLit Validation      NMSE: " + str(rlit_nmse_valid))
            print("Final RLit Test            NMSE: " + str(rlit_nmse_test))
            print("")
            
            print("Final Demo Training        NMSE: " + str(demo_nmse_train))
            print("Final Demo Validation      NMSE: " + str(demo_nmse_valid))
            print("Final Demo Test            NMSE: " + str(demo_nmse_test))
            print("")
            
            print("Final Training             NMSE: " + str(nmse_train))
            print("Final Validation           NMSE: " + str(nmse_valid))
            print("Final Test                 NMSE: " + str(nmse_test))
            print("")
            
            # update optimized/trained PMNN params to self.prim_pmnn_params_dirpath
            self.pmnn.saveNeuralNetworkToTextFiles(self.prim_pmnn_params_dirpath)
        
        return NN_model_params, eval_info