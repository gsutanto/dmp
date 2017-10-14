#include "amd_clmc_dmp/neural_nets/PMNN.h"

namespace dmp
{

    PMNN::PMNN():
        num_dimensions(0), data_io(DataIO()), rt_assertor(NULL)
    {
        weights.resize(0);
        biases.resize(0);
    }

    /**
     * @param num_dimensions_init Initialization of the number of dimensions that will be used
     * @param topology_init Initialization of the neural network topology that will be used
     * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
     */
    PMNN::PMNN(uint num_dimensions_init, std::vector< uint > topology_init, RealTimeAssertor* real_time_assertor):
        num_dimensions(num_dimensions_init),
        data_io(DataIO(real_time_assertor)), rt_assertor(real_time_assertor)
    {
        bool is_real_time   = false;

        topology    = topology_init;
        weights.resize(num_dimensions_init);
        biases.resize(num_dimensions_init);
        for (uint i=0; i<num_dimensions_init; ++i)
        {
            weights[i].resize(topology_init.size()-1);
            biases[i].resize(topology_init.size()-2);

            for (uint j=0; j<topology_init.size()-1; ++j)
            {
                rt_assert(allocateMemoryIfNonRealTime(is_real_time, weights[i][j],
                                                      topology_init[j], topology_init[j+1]));
                if (j < topology_init.size()-2)
                {
                    rt_assert(allocateMemoryIfNonRealTime(is_real_time, biases[i][j],
                                                          1, topology_init[j+1]));
                }
            }
        }
    }

    /**
     * Checks whether this special neural network is valid or not.
     *
     * @return Valid (true) or invalid (false)
     */
    bool PMNN::isValid()
    {
        if (rt_assertor == NULL)
        {
            return false;
        }
        if (rt_assert(data_io.isValid()) == false)
        {
            return false;
        }
        if (rt_assert(num_dimensions > 0) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(weights.size() == num_dimensions)) &&
                      (rt_assert(biases.size()  == num_dimensions))) == false)
        {
            return false;
        }
        if (rt_assert(topology[topology.size()-1] == 1) == false)
        {
            return false;
        }
        for (uint i=0; i<num_dimensions; ++i)
        {
            if (rt_assert((rt_assert(weights[i].size() > 0)) &&
                          (rt_assert(weights[i].size() == (topology.size()-1))) &&
                          (rt_assert(biases[i].size()  == (weights[i].size() - 1)))) == false)   // because output layer does NOT have biases
            {
                return false;
            }

            for (uint j=0; j<topology.size()-1; ++j)
            {
                if (rt_assert(weights[i][j]->rows() == topology[j]) == false)
                {
                    return false;
                }
                if (rt_assert(weights[i][j]->cols() == topology[j+1]) == false)
                {
                    return false;
                }
                if (j < topology.size()-2)
                {
                    if (rt_assert((rt_assert(biases[i][j]->rows() == 1)) &&
                                  (rt_assert(biases[i][j]->cols() == topology[j+1]))) == false)   // because output layer does NOT have biases
                    {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    /**
     * NOT REAL-TIME!!!
     *
     * Load neural network parameters from text files in the specified directory.
     *
     * @param dir_path Directory containing the parameters
     * @param start_idx Start sub-directory index to read from
     * @return Success or failure
     */
    bool PMNN::loadParams(const char* dir_path, uint start_idx)
    {
        // pre-condition(s) checking
        if (rt_assert(this->isValid()) == false)
        {
            return false;
        }

        for (uint dim_idx=start_idx; dim_idx<(start_idx+num_dimensions); ++dim_idx)
        {
            // Count how many weight files are available:
            uint    weight_file_counter         = 1;
            char    var_weight_file_path[1000]  = "";
            sprintf(var_weight_file_path, "%s/%u/w%u", dir_path, dim_idx, weight_file_counter-1);
            while (file_type(var_weight_file_path) == _REG_)
            {
                weight_file_counter++;
                sprintf(var_weight_file_path, "%s/%u/w%u", dir_path, dim_idx, weight_file_counter-1);
            }
            weight_file_counter--;
            if (rt_assert(weight_file_counter == (topology.size()-1)) == false)
            {
                return false;
            }

            // Now load the weights:
            for (uint weight_file_count=0; weight_file_count<weight_file_counter; ++weight_file_count)
            {
                sprintf(var_weight_file_path, "%s/%u/w%u", dir_path, dim_idx, weight_file_count);
                if (rt_assert(data_io.readMatrixFromFile(var_weight_file_path, *(weights[dim_idx][weight_file_count]))) == false)
                {
                    return false;
                }
            }

            // Count how many bias files are available:
            uint    bias_file_counter           = 1;
            char    var_bias_file_path[1000]    = "";
            sprintf(var_bias_file_path, "%s/%u/b%u", dir_path, dim_idx, bias_file_counter-1);
            while (file_type(var_bias_file_path) == _REG_)
            {
                bias_file_counter++;
                sprintf(var_bias_file_path, "%s/%u/b%u", dir_path, dim_idx, bias_file_counter-1);
            }
            bias_file_counter--;
            if (rt_assert(bias_file_counter == (topology.size()-2)) == false)
            {
                return false;
            }

            // Now load the biases:
            for (uint bias_file_count=0; bias_file_count<bias_file_counter; ++bias_file_count)
            {
                sprintf(var_bias_file_path, "%s/%u/b%u", dir_path, dim_idx, bias_file_count);
                if (rt_assert(data_io.readMatrixFromFile(var_bias_file_path, *(biases[dim_idx][bias_file_count]))) == false)
                {
                    return false;
                }
            }
        }

        // post-condition(s) checking
        return (this->isValid());
    }

    /**
     * MUST BE REAL-TIME!!!
     *
     * Compute prediction, based on given input and phase kernel modulator.
     *
     * @param input Input vector
     * @param phase_kernel_modulation (Input) vector of phase kernel modulation
     * @param output Output vector
     * @param start_index (Optional) selected start index of the dimensionality of the prediction
     * @param end_index (Optional) selected end index of the dimensionality of the prediction
     * @return Success or failure
     */
    bool PMNN::computePrediction(const VectorNN_N& input,
                                 const VectorNN_N& phase_kernel_modulation,
                                 VectorNN_N& output,
                                 int start_index, int end_index)
    {
        // pre-condition(s) checking
        if (rt_assert(this->isValid()) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(input.rows() == topology[0])) &&
                      (rt_assert(phase_kernel_modulation.rows() == topology[topology.size()-2])) &&
                      (rt_assert(output.rows()  == num_dimensions)) &&
                      (rt_assert(start_index >= 0)) &&
                      (rt_assert((end_index < num_dimensions) || (end_index == -1))) &&
                      (rt_assert((start_index <= end_index) || (end_index == -1)))) == false)   // because output layer does NOT have biases
        {
            return false;
        }

        if (end_index == -1)
        {
            end_index   = num_dimensions - 1;
        }

        MatrixNN_NxN    intermediate_layer_input(1, topology[0]);
        MatrixNN_NxN    intermediate_layer_output(1, topology[1]);
        for (uint dim_idx = start_index; dim_idx < (end_index+1); ++dim_idx)
        {
            intermediate_layer_input        = input.transpose();
            for (uint l=0; l<topology.size()-1; ++l)
            {
                intermediate_layer_output.resize(1, topology[l+1]);
                intermediate_layer_output   = intermediate_layer_input * *(weights[dim_idx][l]);
                if (l < topology.size()-2)
                {
                    intermediate_layer_output   = intermediate_layer_output + *(biases[dim_idx][l]);
                }
                if (l < topology.size()-3)
                {
                    for (uint ilo_idx=0; ilo_idx<intermediate_layer_output.cols(); ++ilo_idx)
                    {
                        intermediate_layer_output(0,ilo_idx)    = tanh(intermediate_layer_output(0,ilo_idx));
                    }
                }
                if (l == topology.size()-3)
                {
                    intermediate_layer_output   = intermediate_layer_output.cwiseProduct(phase_kernel_modulation.transpose());
                }
                if (l < topology.size()-2)
                {
                    intermediate_layer_input.resize(1, topology[l+1]);
                    intermediate_layer_input    = intermediate_layer_output;
                }
            }
            output(dim_idx, 0)  = intermediate_layer_output(0,0);
        }

        // post-condition(s) checking
        return (this->isValid());
    }

    PMNN::~PMNN()
    {}

}