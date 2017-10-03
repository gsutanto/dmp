#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>

#include "amd_clmc_dmp/neural_nets/FFNNFinalPhaseLWRLayerPerDims.h"
#include "amd_clmc_dmp/paths.h"

using namespace dmp;

void print_usage()
{
    printf("Usage: amd_clmc_dmp_ffnn_final_phase_lwr_per_dims_demo [-e rt_err_file_path]\n");
}

int main(int argc, char** argv)
{

    bool                is_real_time= false;

    std::vector< uint > topology(4);
    topology[0]         = 45;
    topology[1]         = 100;
    topology[2]         = 25;
    topology[3]         = 1;

    char    rt_err_file_path[1000];
    get_rt_errors_path(rt_err_file_path,"/rt_err.txt");

    static  struct  option  long_options[]          = {
        {"rt_err_file_path",    required_argument, 0,  'e' },
        {0,                     0,                 0,  0   }
    };

    int     opt                     = 0;
    int     long_index              = 0;
    while ((opt = getopt_long(argc, argv,"e:", long_options, &long_index )) != -1)
    {
        switch (opt)
        {
            case 'e'    : strcpy(rt_err_file_path, optarg);
                break;
            default: print_usage();
                exit(EXIT_FAILURE);
        }
    }

    if (strcmp(rt_err_file_path, "") == 0)
    {
        print_usage();
        exit(EXIT_FAILURE);
    }

    // Initialize real-time assertor
    RealTimeAssertor                rt_assertor(rt_err_file_path);
    rt_assertor.clear_rt_err_file();

    DataIO                          data_io(&rt_assertor);

    FFNNFinalPhaseLWRLayerPerDims   ffnn_phaselwr(6, topology, &rt_assertor);

    if (rt_assert_main(ffnn_phaselwr.loadParams(get_data_path("/dmp_coupling/learn_tactile_feedback/scraping/neural_nets/FFNNFinalPhaseLWRLayerPerDims/cpp_models/prim1/").c_str(),
                                                0)) == false)
    {
        return (-1);
    }

    MatrixXxXPtr                    X;
    if (rt_assert_main(data_io.readMatrixFromFile(get_data_path("/dmp_coupling/learn_tactile_feedback/scraping/neural_nets/FFNNFinalPhaseLWRLayerPerDims/unroll_test_dataset/test_unroll_prim_1_X_raw_scraping.txt").c_str(), X)) == false)
    {
        return (-1);
    }

    MatrixXxXPtr                    normalized_phase_PSI_mult_phase_V;
    if (rt_assert_main(data_io.readMatrixFromFile(get_data_path("/dmp_coupling/learn_tactile_feedback/scraping/neural_nets/FFNNFinalPhaseLWRLayerPerDims/unroll_test_dataset/test_unroll_prim_1_normalized_phase_PSI_mult_phase_V_scraping.txt").c_str(), normalized_phase_PSI_mult_phase_V)) == false)
    {
        return (-1);
    }

    uint Nri    = X->rows();
    uint Nci    = X->cols();
    uint Nrp    = normalized_phase_PSI_mult_phase_V->rows();
    uint Ncp    = normalized_phase_PSI_mult_phase_V->cols();
    if (rt_assert_main(Nri == Nrp) == false)
    {
        return (-1);
    }

    VectorNN_N  input(Nci), phase_kernel_modulation(Ncp), output(6);
    input                   = ZeroVectorNN_N(Nci);
    phase_kernel_modulation = ZeroVectorNN_N(Ncp);
    output                  = ZeroVectorNN_N(6);
    for (uint t=0; t<Nri; ++t)
    {
        input                   = X->block(t, 0, 1, Nci).transpose();
        phase_kernel_modulation = normalized_phase_PSI_mult_phase_V->block(t, 0, 1, Ncp).transpose();
        if (rt_assert_main(ffnn_phaselwr.computePrediction(input,
                                                           phase_kernel_modulation,
                                                           output)) == false)
        {
            printf("Failed making prediction!");
            return (-1);
        }

        printf("%.05f %.05f %.05f %.05f %.05f %.05f\n",
               output(0,0), output(1,0), output(2,0), output(3,0), output(4,0), output(5,0));
    }

	return 0;
}
