/*
 * QuaternionDMP.h
 *
 *  Class definition for Quaternion DMP.
 *
 *  Created on: June 20, 2017
 *  Author: Giovanni Sutanto
 */

#ifndef QUATERNION_DMP_H
#define QUATERNION_DMP_H

#include <boost/filesystem.hpp>

#include "amd_clmc_dmp/dmp_discrete/DMPDiscrete.h"
#include "amd_clmc_dmp/cart_dmp/quat_dmp/TransformSystemQuaternion.h"

namespace dmp
{

    class QuaternionDMP : public DMPDiscrete
    {

    protected:

        TransformSystemQuaternion   transform_sys_discrete_quat;

    public:

        /**
         * NEVER RUN IN REAL-TIME, ONLY RUN IN INIT ROUTINE
         */
        QuaternionDMP();

        /**
         * Original formulation by Schaal, AMAM 2003 (default formulation type) is selected.
         *
         * @param model_size_init Size of the model (M: number of basis functions or others) used to represent the function
         * @param canonical_system_discrete (Pointer to) discrete canonical system that drives this DMP
         * @param learning_method Learning method that will be used to learn the DMP basis functions' weights\n
         *        0=_SCHAAL_LWR_METHOD_: Stefan Schaal's Locally-Weighted Regression (LWR): Ijspeert et al., Neural Comput. 25, 2 (Feb 2013), p328-373\n
         *        1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_: Jan Peters' Global Least Squares method
         * @param real_time_assertor Real-Time Assertor for troubleshooting and debugging
         * @param opt_data_directory_path [optional] Full directory path where to save/log all the data
         * @param transform_couplers All spatial couplings associated with the discrete transformation system
         * (please note that this is a pointer to vector of pointer to TransformCoupling data structure, for flexibility and re-usability)
         */
        QuaternionDMP(uint model_size_init, CanonicalSystemDiscrete* canonical_system_discrete,
                      uint learning_method, RealTimeAssertor* real_time_assertor,
                      const char* opt_data_directory_path="",
                      std::vector<TransformCoupling*>* transform_couplers=NULL);

        /**
         * Checks whether this Quaternion DMP is valid or not.
         *
         * @return Quaternion DMP is valid (true) or Quaternion DMP is invalid (false)
         */
        bool isValid();

        /**
         * Resets DMP and sets all needed parameters (including the reference coordinate frame) to start execution.
         * Call this function before beginning any DMP execution.
         *
         * @param critical_states Critical states defined for DMP trajectory reproduction/unrolling.\n
         *                        Start state is the first state in this trajectory/vector of DMPStates.\n
         *                        Goal state is the last state in this trajectory/vector of DMPStates.
         * @param tau_init Initialization for tau (time parameter that influences the speed of the execution; usually this is the time-length of the trajectory)
         * @return Success or failure
         */
        bool startQuaternionDMP(const QuaternionTrajectory& critical_states, double tau_init);

        /**
         * Runs one step of the DMP based on the time step dt.
         *
         * @param dt Time difference between the previous and the current step
         * @param update_canonical_state Set to true if the canonical state should be updated here.\n
         *        Otherwise you are responsible for updating it yourself.
         * @param next_state Computed next DMPState
         * @param transform_sys_forcing_term (optional) If you also want to know \n
         *                                   the transformation system's forcing term value of the next state, \n
         *                                   please put the pointer here
         * @param transform_sys_coupling_term_acc (optional) If you also want to know \n
         *                                        the acceleration-level coupling term value \n
         *                                        of the next state, please put the pointer here
         * @param transform_sys_coupling_term_vel (optional) If you also want to know \n
         *                                        the velocity-level coupling term value \n
         *                                        of the next state, please put the pointer here
         * @param func_approx_basis_functions (optional) (Pointer to) vector of basis functions constructing the forcing term
         * @return Success or failure
         */
        bool getNextQuaternionState(double dt, bool update_canonical_state,
                                    QuaternionDMPState& next_state, VectorN* transform_sys_forcing_term=NULL,
                                    VectorN* transform_sys_coupling_term_acc=NULL,
                                    VectorN* transform_sys_coupling_term_vel=NULL,
                                    VectorM* func_approx_basis_functions=NULL);

        /**
         * Returns current Quaternion DMP state.
         */
        QuaternionDMPState getCurrentQuaternionState();

        /**
         * Based on the start state, current state, evolving goal state and steady-state goal position,
         * compute the target value of the coupling term, and then update the canonical state and goal state.
         * Needed for learning the coupling term function based on a given sequence of DMP states (trajectory).
         * [IMPORTANT]: Before calling this function, the TransformationSystem's start_state, current_state, and
         * goal_sys need to be initialized first using TransformationSystem::start() function,
         * and the forcing term weights need to have already been learned beforehand,
         * using the demonstrated trajectory.
         *
         * @param current_state_demo_local Demonstrated current DMP state in the local coordinate system \n
         *                                 (if there are coordinate transformations)
         * @param dt Time difference between the current and the previous execution
         * @param ct_acc_target Target value of the acceleration-level coupling term at
         *                      the demonstrated current state,\n
         *                      given the demonstrated start state, evolving goal state, \n
         *                      demonstrated steady-state goal position, and
         *                      the learned forcing term weights \n
         *                      (to be computed and returned by this function)
         * @return Success or failure
         */
        bool getTargetQuaternionCouplingTermAndUpdateStates(const QuaternionDMPState& current_state_demo_local,
                                                            const double& dt,
                                                            VectorN& ct_acc_target);

        ~QuaternionDMP();

    private:

        /**
         * NON-REAL-TIME!!!\n
         * Extract a set of trajectory(ies) from text file(s). Implementation is in derived classes.
         *
         * @param dir_or_file_path Two possibilities:\n
         *        If want to extract a set of trajectories, \n
         *        then specify the directory path containing the text files, \n
         *        each text file represents a single trajectory.
         *        If want to extract just a single trajectory, \n
         *        then specify the file path to the text file representing such trajectory.
         * @param trajectory_set (Blank/empty) set of trajectories to be filled-in
         * @param is_comma_separated If true, trajectory fields are separated by commas;\n
         *        If false, trajectory fields are separated by white-space characters (default is false)
         * @param max_num_trajs [optional] Maximum number of trajectories that could be loaded into a trajectory set (default is 500)
         * @return Success or failure
         */
        bool extractSetTrajectories(const char* dir_or_file_path, TrajectorySet& trajectory_set,
                                    bool is_comma_separated=false, uint max_num_trajs=500);

        // hide some methods:
        using DMP::start;

        using DMPDiscrete::getNextState;
        using DMPDiscrete::getCurrentState;
        using DMPDiscrete::getTargetCouplingTermAndUpdateStates;

    };
}
#endif