#include "dmp/dmp_param/DMPUnrollInitParams.h"

namespace dmp {

DMPUnrollInitParams::DMPUnrollInitParams() : tau(0.0), rt_assertor(NULL) {}

DMPUnrollInitParams::DMPUnrollInitParams(
    const double& tau_init, const DMPState& start_state,
    const DMPState& goal_state, RealTimeAssertor* real_time_assertor,
    bool is_zeroing_out_velocity_and_acceleration)
    : tau(tau_init), rt_assertor(real_time_assertor) {
  critical_states.resize(2);
  if (is_zeroing_out_velocity_and_acceleration) {
    critical_states[0] = DMPState(start_state.getX(), rt_assertor);
    critical_states[1] = DMPState(goal_state.getX(), rt_assertor);
  } else {
    critical_states[0] = start_state;
    critical_states[1] = goal_state;
  }
}

DMPUnrollInitParams::DMPUnrollInitParams(
    const double& tau_init, const Trajectory& trajectory,
    RealTimeAssertor* real_time_assertor,
    bool is_zeroing_out_velocity_and_acceleration)
    : tau(tau_init), rt_assertor(real_time_assertor) {
  uint traj_size = trajectory.size();

  if (traj_size >= 2) {
    if (traj_size >= 3) {
      critical_states.resize(3);
    } else {
      critical_states.resize(2);
    }

    if (critical_states.size() >= 1) {
      if (is_zeroing_out_velocity_and_acceleration) {
        critical_states[0] = DMPState(trajectory[0].getX(), rt_assertor);
      } else {
        critical_states[0] = trajectory[0];
      }
    }

    if (critical_states.size() >= 2) {
      if (is_zeroing_out_velocity_and_acceleration) {
        critical_states[critical_states.size() - 1] =
            DMPState(trajectory[traj_size - 1].getX(), rt_assertor);
      } else {
        critical_states[critical_states.size() - 1] = trajectory[traj_size - 1];
      }
    }

    if (critical_states.size() >= 3) {
      if (is_zeroing_out_velocity_and_acceleration) {
        critical_states[critical_states.size() - 2] =
            DMPState(trajectory[traj_size - 2].getX(), rt_assertor);
      } else {
        critical_states[critical_states.size() - 2] = trajectory[traj_size - 2];
      }
    }
  }
}

DMPUnrollInitParams::DMPUnrollInitParams(
    const Trajectory& trajectory, RealTimeAssertor* real_time_assertor,
    bool is_zeroing_out_velocity_and_acceleration)
    : rt_assertor(real_time_assertor) {
  uint traj_size = trajectory.size();

  if (traj_size >= 2) {
    // if the trajectory is containing the "correct" information of the
    // trajectory timing, then use this timing information to compute tau:
    tau = trajectory[traj_size - 1].getTime() - trajectory[0].getTime();

    if (traj_size >= 3) {
      critical_states.resize(3);
    } else {
      critical_states.resize(2);
    }

    if (critical_states.size() >= 1) {
      if (is_zeroing_out_velocity_and_acceleration) {
        critical_states[0] = DMPState(trajectory[0].getX(), rt_assertor);
      } else {
        critical_states[0] = trajectory[0];
      }
    }

    if (critical_states.size() >= 2) {
      if (is_zeroing_out_velocity_and_acceleration) {
        critical_states[critical_states.size() - 1] =
            DMPState(trajectory[traj_size - 1].getX(), rt_assertor);
      } else {
        critical_states[critical_states.size() - 1] = trajectory[traj_size - 1];
      }
    }

    if (critical_states.size() >= 3) {
      if (is_zeroing_out_velocity_and_acceleration) {
        critical_states[critical_states.size() - 2] =
            DMPState(trajectory[traj_size - 2].getX(), rt_assertor);
      } else {
        critical_states[critical_states.size() - 2] = trajectory[traj_size - 2];
      }
    }
  }
}

DMPUnrollInitParams::DMPUnrollInitParams(
    const Trajectory& trajectory, const double& robot_task_servo_rate,
    RealTimeAssertor* real_time_assertor,
    bool is_zeroing_out_velocity_and_acceleration)
    : rt_assertor(real_time_assertor) {
  uint traj_size = trajectory.size();

  if (traj_size >= 2) {
    // if the trajectory is containing the "correct" information of the
    // trajectory timing, then use this timing information to compute tau:
    tau = trajectory[traj_size - 1].getTime() - trajectory[0].getTime();
    // otherwise use the robot_task_servo_rate to compute tau
    // (assuming that the trajectory is sampled at robot_task_servo_rate):
    if (tau < MIN_TAU) {
      tau = (1.0 * (traj_size - 1)) / robot_task_servo_rate;
    }

    if (traj_size >= 3) {
      critical_states.resize(3);
    } else {
      critical_states.resize(2);
    }

    if (critical_states.size() >= 1) {
      if (is_zeroing_out_velocity_and_acceleration) {
        critical_states[0] = DMPState(trajectory[0].getX(), rt_assertor);
      } else {
        critical_states[0] = trajectory[0];
      }
    }

    if (critical_states.size() >= 2) {
      if (is_zeroing_out_velocity_and_acceleration) {
        critical_states[critical_states.size() - 1] =
            DMPState(trajectory[traj_size - 1].getX(), rt_assertor);
      } else {
        critical_states[critical_states.size() - 1] = trajectory[traj_size - 1];
      }
    }

    if (critical_states.size() >= 3) {
      if (is_zeroing_out_velocity_and_acceleration) {
        critical_states[critical_states.size() - 2] =
            DMPState(trajectory[traj_size - 2].getX(), rt_assertor);
      } else {
        critical_states[critical_states.size() - 2] = trajectory[traj_size - 2];
      }
    }
  }
}

bool DMPUnrollInitParams::isValid() const {
  if (rt_assert(tau >= MIN_TAU) == false) {
    return false;
  }
  if (rt_assert(critical_states.size() >= 2) == false) {
    return false;
  }
  for (uint i = 0; i < critical_states.size(); i++) {
    if (rt_assert(critical_states[i].isValid()) == false) {
      return false;
    }
  }
  return true;
}

DMPUnrollInitParams::~DMPUnrollInitParams() {}

}  // namespace dmp
