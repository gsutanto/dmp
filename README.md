# Dynamic Movement Primitive

Source: https://github.com/gsutanto/dmp

A library for Dynamic Movement Primitive (DMP), a compact
learning representation of (time-scale-invariant)
behaviors/trajectories/time-series signals.

This codebase contains the implementation of DMP in C++, Python, and MATLAB.
The C++ implementation supports execution in a (hard) real-time control for robots.
The MATLAB implementation is not maintained and is only provided here for references.

## Acknowledgement

This codebase was developed with the generous support from
the Max Planck Institute for Intelligent Systems and
University of Southern California.

## Citation:

```
@phdthesis{sutanto2020phddissertation,
  title = {Leveraging Structure for Learning Robot Control and Reactive Planning},
  school = {University of Southern California},
  author = {Giovanni Sutanto},
  year = {2020},
}
```

## Download

```
git clone git@github.com:gsutanto/dmp.git
```

## C++ Prerequisite

Tested on C++17 on Ubuntu 20.04 LTS

Dependencies: CMake and Eigen C++ Linear Algebra library

```
sudo apt-get install cmake libeigen3-dev
```

## Python Prerequisite

Tested on Python 3.7 on Ubuntu 20.04 LTS

```
conda create --name dmp python=3.7
conda activate dmp
cd dmp/
pip install -e .
```

## Running C++ and Python Software Integration Tests:

```
cd dmp/
mkdir build
cd build/
cmake ..
make
../software_test/dmp_software_test.sh
```

There should be no errors and no failed assertions printed out,
only some informational print-outs. If there is, then it is possible that
the software has been changed, and it is not producing consistent results anymore.
Please check the software and try to return it to a consistent state,
or reconcile the differences.

## Python Examples Execution

Open Spyder IDE ( https://docs.spyder-ide.org ):

```
conda activate dmp
spyder &
```

1.  One-Dimensional DMP Test:

    Open the file `dmp/python/dmp_test/dmp_1D/dmp_1D_test.py`

    Execute (`Run`) the file inside the Spyder IDE.

    (it will generate some plots comparing the training/demonstration data (in dotted blue line)
     versus the unrolled trajectory from the learned DMP primitive (in solid green line),
     as well as some plots about the DMP basis function activations versus time
     and some plots about canonical state variables' trajectories)

2.  Cartesian DMP Test:

    ### Learning from Single Trajectory

    Open the file `dmp/python/dmp_test/cart_dmp/cart_coord_dmp/cart_coord_dmp_single_traj_training_test.py`

    Execute (`Run`) the file inside the Spyder IDE.

    (it will generate some plots comparing the training/demonstration data (in dotted blue line)
     versus the unrolled trajectory from the learned DMP primitive (in solid green line))

    ### Learning from Multiple Trajectories

    Open the file `dmp/python/dmp_test/cart_dmp/cart_coord_dmp/cart_coord_dmp_multi_traj_training_test.py`

    Execute (`Run`) the file inside the Spyder IDE.

    (it will generate some plots comparing the training/demonstration data (in dotted blue lines)
     versus the unrolled trajectory from the learned DMP primitive (in solid green line))

3.  Quaternion DMP Test:

    ### Learning from Single Trajectory

    Open the file `dmp/python/dmp_test/cart_dmp/quat_dmp/quat_dmp_single_traj_training_test.py`

    Execute (`Run`) the file inside the Spyder IDE.

    (it will generate some plots comparing the training/demonstration data (in dotted blue line)
     versus the unrolled trajectory from the learned DMP primitive (in solid green line))

    ### Learning from Multiple Trajectories

    Open the file `dmp/python/dmp_test/cart_dmp/quat_dmp/quat_dmp_multi_traj_training_test.py`

    Execute (`Run`) the file inside the Spyder IDE.

    (it will generate some plots comparing the training/demonstration data (in dotted blue lines)
     versus the unrolled trajectory from the learned DMP primitive (in solid green line))

## Remark

For codes under directory `dmp/python/dmp_coupling/` that depends on TensorFlow,
the latest version of TensorFlow that works is TensorFlow 1.15.0.
However, currently the software tests for these parts have not been revived yet.

## C++ Examples Execution

C++ implementation is HARD real-time safe for execution in a robot control loop.

After `Running C++ and Python Software Integration Tests`,
and still in `dmp/build/` directory:

1.  One-Dimensional DMP Test:

    ```
    ./dmp_1D_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r time_reproduce_max(>=0)] [-h time_goal_change(>=0)] [-g new_goal] [-t tau_reproduce(>=0)] [-e rt_err_file_path]
    ```

    (it will generate files and/or folders of files inside
     `../plot/dmp_1D/` directory,
     e.g. for plotting on MATLAB)

2.  Cartesian DMP Test:

    ### Learning from Single Trajectory

    ```
    ./dmp_cart_coord_dmp_single_traj_training_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r time_reproduce_max(>=0)] [-h time_goal_change(>=0)] [-t tau_reproduce(>=0)] [-e rt_err_file_path]
    ```

    (it will generate files and/or folders of files inside
     `../plot/cart_dmp/cart_coord_dmp/single_traj_training/` directory,
     e.g. for plotting on MATLAB)

    ### Learning from Multiple Trajectories

    ```
    ./dmp_cart_coord_dmp_multi_traj_training_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r time_reproduce_max(>=0)] [-t tau_reproduce(>=0)] [-e rt_err_file_path]
    ```

    (it will generate files and/or folders of files inside
     `../plot/cart_dmp/cart_coord_dmp/multi_traj_training/` directory,
     e.g. for plotting on MATLAB)
