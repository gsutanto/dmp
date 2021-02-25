#!/bin/bash

../software_test/execute_cpp_test.sh "$1" $2
python ../software_test/compare_two_numeric_files.py -f1 ../software_test/$2 -f2 ../software_test/$3
