#!/bin/bash

$1 -e ../software_test/test_rt_err.txt > ../software_test/$2

if [[ -s ../software_test/test_rt_err.txt ]] ; then
	echo "ERROR: Error occurs when executing command  : $1 ; the error is:"
	cat ../software_test/test_rt_err.txt
	exit 1
fi
