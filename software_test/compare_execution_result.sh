#!/bin/bash

if [[ $(diff <($1 -e ../software_test/test_rt_err.txt) ../software_test/$2) ]]; then
	echo "ERROR: Unmatched execution result of command: $1"
	exit 1
fi

if [[ -s ../software_test/test_rt_err.txt ]] ; then
	echo "ERROR: Error occurs when executing command  : $1 ; the error is:"
	cat ../software_test/test_rt_err.txt
	exit 1
fi
