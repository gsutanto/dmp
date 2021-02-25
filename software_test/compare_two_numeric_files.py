#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
dmp_home_dir_abs_path = os.path.dirname(__file__) + "/../"
sys.path.append(os.path.join(dmp_home_dir_abs_path + '/python/utilities/'))
from utilities import compareTwoNumericFiles


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-f1", "--file1_path", type=str)
parser.add_argument("-f2", "--file2_path", type=str)


if __name__ == '__main__':
    args = parser.parse_args()
    file1_path = args.file1_path
    file2_path = args.file2_path
    assert(os.path.exists(file1_path)), 'file1: %s does NOT exist!' % file1_path
    assert(os.path.exists(file2_path)), 'file2: %s does NOT exist!' % file2_path
    compareTwoNumericFiles(file1_path, file2_path)
