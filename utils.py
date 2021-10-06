"""
File:    utils.py
Created: December 2, 2019
Revised: December 2, 2019
Authors: Howard Heaton, Xiaohan Chen
Purpose: Definition of different utility functions used in other files, e.g.,
         logging handler.
"""

import logging

def setup_logger(log_file):
    if log_file is not None:
        logging.basicConfig(filename=log_file, level=logging.INFO)
        lgr = logging.getLogger()
        lgr.addHandler(logging.StreamHandler())
        lgr = lgr.info
    else:
        lgr = print

    return lgr

