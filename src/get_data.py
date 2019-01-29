# -*- coding: utf-8 -*-
# Author Yunguan 'Jake' Wang
# 07/16/2018
"""
Read a table from synapse, common formats such as txt, csv or excel
"""
import synapseclient
import os
import pandas as pd
from dropbox import Dropbox
import pandas as pd
from io import StringIO


def read_file(fn):
    """Read local files, either csv, txt or excel.
    """
    if fn[-3:] == 'csv':
        reader = pd.read_csv
    elif fn[-3:] in ['txt', 'tsv']:
        reader = pd.read_table
    elif fn[-4:] == 'xlsx':
        reader = pd.read_excel
    else:
        raise ValueError('File type is not supported!')
    return reader(fn)


def read_synapse_file(synid):
    """Read Synapse file, need a synapse id as input.
    """
    syn = synapseclient.Synapse()
    try:
        syn.login()
    except:
        username = input('User?')
        password = input('Password?')
        os.system('cls')
        syn.login(username, password, rememberMe=True)

    # Obtain a pointer and download the data
    syn_file = syn.get(synid)
    # Get the path to the local copy of the data file
    filepath = syn_file.path
    return filepath
