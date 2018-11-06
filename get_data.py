# -*- coding: utf-8 -*-
# Author Yunguan 'Jake' Wang 
# 07/16/2018
"""
Read a table from synapse, common formats such as txt, csv or excel
"""
import synapseclient,os
import pandas as pd
from dropbox import Dropbox
import pandas as pd
from io import StringIO

def read_file(fn, filecontent = None):
	if fn[-3:] == 'csv':
		reader = pd.read_csv
	elif fn[-3:] in ['txt','tsv']:
		reader = pd.read_table
	elif fn[-4:] == 'xlsx':
		reader = pd.read_excel
	else:
		raise KeyboardInterrupt	('File type is not supported!')
		return
	if filecontent is None:
		return reader(fn)
	else:
		return reader(filecontent)

def read_synapse_file(synid):
	syn = synapseclient.Synapse()
	try:
		syn.login()
	except:
		username = input('User?')
		password = input('Password?')
		os.system( 'cls' )
		syn.login(username,password,rememberMe=True) # better not sharing my password like that!!!
	
	# Obtain a pointer and download the data
	syn_file = syn.get(synid)
	# Get the path to the local copy of the data file
	filepath = syn_file.path
	return filepath

def read_dbx_file(dbx_file_path):
    dbx_oauth = 'xXh2jnMU2-EAAAAAAAAZENzE7qh9xg_t45npARYvQY4sifKDsUWdq6R3_x2Lco2e'
    # dbx_oauth = input('Paster your HMS dropbox OAuth here: ')
    dbx = Dropbox(dbx_oauth)
    _,dbx_file = dbx.files_download(dbx_file_path)
    return StringIO(dbx_file.text)

def file2frame(filename):
	local_filename = filename
	filecontent = None
	if not os.path.exists(filename):
		if filename[:3] == 'syn':
			local_filename = read_synapse_file(filename)
		else:
			filecontent = read_dbx_file(filename)
	data = read_file(local_filename, filecontent)
	return data

if __name__ == "__main__":
	syn_example = 'syn7066916'
	dbx_example = '/hms cycif/test3195.csv'
	local_example = 'd:/Dropbox (Personal)/HMS Cycif/test3195.csv'
	for file in [syn_example,dbx_example,local_example]:
		data = file2frame(file)
		print(file)
		print(data.iloc[:5,:5])