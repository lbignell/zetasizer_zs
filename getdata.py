# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:52:28 2016

@author: lbignell
"""

import numpy as np
import datetime
from collections import Counter # Counter counts the number of occurrences of each item
import re

class sizemeasurements:
    '''
    Zetasizer size measurements object.
    '''
    def __init__(self, fname):
        '''
        Initialize the object, where fname is the measurement session exported
        using the SizeDataSimple template.
        '''
        thefile = open(fname, 'r')
        headerline = thefile.readline()
        firstline = thefile.readline()
        firstlinedata = firstline.split('\t')
        self.SWversion = firstlinedata[2]
        self.serialnum = firstlinedata[3]
        thefile.close()
        self.meastypes = np.genfromtxt(fname, delimiter='\t', dtype=str,
                                       usecols=(0), skip_header=1)
        self.samplenames_duplicates = np.genfromtxt(fname, delimiter='\t', dtype=str,
                                                    usecols=(1), skip_header=1)
        self.samplenames = self.removeduplicates(list(self.samplenames_duplicates))
        self.nsamples = len(self.samplenames)
        print('{0} samples in this file'.format(self.nsamples))
        self.meastimes_str = np.genfromtxt(fname, delimiter='\t', dtype=str,
                                           usecols=(4), skip_header=1)
        self.meastimes = [self.dateconvert(i) for i in self.meastimes_str]
        self.measdata = np.genfromtxt(fname, delimiter='\t', names=True)
        colnames = self.measdata.dtype.names
        label = 'initializedlabel'
        for name in colnames:
            print('name = {0}'.format(name))
            if re.search('\d_', name) is not None and label not in name:
                label = re.split('\d_', name)[0]
                print('found vector: label = {0}'.format(label))
                itms = [itm for itm in colnames if label in colnames]
                print('items: {0}'.format(itms))
                val = {self.samplenames[i]: 
                            self.measdata[[itm for itm in colnames
                                if label in itm]][i]
                                    for i in range(self.nsamples)}                
                setattr(self, label, val)
            elif re.search('\d_', name) is None:
                print('found scalar: label = {0}'.format(name))
                setattr(self, name, self.measdata[name])
        return
    
    def dateconvert(self, datestr):
        return datetime.datetime.strptime(datestr, '%A, %B %d, %Y %I:%M:%S %p')
        
    def removeduplicates(self, mylist):
        counts = Counter(mylist) # so we have {'sample name':# occurrences}
        for name,num in counts.items():
            if num > 1: # ignore strings that only appear once
                for suffix in range(1, num + 1): # suffix starts at 1 and increases by 1 each time
                    mylist[mylist.index(name)] = name.replace(' ', '_') \
                                                + '_' + str(suffix)
        return mylist