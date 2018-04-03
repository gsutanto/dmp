'''
Created on Dec 12, 2011

@author: righetti
'''

import os
import numpy
import struct

class ClmcFile:
    #defines cols, rows, freq, names and units
    def __init__(self, filename=''):
        if os.path.exists(filename) == False:
            print 'Error the file - ', filename, ' - does not exist'
            return
        
        #read the file
        with open(filename, 'rb') as my_file:
            #get the header right
            temp = my_file.readline().split()
            cols = int(temp[1])
            rows = int(temp[2])
            self.freq = float(temp[3])
    
            #get the names and units
            self.names = {}
            self.units = []
            temp = my_file.readline().split()
            for i in range(0, cols) :
                self.names[(temp[2*i])] = i
                self.units.append(temp[2*i+1])
            #get all of the data
            self.data = numpy.array(struct.unpack('>'+'f'*cols*rows,my_file.read(4*cols*rows))).reshape(rows, cols).transpose()
                    
        
    def get_variables(self, names):
        #call this function with a list of names and it will return a list of numpy arrays
        #containing the desired data
        #if names is a string - we return just the associated vector
        
        #check if it asks only for one name - names is a string
        if(isinstance(names, str)):
            if names in self.names:
                return self.data[self.names[names]]
            else:
                print 'ERROR ', names, ' is not a valid name'
        else:    
            result = []
            for item in names:
                if item in self.names:
                    result.append(self.data[self.names[item]])
                else:
                    print 'ERROR ', item, ' is not a valid name'
                    break
            return result
