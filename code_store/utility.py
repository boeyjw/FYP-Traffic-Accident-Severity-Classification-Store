# Utility

ta_missing = {
    'acc': acc.isnull().any(),
    'cas': cas.isnull().any(),
    'veh': veh.isnull().any()
}


"""
=====================================================
"""
import shelve
filename = 'tap_alldata'
# Save workspace vars
my_shelf = shelve.open('tap_alldata', 'n')
for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except:
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()

import shelve
filename = 'tap_alldata'
# Load workspace vars
my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()
"""
=====================================================
"""

