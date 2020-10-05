#!/usr/bin/env python3
# TODO: add path to the application here if it isn't in the system path (note
#  that the path to the application can be added in the apache configuration...)
#  Note that the parent directory of the SIRNet and scripts directories should
#  also be in the path
import sys
sys.path.insert(0, '/opt/dashboard/Livid-About-COVID/')
sys.path.insert(0, '/opt/dashboard/dash_env/lib/python3.6/site-packages')
sys.path.insert(0, '/opt/dashboard/Livid-About-Covid/Dashboard/dash_wsgi/')
#raise ValueError(sys.path)

#import site
#site.addsitedir('/opt/dashboard/dash_env/lib/python3.6/site-packages')
#sys.path.insert(0, '/opt/dashboard/Livid-About-COVID/')

# Activate your virtual env
#activate_env = '/opt/dashboard/dash_env/bin/activate_this.py'
#execfile(activate_env, dict(__file__=activate_env))

# Do whatever one works for you, man
#raise ValueError(sys.path)
# from texas_dashboard import app as application
from Dashboard.texas_dashboard import server as application
