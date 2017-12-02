import os
import sys
import record as rec
import db_framework as db
import notify_report_module as sms


db.init_Tables()
#rec.init_db()


nXcoord = [120,0,0,0,0,0,0,0,0,0]
nYcoord = [120,0,0,0,0,0,0,0,0,0]

pXcoord = [0,0,0,0,0,0,0,0,0,0]
pYcoord = [0,0,0,0,0,0,0,0,0,0]


newcoord = [nXcoord, nYcoord]
prevcoord = [pXcoord, pYcoord]

phour = 10
nhour = 10
day = 1

#newcoord = img.imgproc()

phour = rec.record(day, nhour, phour, newcoord, prevcoord)
if(phour != 10):
    sms.notify()
else:
    sms.report(day)


db.view_general()
db.view_daily()