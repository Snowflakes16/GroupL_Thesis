import sys
import os
import sqlite3
import db_framework as db


def init_db():
    colorid = ["RGB", "RBG", "BGR", "BRG", "GBR", "GRB", "RGR", "RBR", "BRB", "BGB"]
    for x in range(0, 10):
        db.insert_general(x, colorid[x])


def new_day(imgdt):
    for x in range(0, 10):
        db.insert_daily(x, imgdt, 6)

def new_hour(imgdt, imghr):
    for x in range(0, 10):
        db.insert_daily(x, imgdt, imghr)


#Main

def record(day, nhour, phour, ncoord, pcoord):


    #coodinates of the feeder area

    feeder_xleft = 100
    feeder_xright = 200
    feeder_ytop = 200
    feeder_ybottom = 100

    #coordinates of the drinker area

    drinker_xleft = 100
    drinker_xright = 200
    drinker_ytop = 200
    drinker_ybottom = 100

    if nhour > phour:
        new_hour(day, nhour)
        phour = nhour

    for id in range (0, 10):

        if (ncoord[0][id] == pcoord[0][id] and ncoord[1][id] == pcoord[1][id]):
            db.inc_inactive_t(id)
        else:
            pcoord[0][id] = ncoord[0][id]
            pcoord[1][id] = ncoord[1][id]

    for id in range (0,10):

        if(feeder_xleft < ncoord[0][id] < feeder_xright and feeder_ytop > ncoord[1][id] > feeder_ybottom):
            db.inc_hfvt(id, day, nhour)
        if(drinker_xleft < ncoord[0][id] < drinker_xright and drinker_ytop > ncoord[1][id] > drinker_ybottom):
            db.inc_hdvt(id, day, nhour)

    return phour







