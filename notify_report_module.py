import os
import sys
import db_framework as db


def write(file, towrite):
    notif = open(file, "w")
    notif.write(towrite)
    notif.close()

def read(file):
    notif = open(file, "r")
    print(notif.read())
    notif.close()


def append(file, towrite):
    notif = open(file, "a")
    notif.write(towrite)
    notif.close()


def notify():

    content = db.get_inactive_t()
    tagID = {0:"RGB", 1:"RBG", 2:"BGR", 3:"BRG", 4:"GBR", 5:"GRB", 6:"RGR", 7:"RBR", 8:"BRB", 9:"BGB"}

    for i in range (0, 10):
        if(int(content[i][0])>7200):
            write("notify.txt", "Tag Alert!: ")
            append("notify.txt", tagID[i])
            read("notify.txt")

def report(day):

    tagID = {0: " RGB ", 1: " RBG ", 2: " BGR ", 3: " BRG ", 4: " GBR ", 5: " GRB ", 6: " RGR ", 7: " RBR ", 8: " BRB ",
             9: " BGB "}

    repcont = "First Report  \n"
    for i in range (0, 5):

        content = db.get_report(i, day)
        repcont = repcont + "Pig #" + str(i) + tagID[i] + " : \n"
        hfvt = hdvt = ""

        for x in range (0, len(content[0])):
            hfvt += str(content[0][x][0]) + ", "
            hdvt += str(content[1][x][0]) + ", "

        repcont = repcont + "[" + hfvt + "]\n[" + hdvt + "]\n"

    write("report.txt", repcont)
    read("report.txt")

    repcont = "Second Report  \n"
    for i in range (5, 10):

        content = db.get_report(i, day)
        repcont = repcont + "Pig #" + str(i) + tagID[i] + " : \n"
        hfvt = hdvt = ""

        for x in range (0, len(content[0])):
            hfvt += str(content[0][x][0]) + ", "
            hdvt += str(content[1][x][0]) + ", "

        repcont = repcont + "[" + hfvt + "]\n[" + hdvt + "]\n"

    write("report.txt", repcont)
    read("report.txt")





