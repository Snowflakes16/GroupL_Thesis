import sys
import os
import sqlite3

'''
Author: Japhet Edilvir Undag 
Name: Database Framework
'''



conn = sqlite3.connect("sample1.db")
c = conn.cursor()

def init_Tables():
    c.execute('''CREATE TABLE IF NOT EXISTS Pig_General(
            id INTEGER, 
            colorid TEXT, 
            inactive_t INTEGER, 
            PRIMARY KEY(id),
            UNIQUE (colorid))
            ''')
    conn.commit()
    c.execute('''CREATE TABLE IF NOT EXISTS Pig_Daily(
            id INTEGER, 
            imgdt TEXT, 
            imghr INTEGER,
            hfvt INTEGER, 
            hdvt INTEGER, 
            PRIMARY KEY (id, imgdt, imghr),
            FOREIGN KEY(id) REFERENCES Pig_General(id))''')
    conn.commit()

def insert_general(id, colorid):
    c.execute("INSERT INTO Pig_General values(?,?,0)", (id, colorid))
    conn.commit()

def insert_daily(id, imgdt, imghr):
    c.execute("INSERT INTO Pig_Daily values(?,?,?,0,0)", (id, imgdt, imghr))
    conn.commit()


def inc_inactive_t(id):
    c.execute('''UPDATE Pig_General SET inactive_t = inactive_t + 1 
    WHERE id = :id''', {'id': id})
    conn.commit()


def reset_inactive_t(id):
    c.execute('''UPDATE Pig_General SET inactive_t = 0 
    WHERE id = :id''', {'id': id})
    conn.commit()


def inc_hfvt(id, imgdt, imghr):
    c.execute('''UPDATE Pig_Daily SET hfvt = hfvt+1 
    WHERE id = :id AND imgdt = :imgdt AND imghr = :imghr''', {'id':id, 'imgdt':imgdt, 'imghr':imghr})
    conn.commit()


def inc_hdvt(id, imgdt, imghr):
    c.execute('''UPDATE Pig_Daily SET hdvt = hdvt+1 
    WHERE id = :id AND imgdt = :imgdt AND imghr = :imghr''', {'id': id, 'imgdt': imgdt, 'imghr': imghr})
    conn.commit()


def view_general():
    c.execute('''SELECT * FROM Pig_General''')
    content = c.fetchall()
    print("General Info Table \n")
    i=0
    while (i < len(content)):
        print(content[i], '\n')
        i += 1


def view_daily():
    c.execute('''SELECT * FROM Pig_Daily''')
    content = c.fetchall()
    print("Daily Info Table \n")
    i = 0
    while (i < len(content)):
        print(content[i], '\n')
        i += 1


def del_daily_table():
    c.execute("DROP TABLE IF EXISTS Pig_Daily")
    conn.commit()


def del_general_table():
    c.execute("DROP TABLE IF EXISTS Pig_General")
    conn.commit()

def remove_daily(id, imgdt, imghr):
    c.execute('''DELETE from Pig_Daily  
    WHERE id = :id AND imgdt = :imgdt AND imghr = :imghr''', {'id': id, 'imgdt': imgdt, 'imghr': imghr})
    conn.commit()


def remove_general(id):
    c.execute('''DELETE from Pig_General  
    WHERE id = :id''', {'id': id})
    conn.commit()

def select_report(id, imgdt):
    c.execute('''SELECT * FROM Pig_Daily WHERE id = :id AND imgdt = :imgdt ''', {'id': id, 'imgdt':imgdt})
    content = c.fetchall()

    print("Report \n")
    i = 0
    while (i < len(content)):
        print(content[i], '\n')
        i += 1
def get_inactive_t():
    c.execute('''SELECT inactive_t FROM Pig_General''')
    content = c.fetchall()

    return content

def get_report(id, day):

    c.execute('''SELECT hfvt FROM Pig_Daily WHERE id = :id AND imgdt = :day''', {"id":id, "day":day})
    hfvt = c.fetchall()
    c.execute('''SELECT hdvt FROM Pig_Daily WHERE id = :id AND imgdt = :day''', {"id":id, "day":day})
    hdvt = c.fetchall()
    content = [hfvt,hdvt]

    return content








