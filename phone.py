import sqlite3
import numpy as np
conn=sqlite3.connect('phone.db') #create db
print("Database opened sucessfully")
cursor=conn.cursor()
cursor.execute("SELECT NAME FROM PHONE")
ph=cursor.fetchall()
print ("PHONES AVAILABLE")
for m in ph:
    print (m[0])
print("Search the phone for:")
print("1.Availablity & specification\n2.Color\n3.RAM\n4.CAMERA\n5.OS")
t=input("Enter the Number:")
t=int(t)
#availablity and specification blog
if t==1:
    jk=0
    arr=[]
    r=input("Enter the phone name which you want to search:")
    y=() #empty tuple
    y=y+(r,) #append
    cursor.execute("SELECT * FROM PHONE where NAME=?",y)
    res=cursor.fetchall() #fetch the values
    tt=len(res)
    #res=np.reshape(-1)
    #arr.flatten()
    #arr=np.reshape(res, (np.product(res.shape),))
    arr=list(res)
    length=len(res)
    if length==0:
       print("Phone not available")
    print("NAME  RAM(GB) CAMERA(MP) OS COLOR\n")   
    for i in range(0,len(arr[0])):
        print(arr[0][i])
        
#Color blog        
if t==2:        
    i=input("Enter the phone name to search which colour available:")
    inp=()
    inp=inp+(i,)
    cursor.execute("SELECT COLOUR FROM PHONE WHERE NAME=?",inp)
    n=cursor.fetchall()
    for s in n:
        print("COLOR AVAILABLE:")
        print(s[0])
#RAM blog        
if t==3:        
    k=input("Enter the phone name to search which RAM available:")
    l=()
    l=l+(k,)
    cursor.execute("SELECT RAM FROM PHONE WHERE NAME=?",l)
    cu=cursor.fetchall()
    for xx in cu:
        print("RAM AVAILABLE:")
        print(xx[0])
#CAMERA blog        
if t==4:        
    ii=input("Enter the phone name to search CAMERA MP available:")
    yy=()
    yy=yy+(ii,)
    cursor.execute("SELECT CAMERA FROM PHONE WHERE NAME=?",yy)
    nn=cursor.fetchall()
    for ss in nn:
        print("CAMERA MP:")
        print(ss[0])
#OS blog     
if t==5:        
    jj=input("Enter the phone name to search which os available:")
    pp=()
    pp=pp+(jj,)
    cursor.execute("SELECT OS FROM PHONE WHERE NAME=?",pp)
    mm=cursor.fetchall()
    for aa in mm:
        print("OS AVAILABLE:")
        print(aa[0])        
        
conn.close()
             
             
             
    
    
