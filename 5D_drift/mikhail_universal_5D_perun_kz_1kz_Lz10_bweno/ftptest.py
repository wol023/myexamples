#!/usr/bin/env python
import base64
import pysftp
import os
cnopts = pysftp.CnOpts()
cnopts.hostkeys = None



with open('perun_cluster_target_small.txt','r') as fo:
    lines= fo.read().splitlines()
host=lines[0]
user=lines[1]
pword=lines[2]
basepath=lines[3]
targetpath=lines[4]
nextline=lines[5]
pathfiles=lines[6:]

print host, user, pword, basepath, targetpath
print nextline
print pathfiles
paths=[]
files=[]

for i in range(len(pathfiles)):
    head=os.path.split(pathfiles[i])
    paths.append(head[0])
    files.append(head[1])
print paths
print files

homedir=os.getcwd()

def printTotals(transferred, toBeTransferred):
    #print "Transferred: {0}\tOut of : {1}".format(transferred, toBeTransferred)
    print "Transferred: {:5.2f} %\r".format(float(transferred)/toBeTransferred*100),


with pysftp.Connection(host, username=user, password=base64.b64decode(pword),cnopts=cnopts) as sftp:
    with sftp.cd(basepath):
        with sftp.cd(targetpath):
            #print sftp.listdir()
            for i in range(len(files)):
                print i,paths[i], files[i] 
                if not os.path.exists(paths[i]):
                    os.mkdir(paths[i])
                    print paths[i]+'/', 'is created.'
                else:
                    print paths[i]+'/', 'already exists.'
                currentdirlist=os.listdir(paths[i])
                if files[i] in currentdirlist:
                    print files[i], 'is found.'
                else:
                    print files[i], 'is NOT found.. start downloading.'
                    os.chdir(paths[i])
                    sftp.get(pathfiles[i], preserve_mtime=True,callback=printTotals)
                    print files[i], 'download completed.'
                    os.chdir(homedir)






with pysftp.Connection('cori.nersc.gov', username='wol023', password=base64.b64decode('UHNtNDkwNiEh') ,cnopts=cnopts) as sftp:
    with sftp.cd('./linkToCoriScratch1.ln'):            
        #sftp.put('/my/local/filename')  # upload file to public/ on remote
        with sftp.cd('./eslexamples/myexamples'):
            print sftp.listdir()

            sftp.get('visitlog.py') 

