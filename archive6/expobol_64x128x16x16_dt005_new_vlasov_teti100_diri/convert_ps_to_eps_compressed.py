import os
import fnmatch
import time

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
       for name in files:
           if fnmatch.fnmatch(name, pattern):
               result.append(os.path.join(root, name))
    return result

fname=find('*.ps', './')
for singlefile in fname:
    print 'found ps file',singlefile

for item_num in range(0,len(fname)):
    print ("ps2eps -f -l -B "+fname[item_num])
    os.system("ps2eps -f -l -B "+fname[item_num])
    print ("wait for process...")
    time.sleep(2)

    tempname=fname[item_num].replace('.ps','.eps')
    print ("epstopdf "+tempname)
    os.system("epstopdf "+tempname)
    print ("wait for process...")
    time.sleep(2)

    tempname=fname[item_num].replace('.ps','.pdf')
    print ("pdftops -eps "+tempname)
    os.system("pdftops -eps "+tempname)
    print ("wait for process...")
    time.sleep(2)




