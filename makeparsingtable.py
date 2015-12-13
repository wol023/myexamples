
import os

def remove_comments(line, sep):
    for s in sep:
        line = line.split(s)[0]
    return line.strip()

# Set the directory you want to start from
rootDir = '/home/wonjae/ws/myexamples'
#rootDir = './testdir'
parsingtable = []
filecnt = 0
parsingtable.extend(['AAA;Dir'])
parsingtable.extend(['AAA;File'])
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName)
    for fname in fileList:
        if fname.endswith(".in"):
            print('Found directory: %s' % dirName)
            print('\t%s' % fname)
            dirstrip=os.path.basename(dirName)
            print '\t'+dirstrip
            filecnt=filecnt+1
            with open(dirName+'/'+fname, 'r') as f:
                #parse simulation
                for line in f:
                    if line.lstrip().startswith('#'): #skip comment
                        continue
                    line =line.rstrip() #skip blank line
                    if not line:
                        continue 
                    else: #noncomment line
                        strippedline=remove_comments(line, '#') #remove trailing comment
                        lhsrhs = strippedline.split("=")
                        l=0
                        while l<len(lhsrhs): #strip white spaces in lhs
                            lhsrhs[l]=lhsrhs[l].rstrip()
                            lhsrhs[l]=lhsrhs[l].lstrip()
                            l=l+1
                        splittedlhs=lhsrhs[0].split('.') #split lhs
                        temptable=lhsrhs[0].replace('.',';') #replace comma to semicolon
                        if not temptable in parsingtable:
                            parsingtable.extend([temptable])

            f.closed
#sortedparsingtable=sorted(parsingtable,reverse=True)
sortedparsingtable=sorted(parsingtable)
#for pl in sortedparsingtable:
#    print pl
#Done parsing table
#transpose parsing table after fitting maximum fields
cntsemicolons=0
for pl in sortedparsingtable:
    tempcntsemicolons=pl.count(';')
    if tempcntsemicolons > cntsemicolons:
        cntsemicolons=tempcntsemicolons
t_rows = cntsemicolons+1
t_cols = len(sortedparsingtable)
t_sortedparsingtable = [ (['-'] * t_cols) for row in xrange(t_rows) ]

for c in xrange(t_cols):
    pl=sortedparsingtable[c]
    lhstable = pl.split(";")
    for r in xrange(len(lhstable)):
        t_sortedparsingtable[r][c]=lhstable[r]




#parsingvalue = []
rows = len(sortedparsingtable)
cols = filecnt
parsingvalue = [ (['UNDEF'] * cols) for row in xrange(rows) ]
t_parsingvalue = [ (['UNDEF'] * rows) for col in xrange(cols) ]

filecnt=0
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName)
    for fname in fileList:
        if fname.endswith(".in"):
            #print('Found directory: %s' % dirName)
            #print('\t%s' % fname)
            dirstrip=os.path.basename(dirName)
            #print '\t'+dirstrip
            filecnt=filecnt+1
            parsingvalue[sortedparsingtable.index('AAA;Dir')][filecnt-1]=dirstrip
            parsingvalue[sortedparsingtable.index('AAA;File')][filecnt-1]=fname
            t_parsingvalue[filecnt-1][sortedparsingtable.index('AAA;Dir')]=dirstrip
            t_parsingvalue[filecnt-1][sortedparsingtable.index('AAA;File')]=fname
            with open(dirName+'/'+fname, 'r') as f:
                #parse simulation
                for line in f:
                    if line.lstrip().startswith('#'): #skip comment
                        continue
                    line =line.rstrip() #skip blank line
                    if not line:
                        continue 
                    else: #noncomment line
                        strippedline=remove_comments(line, '#') #remove trailing comment
                        lhsrhs = strippedline.split("=")
                        l=0
                        while l<len(lhsrhs): #strip white spaces in lhs
                            lhsrhs[l]=lhsrhs[l].rstrip()
                            lhsrhs[l]=lhsrhs[l].lstrip()
                            l=l+1
                        splittedlhs=lhsrhs[0].split('.') #split lhs
                        temptable=lhsrhs[0].replace('.',';') #replace comma to semicolon
                        parsingvalue[sortedparsingtable.index(temptable)][filecnt-1]=lhsrhs[len(lhsrhs)-1]
                        t_parsingvalue[filecnt-1][sortedparsingtable.index(temptable)]=lhsrhs[len(lhsrhs)-1]

            f.closed
        #filecnt=filecnt+1





def maxItemLength(a):
    maxLen = 0
    rows = len(a)
    cols = len(a[0])
    for row in xrange(rows):
        for col in xrange(cols):
            maxLen = max(maxLen, len(str(a[row][col])))
    return maxLen

def print2dList(a):
    if (a == []):
        # So we don't crash accessing a[0]
        print []
        return
    rows = len(a)
    cols = len(a[0])
    fieldWidth = maxItemLength(a)
    print "[ ",
    for row in xrange(rows):
        if (row > 0): print "\n  ",
        print "[ ",
        for col in xrange(cols):
            if (col > 0): print ",",
            # The next 2 lines print a[row][col] with the given fieldWidth
            format = "%" + str(fieldWidth) + "s"
            print format % str(a[row][col]),
        print "]",
    print "]"

#print2dList(parsingvalue)
#print2dList(t_sortedparsingtable)


import csv
with open('testheader.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',',quotechar="'")
    dataa = [['Me', 'You'],
            ['293', '219'],
            ['54', '13']]
    a.writerows(t_sortedparsingtable)
with open('testheader.csv', 'a') as fp:
    a = csv.writer(fp, delimiter=',',quotechar="'")
    dataa = [['Me', 'You'],
            ['293', '219'],
            ['54', '13']]
    a.writerows(t_parsingvalue)


