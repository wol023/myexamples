from subprocess import Popen, PIPE

(pbsnodes, stderr) = Popen(["pbsnodes"], stdout=PIPE).communicate()
class Node:
	def __init__(self,name):
		self.name=name
		self.state='NOTDEF'
		self.power_state='NOTDEF'
		self.np='NOTDEF'
		self.ntype='NOTDEF'
		self.jobs='NOTDEF'
	def set_state(self,state):
		self.state=state
	def set_power_state(self,power_state):
		self.power_state=power_state
	def set_np(self,np):
		self.np=np
	def set_ntype(self,ntype):
		self.ntype=ntype
	def set_jobs(self,jobs):
		self.jobs=jobs

splitted_stdout=pbsnodes.split("\n")
#for item_num in range(0,len(splitted_stdout)):
#	print item_num,splitted_stdout[item_num]

import fnmatch
j=-1
nodes=[]
for item_num in range(0,len(splitted_stdout)):
	if fnmatch.fnmatch(splitted_stdout[item_num],'node??' ):
		j=j+1
		nodes.append(Node(splitted_stdout[item_num]))
	if fnmatch.fnmatch(splitted_stdout[item_num],'* state = *'):
		splitted_state=(splitted_stdout[item_num]).split(" = ")
		if len(splitted_state)==2:
			nodes[j].set_state(splitted_state[1])
	if fnmatch.fnmatch(splitted_stdout[item_num],'* power_state = *'):
		splitted_state=(splitted_stdout[item_num]).split(" = ")
		if len(splitted_state)==2:
			nodes[j].set_power_state(splitted_state[1])
	if fnmatch.fnmatch(splitted_stdout[item_num],'*jobs =*'):
		splitted_state=(splitted_stdout[item_num]).split(" = ")
		if len(splitted_state)==2:
			nodes[j].set_jobs(splitted_state[1])


for item_num in range(0,len(nodes)):
	print nodes[item_num].name, nodes[item_num].state, nodes[item_num].power_state, nodes[item_num].jobs

#exclude running node
freenodes=[]
j=-1
for item_num in range(0,len(nodes)):
	if nodes[item_num].state=='job-exclusive' and nodes[item_num].power_state == 'Running' and nodes[item_num].jobs != 'NOTDEF':
		print 'to be excluded',nodes[item_num].name
	else:
		freenodes.append(nodes[item_num])

#clear psm_shm* files
import os
for item_num in range(0,len(freenodes)):
	#print ("ssh "+freenodes[item_num].name+" \'ls /dev/shm\'")
	#os.system("ssh "+freenodes[item_num].name+" \'ls /dev/shm\'")
	print ("ssh "+freenodes[item_num].name+" \'rm -v /dev/shm/psm_shm.*\'")
	os.system("ssh "+freenodes[item_num].name+" \'rm -v /dev/shm/psm_shm.*\'")



#(showq, stderr) = Popen(["showq"], stdout=PIPE).communicate()
#print showq
#splitted_showq=showq.split("\n")
#for item_num in range(0,len(splitted_showq)):
#	print item_num,splitted_showq[item_num]

