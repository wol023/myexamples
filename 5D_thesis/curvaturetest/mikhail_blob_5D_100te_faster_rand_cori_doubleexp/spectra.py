import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.interpolate 

import os, fnmatch
import ConfigParser

from setupplot import init_plotting

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
       for name in files:
           if fnmatch.fnmatch(name, pattern):
               result.append(os.path.join(root, name))
    return result

def matchparen(inputstring,start):
    flag=0
    if start ==-1:
        start= len(inputstring)-1

    if start not in range(len(inputstring)):
        start=0
    else:
        if inputstring[start]=='(':
            flag =1
            start +=1
            while (start<len(inputstring)-1):
                if inputstring[start]=='(':
                    flag+=1
                    start+=1
                    continue
                elif inputstring[start]==')':
                    if flag==1:
                        return start
                    else:
                        flag-=1
                        start+=1
                        continue
                else:
                    start+=1
                    continue
                    
        elif inputstring[start]==')':
            flag =1
            start-=1
            while (start>0):
                if inputstring[start]==')':
                    flag+=1
                    start-=1
                    continue
                elif inputstring[start]=='(':
                    if flag==1:
                        return start
                    else:
                        flag-=1
                        start-=1
                        continue
                else:
                    start-=1
                    continue
    if not flag== 1:
        return -1
    else:
        return start


def findargument(token,start):
    end = start
    flag=0
    while(start>0):
        print token[start]
        if token[start]=='*':
            end=start
            start-=1
        elif token[start].isdigit():
            flag=1
            start-=1
        elif token[start]=='+':
            break
        elif token[start]=='(':
            if flag==0:
                start-=1
            else:
                break
        else:
            start-=1
    if flag==0:
        return '1.0'
    else:
        return token[start+1:end]
        

    

                
            




def findpertbation(inputstr,tofind):
    result = []
    ind_first=inputstr.find(tofind)
    ind_next=ind_first
    while ind_next >= 0:
        ind_check=inputstr[ind_next:].find(tofind)
        if ind_check>=0:
            ind_first=ind_next
            return token[start+1:end]
        else:
            start-=1
        

    

                
            




def findpertbation(inputstr,tofind):
    result = []
    ind_first=inputstr.find(tofind)
    ind_next=ind_first
    while ind_next >= 0:
        ind_check=inputstr[ind_next:].find(tofind)
        if ind_check>=0:
            ind_first=ind_next
            #print ind_next
            balance=0
            token = ''
            for letter in inputstr[ind_first:]:
                if ind_next==len(inputstr)-1:
                    ind_next=-1
                else:
                    ind_next+=1
                if letter =='(':
                    balance=balance+1
                    token +=letter
                    #print token,balance
                elif letter ==')':
                    balance=balance-1
                    if balance == 0:
                        token+=letter
                        #print token,balance
                        break
                    else:
                        token+=letter
                        #print token,balance
                else:
                    if balance<1:
                        continue
                    else:
                        token+=letter
                        #print token,balance
            #now token has something like (1*y)
            #replace pi to np.pi
            
            token=token.replace('pi','np.pi')
            token=token.replace('PI','np.pi')
            token=token.replace('Pi','np.pi')

            result.append(token)
        else:
            ind_next=-1

    return result

def findmodenumber(token,tofind): #token is (1*y), tofind is y
    modenumber=0.0
    replaced_token=''
    ind_var=token.find(tofind)
    ind_pivot=ind_var
    #scan to the left
    #print 'ind_var=',ind_var
    end_var=-1
    start_var=ind_var
    balance=0
    while ind_var-1>=0:
        #print 'token[',ind_var,']=',token[ind_var]
        if token[ind_var]==')':
            balance+=1
            ind_var-=1
            continue
        if token[ind_var]=='(':
            balance-=1
            if balance<1:
                ind_var-=1
            continue

        if token[ind_var]=='*':
            if end_var==-1:#end_var was not set because * was ommitted
                end_var=ind_var-1
            #print 'end_var=',end_var
            ind_var-=1
            continue
        #elif token[ind_var-1]=='(' or token[ind_var-1]=='+' or token[ind_var-1]=='-' or token[ind_var-1].isalpha():
        elif (token[ind_var-1]=='+' or token[ind_var-1]=='-') and balance<1:
            start_var=ind_var-1
            ind_var-=1
            #print 'start_var=',start_var
            #print 'ind_var=',ind_var
            break
        elif token[ind_var].isdigit()or token[ind_var]=='.' or ind_var==ind_pivot:#digits
            if end_var==-1 and token[ind_var-1]=='*':
                end_var=ind_var-1
            elif end_var==-1:
                end_var=ind_var
            ind_var-=1
            #print 'ind_var=',ind_var
            continue
        else:
            ind_var-=1
            continue
    #if start_var!=-1 and start_var!=end_var:
    #    print 'token[',start_var,':',end_var,']=',token[start_var:end_var]

    #scan to the right
    ind_var=ind_pivot
    balance=0
    #print 'ind_var=',ind_var
    while ind_var+1<=len(token):
        #print 'token[',ind_var,']=',token[ind_var]
        if token[ind_var]=='(':
            balance+=1
            #print 'token[',start_var,':',ind_var,']=',token[start_var:ind_var]
            #print 'token[',ind_var,']=',token[ind_var],':balance=',balance
            ind_var+=1
            continue
        if token[ind_var]==')':
            balance-=1
            #print 'token[',start_var,':',ind_var,']=',token[start_var:ind_var]
            #print 'token[',ind_var,']=',token[ind_var],':balance=',balance
            if balance<1 and balance>=0:
                end_var=ind_var+1
                ind_var+=1
                break
            elif balance<0:
                end_var=ind_var
                ind_var+=1
                break
            else:
                ind_var+=1
            continue

        if (token[ind_var]=='+' or token[ind_var]=='-') and balance<1:
            end_var=ind_var
            ind_var+=1
            break
        else:
            ind_var+=1

    if start_var!=-1 and start_var!=end_var:
        print tofind,': token[',start_var,':',end_var,']=',token[start_var:end_var]
        replaced_token=token[start_var:end_var].replace(tofind,'1.0')
        print tofind,': replaced_token=',replaced_token

    if len(replaced_token)>0:
        modenumber=eval(replaced_token)
        print tofind,': evaluated_token=',modenumber
    return modenumber


#######################################################################
#### ASSIGN CONTANTS
qe=0.00000000048032   # electric unit charge [StatC]
me_theory=9.1094E-028 # realistic electron mass [gram]
mpn=1.6726E-024       # proton or neutron mass [gram]
c= 29979000000        # speed of light [cm/s]
evtoerg=1.6022E-12    # conversion factor [erg/ev]
teslatogause=1.0E4    # conversion factor [gauss/T]
#### ASSIGN CONTANTS
#######################################################################


#######################################################################
#### READ INPUT DECK
## find input file by extension '.in' from current directory
fname=find('*.in', './')

print '************ INPUT FILE *****************'
with open('finish.txt', 'wb') as fh:
    buf = "************ INPUT FILE *****************\n"
    fh.write(buf)

#Start read the input file line by line
#Extract useful parameters
#
# units_number_density
# units_temperature 
# units_length 
# units_mass 
# units_magnetic_field
# boltzmann_electron_temperature 
# bz_inner 
# by_inner 
# x_max 
# y_max 
# z_max 
# ion_mass 
# elec_mass 
# deltaL_analytic 
# m_x
# m_y
# m_z
# t0_grid_func 
# et0_grid_func 
# x_cells
# y_cells
# z_cells
# x_index
# y_index
# z_index
# electron_temperature  
# B_y
# B_z
# B_t
# Te_ev
# Ti_ev
# Te_erg
# Ti_erg
# me_g 
# mi_g


#initialize variables
boltzmann_electron_temperature = -1
electron_temperature = -1
et0_grid_func= -1
x_max=1.0;
y_max=1.0;
z_max=1.0;
#deltaL_analytic=1;

with open(fname[0], 'r') as f:
    for line in f:
        if line.lstrip().startswith('#'): #skip comment
            continue
        line =line.rstrip() #skip blank line
        if not line:
            continue 
        else: #noncomment line
            strippedline=line
            lhsrhs = strippedline.split("=")
            l=0
            while l<len(lhsrhs): #strip white spaces in lhs
                lhsrhs[l]=lhsrhs[l].rstrip()
                lhsrhs[l]=lhsrhs[l].lstrip()
                l=l+1
            #print type( lhsrhs[0])
            if 'units.number_density' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                units_number_density =float(lhsrhs[1])
                #print 'IN:units_number_density = ',units_number_density
            if 'units.temperature' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                units_temperature = float(lhsrhs[1])
                print 'IN:units_temperature = ',units_temperature
            if 'units.length' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                units_length = float(lhsrhs[1])
                #print 'IN:units_length= ',units_length
            if 'units.mass' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                units_mass = float(lhsrhs[1])
                #print 'IN:units_mass = ',units_mass
            if 'units.magnetic_field' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                units_magnetic_field = float(lhsrhs[1])
                #print 'IN:units_magnetic_field = ',units_magnetic_field
            if 'boltzmann_electron.temperature' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                boltzmann_electron_temperature = float(lhsrhs[1])
                print 'IN:boltzmann_electron_temperature = ',boltzmann_electron_temperature
                with open('finish.txt', 'a+') as fh:
                    buf = 'IN:boltzmann_electron_temperature = %f\n' % boltzmann_electron_temperature
                    fh.write(buf)
            if 'gksystem.magnetic_geometry_mapping.slab.Bz_inner' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                bz_inner = float(lhsrhs[1])
                #print 'IN:bz_inner = ',bz_inner
                with open('finish.txt', 'a+') as fh:
                    buf = 'bz_inner = %f\n' % float(lhsrhs[1])
                    fh.write(buf)
            if 'gksystem.magnetic_geometry_mapping.slab.By_inner' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                by_inner = float(lhsrhs[1])
                #print 'IN:by_inner = ',by_inner
                with open('finish.txt', 'a+') as fh:
                    buf = 'by_inner = %f\n' % float(lhsrhs[1])
                    fh.write(buf)

            if 'gksystem.magnetic_geometry_mapping.slab.x_max' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                x_max = float(lhsrhs[1])
                print 'IN:x_max = ',x_max
                with open('finish.txt', 'a+') as fh:
                    buf = 'IN:x_max = %f\n' % x_max
                    fh.write(buf)
            if 'gksystem.magnetic_geometry_mapping.slab.y_max' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                y_max = float(lhsrhs[1])
                print 'IN:y_max = ',y_max
                with open('finish.txt', 'a+') as fh:
                    buf = 'IN:y_max = %f\n' % y_max
                    fh.write(buf)
            if 'gksystem.magnetic_geometry_mapping.slab.z_max' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                z_max = float(lhsrhs[1])
                print 'IN:z_max = ',z_max
                with open('finish.txt', 'a+') as fh:
                    buf = 'IN:z_max = %f\n' % z_max
                    fh.write(buf)

            if 'kinetic_species.1.mass' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                ion_mass = float(lhsrhs[1])
                print 'IN:ion_mass = ',ion_mass
                with open('finish.txt', 'a+') as fh:
                    buf = 'ion_mass = %f\n' % float(lhsrhs[1])
                    fh.write(buf)
            if 'kinetic_species.2.mass' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                elec_mass = float(lhsrhs[1])
                print 'IN:elec_mass = ',elec_mass
                with open('finish.txt', 'a+') as fh:
                    buf = 'elec_mass = %f\n' % float(lhsrhs[1])
                    fh.write(buf)

            if '.N0_grid_func.function' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                n0_grid_func=lhsrhs[1][1:-1] #remove double quotes
                n0_grid_func=n0_grid_func.lstrip()
                n0_grid_func=n0_grid_func.rstrip()
                n0_grid_func=n0_grid_func.replace('^','**')
                print 'IN:n0_grid_func = ',n0_grid_func
                with open('finish.txt', 'a+') as fh:
                    buf = 'IN:no_grid_func = %s\n' % n0_grid_func
                    fh.write(buf)

                m_y=1.0; #default was 0
                m_x=0.0;
                m_z=0.0;
                m_theta=0.0;

                tokens=findpertbation(n0_grid_func,'sin(')
                print 'sin():',tokens
                with open('finish.txt', 'a+') as fh:
                    buf = 'sin(): %s\n' % tokens
                    fh.write(buf)
                for token in tokens:
                    print "For ",token
                    if 'arctan' in token:
                        splited_token=token.split('arctan')
                        temp_theta = splited_token[0][1:-1]
                        m_theta = float(temp_theta)
                        if 'z' in splited_token[1]:
                            z_token = splited_token[1].split('z')
                            ind_z=-1
                            while(-ind_z<=len(z_token[0])):
                                if z_token[0][ind_z]=='+' or z_token[0][ind_z]=='-':
                                    break
                                ind_z-=1
                            m_z = float(z_token[0][ind_z+1:-1])
                    elif 'atwotan' in token:
                        m_theta = float(findargument(token,matchparen(token,token.find('atwotan')-1)))
                        

                        arg1=n0_grid_func[matchparen(n0_grid_func,n0_grid_func.find('atwotan')-1):n0_grid_func.find('atwotan')] 
                        arg2=n0_grid_func[n0_grid_func.find('atwotan')+7:matchparen(n0_grid_func,n0_grid_func.find('atwotan')+7)+1]

                        raw=n0_grid_func[matchparen(n0_grid_func,n0_grid_func.find('atwotan')-1):matchparen(n0_grid_func,n0_grid_func.find('atwotan')+7)+1]

                        n0_grid_func=n0_grid_func.replace(raw,'atan2('+arg1+','+arg2+')')



                        splited_token=token.split('atwotan')
                        if 'z' in splited_token[1]:
                            z_token = splited_token[1].split('z')
                            ind_z=-1
                            while(-ind_z<=len(z_token[0])):
                                if z_token[0][ind_z]=='+' or z_token[0][ind_z]=='-':
                                    break
                                ind_z-=1
                            m_z = float(z_token[0][ind_z+1:-1])

                    else:
                        temp_y=findmodenumber(token,'y')
                        temp_x=findmodenumber(token,'x')
                        temp_z=findmodenumber(token,'z')
                        if abs(temp_y)>abs(m_y):
                                m_y=temp_y
                        if abs(temp_x)>abs(m_x):
                                m_x=temp_x
                        if abs(temp_z)>abs(m_z):
                                m_z=temp_z

                print 'IN:m_y=',m_y
                print 'IN:m_x=',m_x
                print 'IN:m_z=',m_z
                print 'IN:m_theta=',m_theta
                with open('finish.txt', 'a+') as fh:
                    buf = 'IN:m_y= %f\n' % m_y
                    fh.write(buf)
                    buf = 'IN:m_x= %f\n' % m_x
                    fh.write(buf)
                    buf = 'IN:m_z= %f\n' % m_z
                    fh.write(buf)
                    buf = 'IN:m_theta= %f\n' % m_theta
                    fh.write(buf)

                
            if '.T0_grid_func.constant' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                t0_grid_func=float(lhsrhs[1])
                print 'IN:t0_grid_func = ',t0_grid_func
                with open('finish.txt', 'a+') as fh:
                    buf = 'IN:t0_grid_func = %f\n' % t0_grid_func
                    fh.write(buf)
            if '.T0_grid_func.value' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                t0_grid_func=float(lhsrhs[1])
                print 'IN:t0_grid_func = ',t0_grid_func
                with open('finish.txt', 'a+') as fh:
                    buf = 'IN:t0_grid_func = %f\n' % t0_grid_func
                    fh.write(buf)
            if '.eT0_grid_func.constant' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                et0_grid_func=float(lhsrhs[1])
                print 'IN:et0_grid_func = ',et0_grid_func
                with open('finish.txt', 'a+') as fh:
                    buf = 'IN:et0_grid_func = %f\n' % et0_grid_func
                    fh.write(buf)
            if '.eT0_grid_func.value' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                et0_grid_func=float(lhsrhs[1])
                print 'IN:et0_grid_func = ',et0_grid_func
                with open('finish.txt', 'a+') as fh:
                    buf = 'IN:et0_grid_func = %f\n' % et0_grid_func
                    fh.write(buf)
            
            if 'gksystem.num_cells' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                num_cells=lhsrhs[1].split()
                #print 'IN:num_cells = ',num_cells
                if len(num_cells) == 4:
                    x_cells=int(num_cells[0])
                    #print 'IN:x_cells = ', x_cells
                    y_cells=int(num_cells[1])
                    #print 'IN:y_cells = ', y_cells
                elif len(num_cells) == 5:
                    x_cells=int(num_cells[0])
                    #print 'IN:x_cells = ', x_cells
                    y_cells=int(num_cells[1])
                    #print 'IN:y_cells = ', y_cells
                    z_cells=int(num_cells[2])
                    #print 'IN:z_cells = ', z_cells
            if '.history_indices' in lhsrhs[0]:
                #print lhsrhs[0],'=',lhsrhs[1]
                history_indices=lhsrhs[1].split()
                if len(history_indices) == 2:
                    x_index=int(history_indices[0])
                    #print 'IN:x_index = ', x_index
                    y_index=int(history_indices[1])
                    #print 'IN:y_index = ', y_index
                elif len(history_indices) == 3:
                    x_index=int(history_indices[0])
                    #print 'IN:x_index = ', x_index
                    y_index=int(history_indices[1])
                    #print 'IN:y_index = ', y_index
                    z_index=int(history_indices[2])
                    #print 'IN:z_index = ', z_index


f.closed

if  boltzmann_electron_temperature == -1:
    if et0_grid_func != -1:
        electron_temperature=et0_grid_func
    else:
        electron_temperature=t0_grid_func

    print 'Te(kin) = ', electron_temperature
    with open('finish.txt', 'a+') as fh:
        buf = 'Te(kin) = %f\n' % electron_temperature
        fh.write(buf)
else:
    electron_temperature = boltzmann_electron_temperature
    print 'Te(bol) = ', electron_temperature
    with open('finish.txt', 'a+') as fh:
        buf = 'Te(bol) = %f\n' % electron_temperature
        fh.write(buf)




B_z = bz_inner*units_magnetic_field*teslatogause
B_y = by_inner*units_magnetic_field*teslatogause
B_t = np.sqrt(B_z**2+B_y**2)
print 'B_z          [gauss] = ', B_z
print 'B_y          [gauss] = ', B_y
print 'B_t          [gauss] = ', B_t
with open('finish.txt', 'a+') as fh:
    buf = 'B_z          [gauss] = %f\n' % B_z
    fh.write(buf)
    buf = 'B_y          [gauss] = %f\n' % B_y
    fh.write(buf)
    buf = 'B_t          [gauss] = %f\n' % B_t
    fh.write(buf)
    

Te_ev = electron_temperature*units_temperature
Ti_ev = t0_grid_func * units_temperature

Te_erg = Te_ev * evtoerg
Ti_erg = Ti_ev * evtoerg

me_g= elec_mass*units_mass*mpn
mi_g= ion_mass*units_mass*mpn

#### READ INPUT DECK
#######################################################################


# calc ref_time from input deck parameters
const_ELEMENTARY_CHARGE   = 1.60217653e-19 # C
const_MASS_OF_PROTON      = 1.67262171e-27 # kg
tempJoules = const_ELEMENTARY_CHARGE* units_temperature
masskg = units_mass * const_MASS_OF_PROTON
ref_speed = np.sqrt(tempJoules/masskg) 
ref_time=units_length/ref_speed
print '********** OUTPUT FILE ******************'
with open('finish.txt', 'a+') as fh:
    buf = '********** OUTPUT FILE ******************\n'
    fh.write(buf)
    buf = 'THERMAL SPEED       [m/s] = %g\n' % ref_speed
    fh.write(buf)
    buf = 'TRANSIT TIME          [s] = %g\n' % ref_time
    fh.write(buf)

########################################################################
##### READ OUTPUT FILE
## Try reading slurm-*.out file from nersc CORI run
## If the file is not found, try reading output file from from perun-cluster 
## varialbe list
##
## ref_time
## ref_speed
## ref_gyrofrequency 
## ref_gyroradius 
## ref_debyelength
## ref_larmornumber
## ref_debyenumber
##
#
#ref_time=0.0
#
#print '********** OUTPUT FILE ******************'
#with open('finish.txt', 'a+') as fh:
#    buf = '********** OUTPUT FILE ******************\n'
#    fh.write(buf)
#
#fname=find('slurm-*.out', './')
#if len(fname)==0:
#	fname=find('perunoutput.out', './')
#
#with open(fname[0], 'r') as f:
#    for line in f:
#        if line.lstrip().startswith('*'): #skip comment
#            continue
#        line =line.rstrip() #skip blank line
#        if not line:
#            continue 
#        else: #noncomment line
#            strippedline=line
#            lhsrhs = strippedline.split(":")
#            l=0
#            while l<len(lhsrhs): #strip white spaces in lhs
#                lhsrhs[l]=lhsrhs[l].rstrip()
#                lhsrhs[l]=lhsrhs[l].lstrip()
#                l=l+1
#            #print type( lhsrhs[0])
#            if 'TRANSIT TIME' in lhsrhs[0]:
#                print lhsrhs[0],'=',lhsrhs[1]
#                with open('finish.txt', 'a+') as fh:
#                    buf = '%s = ' % lhsrhs[0]
#                    fh.write(buf)
#                    buf = '%s\n' % lhsrhs[1]
#                    fh.write(buf)
#                ref_time=float(lhsrhs[1])
#            if 'THERMAL SPEED' in lhsrhs[0]:
#                print lhsrhs[0],'=',lhsrhs[1]
#                with open('finish.txt', 'a+') as fh:
#                    buf = '%s = ' % lhsrhs[0]
#                    fh.write(buf)
#                    buf = '%s\n' % lhsrhs[1]
#                    fh.write(buf)
#                ref_speed=float(lhsrhs[1])
#            if 'GYROFREQUENCY' in lhsrhs[0]:
#                print lhsrhs[0],'=',lhsrhs[1]
#                with open('finish.txt', 'a+') as fh:
#                    buf = '%s = ' % lhsrhs[0]
#                    fh.write(buf)
#                    buf = '%s\n' % lhsrhs[1]
#                    fh.write(buf)
#                ref_gyrofrequency=float(lhsrhs[1])
#            if 'GYRORADIUS' in lhsrhs[0]:
#                print lhsrhs[0],'=',lhsrhs[1]
#                with open('finish.txt', 'a+') as fh:
#                    buf = '%s = ' % lhsrhs[0]
#                    fh.write(buf)
#                    buf = '%s\n' % lhsrhs[1]
#                    fh.write(buf)
#                ref_gyroradius=float(lhsrhs[1])
#            if 'DEBYE LENGTH' in lhsrhs[0]:
#                print lhsrhs[0],'=',lhsrhs[1]
#                with open('finish.txt', 'a+') as fh:
#                    buf = '%s = ' % lhsrhs[0]
#                    fh.write(buf)
#                    buf = '%s\n' % lhsrhs[1]
#                    fh.write(buf)
#                ref_debyelength=float(lhsrhs[1])
#            if 'LARMOR NUMBER' in lhsrhs[0]:
#                print lhsrhs[0],'=',lhsrhs[1]
#                with open('finish.txt', 'a+') as fh:
#                    buf = '%s = ' % lhsrhs[0]
#                    fh.write(buf)
#                    buf = '%s\n' % lhsrhs[1]
#                    fh.write(buf)
#                ref_larmornumber=float(lhsrhs[1])
#            if 'DEBYE NUMBER' in lhsrhs[0]:
#                print lhsrhs[0],'=',lhsrhs[1]
#                with open('finish.txt', 'a+') as fh:
#                    buf = '%s = ' % lhsrhs[0]
#                    fh.write(buf)
#                    buf = '%s\n' % lhsrhs[1]
#                    fh.write(buf)
#                ref_debyenumber=float(lhsrhs[1])
#                break
#
#f.closed
##### READ OUTPUT FILE
########################################################################

#######################################################################
#### RECONSTRUCT BACKGROUND DENSITY FROM INPUT DECK
# After executing this routine we have
# xx : 0 ~ 2pi
# yy : density profile, n(x)
# xcm : physical x domain in [cm] 
# dx : x direction grid spacing  in [cm]

from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations
from sympy.parsing.sympy_parser import implicit_multiplication_application
from sympy import *
from sympy.abc import x, y, z
from sympy.utilities.lambdify import implemented_function
from sympy import Function

transformations = (standard_transformations + (implicit_multiplication_application,))
n0_grid_func_compatible=n0_grid_func.replace('arctan','atan')
print 'n0_grid_func_compatible = ',n0_grid_func_compatible
n0_grid_func_compatible=n0_grid_func_compatible.replace('rand','1.0*')
print 'n0_grid_func_compatible = ',n0_grid_func_compatible
pe=parse_expr(n0_grid_func_compatible,transformations=transformations)
print 'parsed expression = ',pe

f = lambdify((x,y,z),pe)
#print f(pi,pi) #test

xx = np.linspace(0.0, np.pi*2, (x_cells))
xcm  = np.linspace(0.0, x_max*100, (x_cells))
yy = np.linspace(0.0, np.pi*2, (x_cells))
yypert = np.linspace(0.0, np.pi*2, (x_cells))

y_point_rad= float(y_index)/float(y_cells)*np.pi*2
for i in range(len(xx)):
    yy[i] = f(xx[i],y_point_rad,0)


#calc yypert
dphasepert = np.pi/10
#find optimum phase shift
maxyypert_amp = 0.0
maxyypert_phi_ind =0

for j in range(10):
    for i in range(len(xx)):
        yypert[i] =abs(f(xx[i],dphasepert*j,0) - yy[i])
    tempyypert_amp = max( (yypert))
    if tempyypert_amp>maxyypert_amp:
        maxyypert_amp=tempyypert_amp
        maxyypert_phi_ind=j
for i in range(len(xx)):
    yypert[i] =abs(f(xx[i],dphasepert*maxyypert_phi_ind,0) - yy[i])


dx = x_max*100/len(xx) #in [cm]
dyydx = np.gradient(yy)/dx

inter_yy = scipy.interpolate.InterpolatedUnivariateSpline(xcm,yy,k=3)
dinter_yydx=inter_yy.derivative()
dlninter_yydx = dinter_yydx(xcm)/yy

dlnyydx = dyydx/yy
dlnyydx[0] = dlnyydx[1] #fix discontinuity
dlnyydx[len(dlnyydx)-1] = dlnyydx[len(dlnyydx)-2] #fix discontinuity



#print 'average density = ',np.average(yy)

#calculate spread indices
contains_tanh =0
contains_pert =0

if 'tanh' in n0_grid_func:
    contains_tanh =1
    #print "contains_tanh"
if ')**2' in n0_grid_func:
    contains_pert =1
    #print "contains_pert"

if contains_tanh and contains_pert:
    dlnyydx_abs = abs(dlnyydx)
    dlnyydx_amplitude= max(dlnyydx_abs)
    ispread_width=sum(dlnyydx_abs>dlnyydx_amplitude*0.33)
    yypert_amplitude = max(yypert)
    ispread_width_density_pert=sum(yypert>yypert_amplitude*0.10)
    ispread_width = min(ispread_width,ispread_width_density_pert)
elif contains_tanh:
    dlnyydx_abs = abs(dlnyydx)
    dlnyydx_amplitude= max(dlnyydx_abs)
    ispread_width=sum(dlnyydx_abs>dlnyydx_amplitude*0.33)
elif contains_pert:
    yypert_amplitude = max(yypert)
    ispread_width=sum(yypert>yypert_amplitude*0.45)
else:
    ispread_width=1
    #print "does not contain spreading"

#### RECONSTRUCT BACKGROUND DENSITY FROM INPUT DECK
#######################################################################

print '********** DERIVED VARS *****************'
with open('finish.txt', 'a+') as fh:
    buf = '********** DERIVED VARS *****************\n'
    fh.write(buf)


##################  NOT USED
#c_ion_thermalspeed         = 41900000*(( t0_grid_func*units_temperature)**0.5)*((me_theory/mpn/ion_mass)**0.5)
#c_elec_thermalspeed        = 41900000*(electron_temperature*units_temperature)**0.5
#c_ion_transittimefor100cm  = units_length*100/ c_ion_thermalspeed
#
#c_elec_transittimefor100cm = units_length*100/ c_elec_thermalspeed
#c_ion_gyrofrequency        = 9580*(B_t/ ion_mass )
#c_elec_gyrofrequency       = 9580*(B_t/ ion_mass *ion_mass )*mpn/ me_theory
#c_ion_gyroradius           = c_ion_thermalspeed / c_ion_gyrofrequency
#c_elec_gyroradius          = c_elec_thermalspeed / c_elec_gyrofrequency
#
#print 'c_ion_thermalspeed      [cm/s] = ', c_ion_thermalspeed, '     (/ref: ', c_ion_thermalspeed/(ref_speed*100),' )' 
#with open('finish.txt', 'a+') as fh:
#    buf = 'c_ion_thermalspeed      [cm/s] = %f' % c_ion_thermalspeed
#    fh.write(buf)
#    buf = '     (ref: %f' % (c_ion_thermalspeed/(ref_speed*100))
#    fh.write(buf)
#    buf = ')\n'
#    fh.write(buf)
#print 'c_elec_thermalspeed     [cm/s] = ', c_elec_thermalspeed, '     (/ref: ', c_elec_thermalspeed/(ref_speed*100), ' )'
#with open('finish.txt', 'a+') as fh:
#    buf = 'c_elec_thermalspeed     [cm/s] = %f' % c_elec_thermalspeed
#    fh.write(buf)
#    buf = '     (ref: %f' % (c_elec_thermalspeed/(ref_speed*100))
#    fh.write(buf)
#    buf = ')\n'
#    fh.write(buf)
#print 'c_ion_transittimefor100cm  [s] = ', c_ion_transittimefor100cm, ' (/ref: ', c_ion_transittimefor100cm/ref_time,' )'
#with open('finish.txt', 'a+') as fh:
#    buf = 'c_ion_transittimefor100cm  [s] = %f' % c_ion_transittimefor100cm
#    fh.write(buf)
#    buf = '     (ref: %f' % (c_ion_transittimefor100cm/ref_time )
#    fh.write(buf)
#    buf = ')\n'
#    fh.write(buf)
#print 'c_elec_transittimefor100cm [s] = ', c_elec_transittimefor100cm, ' (/ref: ', c_elec_transittimefor100cm/ref_time, ' )'
#with open('finish.txt', 'a+') as fh:
#    buf = 'c_elec_transittimefor100cm [s] = %f' % c_elec_transittimefor100cm
#    fh.write(buf)
#    buf = '     (ref: %f' % (c_elec_transittimefor100cm/ref_time )
#    fh.write(buf)
#    buf = ')\n'
#    fh.write(buf)
#print 'c_ion_gyrofrequency      [1/s] = ', c_ion_gyrofrequency, '       (/ref: ', c_ion_gyrofrequency/ ref_gyrofrequency, ' )'
#with open('finish.txt', 'a+') as fh:
#    buf = 'c_ion_gyrofrequency      [1/s] = %f' % c_ion_gyrofrequency
#    fh.write(buf)
#    buf = '     (ref: %f' % (c_ion_gyrofrequency/ ref_gyrofrequency)
#    fh.write(buf)
#    buf = ')\n'
#    fh.write(buf)
#print 'c_elec_gyrofrequency     [1/s] = ', c_elec_gyrofrequency, ' (/ref: ', c_elec_gyrofrequency/ ref_gyrofrequency, ' )'
#with open('finish.txt', 'a+') as fh:
#    buf = 'c_elec_gyrofrequency     [1/s] = %f' % c_elec_gyrofrequency
#    fh.write(buf)
#    buf = '     (ref: %f' % (c_elec_gyrofrequency/ ref_gyrofrequency)
#    fh.write(buf)
#    buf = ')\n'
#    fh.write(buf)
#print 'c_ion_gyroradius          [cm] = ', c_ion_gyroradius, '   (/ref: ', c_ion_gyroradius / (units_length*100), ' )'
#with open('finish.txt', 'a+') as fh:
#    buf = 'c_ion_gyroradius          [cm] = %f' % c_ion_gyroradius
#    fh.write(buf)
#    buf = '     (ref: %f' % (c_ion_gyroradius/ (units_length*100))
#    fh.write(buf)
#    buf = ')\n'
#    fh.write(buf)
#print 'c_elec_gyroradius         [cm] = ', c_elec_gyroradius, '   (/ref: ', c_elec_gyroradius / (units_length*100), ' )'
#with open('finish.txt', 'a+') as fh:
#    buf = 'c_elec_gyroradius         [cm] = %f' % c_elec_gyroradius
#    fh.write(buf)
#    buf = '     (ref: %f' % (c_elec_gyroradius/ (units_length*100))
#    fh.write(buf)
#    buf = ')\n'
#    fh.write(buf)
#############################

#######################################################################
#### DEFINE SPEED and GYROFREQUENCY and GYRORADIUS
# v_ti [cm/s]
# v_te [cm/s]
# omega_ci
# omega_ce
# omega_pi
# c_s
# rho_s
# rho_i
# omega_star

v_ti = (2.0*Ti_erg/mi_g)**0.5
v_te = (2.0*Te_erg/me_g)**0.5

omega_ci =  qe*B_t/mi_g/c
omega_ce =  qe*B_t/me_g/c
omega_pi = (4.0*np.pi*qe**2*units_number_density*1.E-6/mi_g)**0.5 #assumed n0 is 1

c_s      = (Te_erg/mi_g)**0.5
rho_s    = c_s/omega_ci
rho_i    = v_ti/omega_ci

k_y        = 2.0*np.pi*m_y/(y_max*100)
k_x        = 2.0*np.pi*m_x/(x_max*100) 
k_z        = 2.0*np.pi*m_z/(z_max*100) 


deltaL_max = 1./max(abs(dlnyydx))
deltaL_spline = 1./max(abs(dlninter_yydx))
x_point_index_in_plot = int(float(x_index)/float(x_cells)*len(dlnyydx))-1 
#print 'x_point_index_in_plot = ', x_point_index_in_plot
deltaL_point= abs(1./(dlnyydx[x_point_index_in_plot]))

deltaL_spread =0.0
#ispread_width = 10
ispread_width = float(ispread_width)
spread_ind_diff = range(-int(np.floor(ispread_width/2)),int(np.ceil(ispread_width/2)))
if int(ispread_width)%2==0:
    tempfirstinddiff=-spread_ind_diff[0]
    del spread_ind_diff[0]
    spread_ind_diff.append(tempfirstinddiff)
#print spread_ind_diff

for ind in spread_ind_diff :
    deltaL_spread = deltaL_spread+dlnyydx[x_point_index_in_plot+ind] 
deltaL_spread = deltaL_spread/ispread_width
deltaL_spread = 1.0/abs(deltaL_spread)
#print type(spread_ind_diff)
spread_ind_diff=np.array(spread_ind_diff)
spread_ind =  spread_ind_diff+ x_point_index_in_plot
#print spread_ind




gradall=0.0
rcm_CM =0.0
center_cm = (xcm[len(xcm)/2-1]+xcm[len(xcm)/2])/2.0
for ind in spread_ind:
    gradall += dlnyydx[ind]
for ind in spread_ind:
    rcm_CM += dlnyydx[ind]*(xcm[ind]-center_cm)
rcm_CM=rcm_CM/gradall

print 'rcm_CM = ', rcm_CM
rcm = abs(xcm[np.argmax(-(dlnyydx))]-(xcm[len(xcm)/2-1]+xcm[len(xcm)/2])/2.0)
#k_theta    = m_theta/(x_max*100/4.0*0.9)
print 'rcm = ', rcm
k_theta    = m_theta/rcm_CM

k_par      = (k_y*B_y+k_z*B_z)/B_t
k_par_z    = k_par*B_z/B_t
k_par_y    = k_par*B_y/B_t

k_perp_z   = (k_z*B_y-k_y*B_z)*B_y/B_t/B_t
k_perp_y   = (k_y*B_z-k_z*B_y)*B_z/B_t/B_t
k_perp_x   = abs(k_x)
k_perp_yz  = np.sqrt(k_perp_z*k_perp_z+k_perp_y*k_perp_y)
k_perp     = np.sqrt(k_perp_yz*k_perp_yz+k_x*k_x)

if k_theta > 0.0:
    k_perp=k_theta
    k_perp_yz=k_theta

chi_x      = k_perp*rho_s
chi        = k_perp_yz*rho_s
#omega_star_max = c_s*rho_s*k_perp_y/deltaL_max
#omega_star_point= c_s*rho_s*k_perp_y/deltaL_point
#omega_star_spline = c_s*rho_s*k_perp_y/deltaL_spline
#omega_star_spread= c_s*rho_s*k_perp_y/ deltaL_spread
omega_star_max = c_s*rho_s*k_perp_yz/deltaL_max
omega_star_point= c_s*rho_s*k_perp_yz/deltaL_point
omega_star_spline = c_s*rho_s*k_perp_yz/deltaL_spline
omega_star_spread= c_s*rho_s*k_perp_yz/deltaL_spread

k_par_hat  = k_par*deltaL_spline*v_te/c_s

print 'k_x             [1/cm] = ', k_x , 'check m_x = (',m_x,') with kinetic.in'
with open('finish.txt', 'a+') as fh:
    buf = 'k_x             [1/cm] = %f' % k_x
    fh.write(buf)
    buf = ' check m_x = (%f' % m_x
    fh.write(buf)
    buf = ') with kinetic.in\n'
    fh.write(buf)
print 'k_y             [1/cm] = ', k_y , 'check m_y = (',m_y,') with kinetic.in'
with open('finish.txt', 'a+') as fh:
    buf = 'k_y             [1/cm] = %f' % k_y
    fh.write(buf)
    buf = ' check m_y = (%f' % m_y
    fh.write(buf)
    buf = ') with kinetic.in\n'
    fh.write(buf)
print 'k_z             [1/cm] = ', k_z , 'check m_z = (',m_z,') with kinetic.in'
with open('finish.txt', 'a+') as fh:
    buf = 'k_z             [1/cm] = %f' % k_z
    fh.write(buf)
    buf = ' check m_z = (%f' % m_z
    fh.write(buf)
    buf = ') with kinetic.in\n'
    fh.write(buf)
print 'k_perp          [1/cm] = ', k_perp
with open('finish.txt', 'a+') as fh:
    buf = 'k_perp          [1/cm] = %f\n' % k_perp
    fh.write(buf)
print 'k_perp_z        [1/cm] = ', k_perp_z
with open('finish.txt', 'a+') as fh:
    buf = 'k_perp_z        [1/cm] = %f\n' % k_perp_z
    fh.write(buf)
print 'k_perp_y        [1/cm] = ', k_perp_y
with open('finish.txt', 'a+') as fh:
    buf = 'k_perp_y        [1/cm] = %f\n' % k_perp_y
    fh.write(buf)
print 'k_perp_x        [1/cm] = ', k_perp_x
with open('finish.txt', 'a+') as fh:
    buf = 'k_perp_x        [1/cm] = %f\n' % k_perp_x
    fh.write(buf)
print 'k_perp_yz       [1/cm] = ', k_perp_yz
with open('finish.txt', 'a+') as fh:
    buf = 'k_perp_yz        [1/cm] = %f\n' % k_perp_yz
    fh.write(buf)
print 'k_par           [1/cm] = ', k_par
with open('finish.txt', 'a+') as fh:
    buf = 'k_par           [1/cm] = %f\n' % k_par
    fh.write(buf)
print 'k_par_y         [1/cm] = ', k_par_y
with open('finish.txt', 'a+') as fh:
    buf = 'k_par_y         [1/cm] = %f\n' % k_par_y
    fh.write(buf)
print 'k_par_z         [1/cm] = ', k_par_z
with open('finish.txt', 'a+') as fh:
    buf = 'k_par_z         [1/cm] = %f\n' % k_par_z
    fh.write(buf)
print 'k_par*(vte*delta/cs)[] = ', k_par_hat
with open('finish.txt', 'a+') as fh:
    buf = 'k_par*(vte*delta/cs)[] = %f\n' % k_par_hat
    fh.write(buf)
print 'deltaL_max        [cm] = ', deltaL_max
with open('finish.txt', 'a+') as fh:
    buf = 'deltaL_max        [cm] = %f\n' % deltaL_max
    fh.write(buf)
print 'deltaL_point      [cm] = ', deltaL_point
with open('finish.txt', 'a+') as fh:
    buf = 'deltaL_point     [cm] = %f\n' % deltaL_point
    fh.write(buf)
print 'deltaL_spline     [cm] = ', deltaL_spline
with open('finish.txt', 'a+') as fh:
    buf = 'deltaL_spline      [cm] = %f\n' % deltaL_spline
    fh.write(buf)
if (ispread_width !=1):
    print 'deltaL_spread     [cm] = ', deltaL_spread
    with open('finish.txt', 'a+') as fh:
        buf = 'deltaL_spread      [cm] = %f\n' % deltaL_spread
        fh.write(buf)
print 'c_s             [cm/s] = ', c_s
with open('finish.txt', 'a+') as fh:
    buf = 'c_s             [cm/s] = %f\n' % c_s
    fh.write(buf)
print 'rho_s             [cm] = ', rho_s
with open('finish.txt', 'a+') as fh:
    buf = 'rho_s             [cm] = %f\n' % rho_s
    fh.write(buf)
print 'k_perp_yz*rho_s    [-] = ', k_perp_yz*rho_s
with open('finish.txt', 'a+') as fh:
    buf = 'k_perp_yz*rho_s    [-] = %f\n' % (k_perp_yz*rho_s)
    fh.write(buf)
print 'k_perp_yz*rho_i    [-] = ', k_perp_yz*rho_i
with open('finish.txt', 'a+') as fh:
    buf = 'k_perp_yz*rho_i    [-] = %f\n' % (k_perp_yz*rho_i)
    fh.write(buf)
print 'k_perp*rho_s       [-] = ', k_perp*rho_s
with open('finish.txt', 'a+') as fh:
    buf = 'k_perp*rho_s       [-] = %f\n' % k_perp*rho_s
    fh.write(buf)
print 'k_perp*rho_i       [-] = ', k_perp*rho_i
with open('finish.txt', 'a+') as fh:
    buf = 'k_perp*rho_i       [-] = %f\n' % (k_perp*rho_i)
    fh.write(buf)
print 'omega*_max       [1/s] = ', omega_star_max
with open('finish.txt', 'a+') as fh:
    buf = 'omega*_max       [1/s] = %f\n' % omega_star_max
    fh.write(buf)
print 'omega*_point     [1/s] = ', omega_star_point
with open('finish.txt', 'a+') as fh:
    buf = 'omega*_point     [1/s] = %f\n' % omega_star_point
    fh.write(buf)
print 'omega*_spline    [1/s] = ', omega_star_spline
with open('finish.txt', 'a+') as fh:
    buf = 'omega*_spline     [1/s] = %f\n' % omega_star_spline
    fh.write(buf)
print 'omega*_spline_1_chi2[1/s] = ', omega_star_spline/(1.0+chi*chi)
with open('finish.txt', 'a+') as fh:
    buf = 'omega*_spline_1_chi2[1/s] = %f\n' % (omega_star_spline/(1.0+chi*chi))
    fh.write(buf)
print 'omega*_spline_1_chi_x2[1/s] = ', omega_star_spline/(1.0+chi_x*chi_x)
with open('finish.txt', 'a+') as fh:
    buf = 'omega*_spline_1_chi_x2[1/s] = %f\n' % (omega_star_spline/(1.0+chi_x*chi_x))
    fh.write(buf)
if (ispread_width !=1):
    print 'omega*_spread    [1/s] = ', omega_star_spread
    with open('finish.txt', 'a+') as fh:
        buf = 'omega*_spread    [1/s] = %f\n' % omega_star_spread
        fh.write(buf)

print '*****************************************'
with open('finish.txt', 'a+') as fh:
    buf = '*****************************************\n'
    fh.write(buf)

#### DEFINE SPEED and GYROFREQUENCY and GYRORADIUS
#######################################################################

#######################################################################
#### plot

init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)
plt.plot(xcm,yy ,linestyle='-',linewidth=1,color='b',label='density')
plt.scatter(xcm[x_point_index_in_plot],yy[x_point_index_in_plot],marker="o",linewidth=1,color='g',label='measured point' )
plt.xlabel(u'x (cm)')
plt.ylabel(u'density')
#plt.ylabel(r'$\omega_{\mathrm{fit}} / (\omega_*/(1+k_y^2\rho_s^2)) $',fontsize=1.5*plt.rcParams['font.size'])
#plt.title(u'Drift wave frequency')
plt.gca().legend(bbox_to_anchor = (0.0, 0.1))
## output resulting plot to file
#plt.ylim(0.8,1.05)
#
plt.tight_layout()
plt.savefig('foo1.png')
plt.savefig('foo1.eps')
plt.close('all')
#plt.clf()

######
init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)
plt.plot(xcm,-dlnyydx,linestyle='-',linewidth=1,color='b',label='inverse gradient length' )
magnify=10**(np.log10(max(abs(dlnyydx))*0.9/max(yypert)).astype(int))
plt.plot(xcm,yypert*magnify ,marker='.',linestyle='-',linewidth=1,color='r',label='perturbationx%d'%magnify )
plt.scatter(xcm[spread_ind],-dlnyydx[spread_ind],label='average points' )
plt.xlabel(u'x (cm)')
plt.ylabel(u'perturbation, -d(ln n)/dx [cm]')
plt.gca().legend(bbox_to_anchor = (0.0, 0.15))
plt.tight_layout()
plt.savefig('foo2.png')
plt.savefig('foo2.eps')
plt.close('all')
#plt.clf()

#### plot
#######################################################################



#######################################################################
#### read history file
## 1. Look for the growth rate
## 2. Perform FFT on raw signal (optional)
## 3. Linear squre fit to optimize freq, phase, mean, amplitude :fix growth rate
## 4. Perform FFT on reconstructed signal (optional)
## 5. Linear squre fit to optimize phase, mean, freq : assume that amplitude and growthrate are correct 


x_list=[]
y_list=[]
prev_lhsrhs_0 = '0000'
with open("potential_hist_1.curve", 'r') as f:
    for line in f:
        if line.lstrip().startswith('#'): #skip comment
            continue
        line =line.rstrip() #skip blank line
        if not line:
            continue 
        else: #noncomment line
            strippedline=line
            lhsrhs = strippedline.split(" ")
            l=0
            while l<len(lhsrhs): #strip white spaces in lhs
                lhsrhs[l]=lhsrhs[l].rstrip()
                lhsrhs[l]=lhsrhs[l].lstrip()
                l=l+1
            if  prev_lhsrhs_0 != lhsrhs[0]:
                x_list.append(float(lhsrhs[0]))
                y_list.append(float(lhsrhs[1]))
                prev_lhsrhs_0 = lhsrhs[0]

f.closed

#print x_list

#del x_list[-25:]
#del y_list[-25:]

#print x_list 


#make time unit to second/2/pi
#print type(x_list)
x_list[:] = [i*ref_time/2.0/np.pi for i in x_list] 
#print x_list

# number of signal points
#N = 400
#T = 1.0 / 800.0
#x = np.linspace(0.0, N*T, N)
#y = np.exp(50.0 * 1.j * 2.0*np.pi*x) #+ 0.5*np.exp(-80.0 * 1.j * 2.0*np.pi*x)

N = len(x_list)
T = (x_list[len(x_list)-1]-x_list[0]) / N
xt = np.array(x_list)
dimensional_xt = xt*2.0*np.pi
y = np.array(y_list)



##finding growth rate
#take log
logy2 = np.log(y*y)
#find zero crossing
#zero_crossings = np.where(np.diff(np.sign(np.gradient(logy2))))[0]
zero_crossings = np.where(np.gradient(np.sign(np.gradient(logy2)))<0 )[0]
extremum_dimensional_xt = dimensional_xt[zero_crossings]
extremum_logy2 = logy2[zero_crossings]

#scan the first slope

from scipy.optimize import curve_fit
def func_lin(x, aa, bb):
        return aa*x + bb

error_array_a=[]
error_array_b=[]
cutoff_index = len(extremum_dimensional_xt)-1
ind_shift = len(extremum_dimensional_xt)-27 #manual shift from begining
for ind in enumerate(extremum_dimensional_xt):
    if ind[0]>2+ind_shift:
        #print ind[0]
        xdata = extremum_dimensional_xt[0+ind_shift:ind[0]]
        ydata = extremum_logy2[0+ind_shift:ind[0]]
        popt, pcov = curve_fit(func_lin, xdata, ydata)
        perr = np.sqrt(np.diag(pcov))
        error_array_a.append(perr[0])
        error_array_b.append(perr[1])
        if len(error_array_a)>3 :
            if abs(error_array_a[-1]-error_array_a[-2])/np.average(error_array_a[0:-1]) > 1.0:
                cutoff_index = ind[0]-1
                del error_array_a[-1]
                del error_array_b[-1]
                break
        if len(error_array_b)>3 :
            if abs(error_array_b[-1]-error_array_b[-2])/np.average(error_array_b[0:-1]) > 1.0:
                cutoff_index = ind[0]-1
                del error_array_a[-1]
                del error_array_b[-1]
                break
#print cutoff_index
        

xdata = extremum_dimensional_xt[0+ind_shift:cutoff_index]
ydata = extremum_logy2[0+ind_shift:cutoff_index]


lin_fitted_logy2 = np.polyfit(xdata,ydata,1)
legend_lin_fitted_logy2 = 'y = (%g) x + (%g)' % (lin_fitted_logy2[0],lin_fitted_logy2[1])
print legend_lin_fitted_logy2
with open('finish.txt', 'a+') as fh:
    buf = '%s\n' % legend_lin_fitted_logy2
    fh.write(buf)

init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)
plt.plot(dimensional_xt,logy2,linestyle='-',linewidth=1,color='b',label='ln(phi^2)' )
plt.plot(extremum_dimensional_xt,extremum_logy2,marker='x',linewidth=1,color='g',label='extremum points' )
plt.plot(xdata,lin_fitted_logy2[0]*xdata+lin_fitted_logy2[1],color='r',linewidth=1,label=legend_lin_fitted_logy2 )
plt.gca().xaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.gca().yaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.xlabel(u'Time (s), x')
plt.ylabel(u'ln(|phi|^2), y')
plt.gca().legend(bbox_to_anchor = (0.1, 0.3))
plt.tight_layout()
plt.savefig('foo5.png')
plt.savefig('foo5.eps')
#plt.show()
plt.close('all')
#plt.clf()



########################### FFT ##################################
yf = scipy.fftpack.fft(y)
xf = scipy.fftpack.fftfreq(N, T)

xf = scipy.fftpack.fftshift(xf)
yplot = scipy.fftpack.fftshift(yf)

#print np.abs(yplot).argmax()

freqmax=xf[np.abs(yplot).argmax()]
print '|maximum  freq. fft| =', abs(freqmax),'[Hz]'
with open('finish.txt', 'a+') as fh:
    buf = '|maximum  freq. fft| = %f\n' % abs(freqmax)
    fh.write(buf)

dphase = np.pi/100
#find optimum phase shift
yvydiff2 = np.zeros(100)
for i in range(100):
    phase=dphase*i
    yv = np.real(y.max()*np.exp(abs(freqmax)*1.j*dimensional_xt+1.j*phase) )
    temperror=0.0
    for j in range(len(yv)):
        temperror = temperror+ (yv[j]-y[j])**2
    yvydiff2[i] = temperror

minphaseind=np.abs(yvydiff2).argmin()
#print minphaseind
yv = np.real(y.max()*np.exp(abs(freqmax)*1.j*dimensional_xt+1.j*dphase*minphaseind) )


yfv = scipy.fftpack.fft(yv)
xfv = scipy.fftpack.fftfreq(N, T)

xfv = scipy.fftpack.fftshift(xfv)
yplotv = scipy.fftpack.fftshift(yfv)


################# leastsq fitting #################################3
from scipy.optimize import leastsq

#find linear growthing source range
ind_source_start=0;
ind_source_end=-1;
for ind in enumerate(dimensional_xt):
    if dimensional_xt[ind[0]]>=xdata[0]:
        ind_source_start=ind[0]
        break
for ind in enumerate(dimensional_xt):
    if dimensional_xt[ind[0]]>=xdata[-1]:
        ind_source_end=ind[0]
        break
lin_dimensional_xt=dimensional_xt[ind_source_start:ind_source_end]
lin_y=y[ind_source_start:ind_source_end]

#find nonlinear growthing source range
ind_source_start=0;
ind_source_end=-1;
for ind in enumerate(dimensional_xt):
    if dimensional_xt[ind[0]]>=extremum_dimensional_xt[0]:
        ind_source_start=ind[0]
        break
for ind in enumerate(dimensional_xt):
    if dimensional_xt[ind[0]]>=extremum_dimensional_xt[-1]:
        ind_source_end=ind[0]
        break
nonlin_dimensional_xt=dimensional_xt[ind_source_start:ind_source_end]
nonlin_y=y[ind_source_start:ind_source_end]



#guess_amplitude = (y.max()-y.min())/2
guess_amplitude= lin_y[-1]/np.exp(lin_fitted_logy2[0]/2.0*(lin_dimensional_xt[-1]-lin_dimensional_xt[0]  )   )
print guess_amplitude
guess_mean = 0
guess_phase = 0
#guess_freq = freqmax #estimate from fft
guess_freq = omega_star_spline/(1.0+chi*chi) #estimate from analytic freq
guess_lin = 0

print freqmax
print guess_freq

#data fit for linear region
optimize_func = lambda z: z[3]*np.exp(lin_fitted_logy2[0]/2.0*lin_dimensional_xt)*np.cos(z[0]*lin_dimensional_xt+z[1]) + z[2] - lin_y
est_freq, est_phase, est_mean, est_amplitude = leastsq(optimize_func, [guess_freq, guess_phase, guess_mean, guess_amplitude ])[0]
lin_data_fit = est_mean + est_amplitude*np.exp(lin_fitted_logy2[0]/2.0*lin_dimensional_xt)*np.cos(est_freq*lin_dimensional_xt+est_phase) 
#data fit for non_linear region
optimize_func = lambda z: z[3]*np.exp(lin_fitted_logy2[0]/2.0*nonlin_dimensional_xt)*np.cos(z[0]*nonlin_dimensional_xt+z[1]) + z[2] - nonlin_y
nonlin_est_freq, nonlin_est_phase, nonlin_est_mean, nonlin_est_amplitude = leastsq(optimize_func, [guess_freq, guess_phase, guess_mean, guess_amplitude ])[0]
nonlin_data_fit = nonlin_est_mean + 0.5*nonlin_est_amplitude*np.exp(lin_fitted_logy2[0]/2.0*np.max(nonlin_dimensional_xt))*np.cos(nonlin_est_freq*nonlin_dimensional_xt+nonlin_est_phase) 
nonlin_data_fit_with_linfreq = nonlin_est_mean + 0.5*nonlin_est_amplitude*np.exp(lin_fitted_logy2[0]/2.0*np.max(nonlin_dimensional_xt))*np.cos(est_freq*nonlin_dimensional_xt+nonlin_est_phase) 
print 'lin freq.= ',est_freq
print 'nonlin_freq. = ',nonlin_est_freq



print '|optimized  freq. fitting| =', abs(est_freq),'[Hz]'
with open('finish.txt', 'a+') as fh:
    buf = '|optimized  freq. fitting| = %f\n' % abs(est_freq)
    fh.write(buf)

######## Check again with FFT #################
lin_N = len(lin_dimensional_xt)
lin_T = (lin_dimensional_xt[len(lin_dimensional_xt)-1]-lin_dimensional_xt[0])/2.0/np.pi/ lin_N
yfv_fit = scipy.fftpack.fft(lin_data_fit)
xfv_fit = scipy.fftpack.fftfreq(lin_N, lin_T)

xfv_fit = scipy.fftpack.fftshift(xfv_fit)
yplotv_fit = scipy.fftpack.fftshift(yfv_fit)
######## Check again with FFT #################
nonlin_N = len(nonlin_dimensional_xt)
nonlin_T = (nonlin_dimensional_xt[len(nonlin_dimensional_xt)-1]-nonlin_dimensional_xt[0])/2.0/np.pi/ nonlin_N
nonlin_yfv_fit = scipy.fftpack.fft(nonlin_data_fit)
nonlin_xfv_fit = scipy.fftpack.fftfreq(nonlin_N, nonlin_T)

nonlin_xfv_fit = scipy.fftpack.fftshift(nonlin_xfv_fit)
nonlin_yplotv_fit = scipy.fftpack.fftshift(nonlin_yfv_fit)



###########################################

legend_data_fit = r'$\omega/\omega^*$'+' = %g'%(abs(est_freq)/omega_star_spline)+'\n'+r'$\omega/\omega^*_d$'+' = %g'%(abs(est_freq)/omega_star_spline*(1.0+chi*chi))

init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)
plt.plot(dimensional_xt,y,marker='.',linestyle='-',linewidth=1,color='b',label='potential' )
#plt.plot(dimensional_xt,yv,linestyle='-',linewidth=1,color='r',label='fourier mode' )
#plt.plot(lin_dimensional_xt,lin_data_fit,marker='.',linestyle='-',linewidth=1,color='g',label=legend_data_fit )
#plt.plot(nonlin_dimensional_xt,nonlin_data_fit,marker='.',linestyle='-',linewidth=1,color='g',label=legend_data_fit )
plt.plot(nonlin_dimensional_xt,nonlin_data_fit_with_linfreq,marker='.',linestyle='-',linewidth=1,color='g',label=legend_data_fit )
plt.gca().xaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.gca().yaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.xlabel(u'Time (s)')
plt.ylabel(u'Amplitude')
plt.gca().legend(bbox_to_anchor = (0.0, 0.2))
plt.tight_layout()
plt.savefig('foo3.png')
plt.savefig('foo3.eps')
plt.close('all')
#plt.clf()



init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)
plt.plot(xf,1.0/N*np.abs(yplot),marker='.',linestyle='-',linewidth=1,color='b',label='FFT spectrum of raw signal' )
#plt.plot(xf,1.0/N*np.abs(yplotv),linestyle='-',linewidth=1,color='r',label='dominant spectrum' )
#plt.plot(xfv_fit,1.0/N*np.abs(yplotv_fit),marker='.',linestyle='-',linewidth=1,color='g',label='FFT spectrum of fitted signal' )
plt.plot(nonlin_xfv_fit,1.0/nonlin_N*np.abs(nonlin_yplotv_fit),marker='.',linestyle='-',linewidth=1,color='g',label='FFT spectrum of fitted signal' )
#xf2lim=xf[len(xf)/2+abs( len(xf)/2-np.argmax(abs(yplotv_fit)) )*3]
#xf2lim=xfv_fit[len(xfv_fit)/2+abs( len(xfv_fit)/2-np.argmax(abs(yplotv_fit)) )*3]
xf2lim=nonlin_xfv_fit[len(nonlin_xfv_fit)/2+abs( len(nonlin_xfv_fit)/2-np.argmax(abs(nonlin_yplotv_fit)) )*3]
plt.gca().xaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.gca().yaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.xlim(-abs(xf2lim),abs(xf2lim))
#plt.xlim(plt.gca().get_xlim()[0]*0.1,plt.gca().get_xlim()[1]*0.1)
plt.ylim(plt.gca().get_ylim()[0],plt.gca().get_ylim()[1]*1.3)
plt.xlabel(u'Freq. (Hz)')
plt.ylabel(u'|Y[Freq.]|')
plt.gca().legend(bbox_to_anchor = (0.1, 0.9))
plt.tight_layout()
plt.savefig('foo4.png')
plt.savefig('foo4.eps')
plt.close('all')
#plt.clf()


starting_amplitude = est_amplitude
est_growth = lin_fitted_logy2[0]/2.0
optimize_func = lambda z: z[1] + starting_amplitude*np.exp(est_growth*lin_dimensional_xt)*np.cos(z[2]*lin_dimensional_xt+z[0]) - lin_y
refine_est_phase, refine_est_mean, refine_est_freq = leastsq(optimize_func, [est_phase, est_mean, est_freq])[0]
data_fit_with_growth_refine = refine_est_mean + starting_amplitude*np.exp(est_growth*lin_dimensional_xt)*np.cos(refine_est_freq*lin_dimensional_xt+refine_est_phase) 
refine_est_growth=est_growth #fix growth rate

#starting_amplitude = est_amplitude
#est_growth = lin_fitted_logy2[0]/2.0
#optimize_func = lambda z: z[3] + starting_amplitude*np.exp(z[1]*lin_dimensional_xt)*np.cos(z[2]*lin_dimensional_xt+z[0]) - lin_y
#refine_est_phase, refine_est_growth,  refine_est_freq, refine_est_mean = leastsq(optimize_func, [est_phase, est_growth, est_freq, est_mean])[0]
#data_fit_with_growth_refine = refine_est_mean + starting_amplitude*np.exp(refine_est_growth*lin_dimensional_xt)*np.cos(refine_est_freq*lin_dimensional_xt+refine_est_phase) 
#

#est_growth = lin_fitted_logy2[0]/2.0
#optimize_func = lambda z: est_mean + z[3]*np.exp(z[1]*lin_dimensional_xt)*np.cos(z[2]*lin_dimensional_xt+z[0]) - lin_y
#refine_est_phase, refine_est_growth,  refine_est_freq, refine_est_amplitude = leastsq(optimize_func, [est_phase, est_growth, est_freq, est_amplitude])[0]
#data_fit_with_growth_refine = est_mean + refine_est_amplitude*np.exp(refine_est_growth*lin_dimensional_xt)*np.cos(refine_est_freq*lin_dimensional_xt+refine_est_phase) 





print 'est_freq. = ',est_freq
print 'refine_est_freq. = ',refine_est_freq

legend_data_fit_with_growth = r'$\gamma/\omega^*$'+' = %g'% ( lin_fitted_logy2[0]/2.0/omega_star_spline)+'\n'+r'$\gamma/\omega^*_d$'+' = %g'% ( lin_fitted_logy2[0]/2.0/omega_star_spline*(1.0+chi*chi))
init_plotting()
plt.subplot(111)
plt.gca().margins(0.1, 0.1)
#plt.plot(dimensional_xt,y,marker='.',linestyle='-',linewidth=1,color='b',label='potential' )
plt.plot(lin_dimensional_xt,lin_y,marker='.',linestyle='-',linewidth=1,color='b',label='potential' )
#plt.plot(dimensional_xt,data_fit_with_growth,linestyle='-',linewidth=1,color='g',label=legend_data_fit_with_growth )
#plt.plot(dimensional_xt,data_fit_with_growth_refine,linestyle='-',linewidth=1,color='r',label=legend_data_fit_with_growth )
plt.plot(lin_dimensional_xt,data_fit_with_growth_refine,linestyle='-',linewidth=1,color='r',label=legend_data_fit_with_growth )
#plt.plot(lin_dimensional_xt,lin_data_fit,linestyle='-',linewidth=1,color='r',label=legend_data_fit )
plt.gca().xaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.gca().yaxis.get_major_formatter().set_powerlimits((-1, 1))
plt.xlabel(u'Time (s)')
plt.ylabel(u'phi')
plt.gca().legend(bbox_to_anchor = (0.0, 0.2))
plt.tight_layout()
plt.savefig('foo6.png')
plt.savefig('foo6.eps')
plt.close('all')
#plt.clf()



with open('finish.txt', 'a+') as fh:
    buf = "te = %f\n" % (units_temperature*electron_temperature)
    fh.write(buf)
    buf = "ti = %f\n" % (units_temperature*t0_grid_func)
    fh.write(buf)
    buf = "omega_star_max   = %f\n" % (omega_star_max)
    fh.write(buf)
    buf = "omega_star_point = %f\n" % (omega_star_point)
    fh.write(buf)
    buf = "omega_star_spline = %f\n" % (omega_star_spline)
    fh.write(buf)
    if (ispread_width !=1):
        buf = "omega_star_spread = %f\n" % (omega_star_spread)
        fh.write(buf)
    buf = 'omega_star_fit/omega*_max    = %f\n'%( abs(est_freq)/omega_star_max )
    fh.write(buf)
    buf = 'omega_star_fit/omega*_point  = %f\n'%( abs(est_freq)/omega_star_point )
    fh.write(buf)
    buf = 'omega_star_fit/omega*_spline = %f\n'%( abs(est_freq)/omega_star_spline )
    fh.write(buf)
    if (ispread_width !=1):
        buf = 'omega_star_fit/omega*_spread = %f\n'%( abs(est_freq)/omega_star_spread )
        fh.write(buf)
    buf = 'omega_star_fit/omega*_max_1_chi2    = %f\n'%( abs(est_freq)/omega_star_max*(1.0+chi*chi) )
    fh.write(buf)
    buf = 'omega_star_fit/omega*_point_1_chi2  = %f\n'%( abs(est_freq)/omega_star_point*(1.0+chi*chi) )
    fh.write(buf)
    buf = 'omega_star_fit/omega*_spline_1_chi2  = %f\n'%( abs(est_freq)/omega_star_spline*(1.0+chi*chi) )
    fh.write(buf)
    buf = 'omega_star_fit/omega*_spread_1_chi2  = %f\n'%( abs(est_freq)/omega_star_spread*(1.0+chi*chi) )
    fh.write(buf)
    buf = 'gamma  = %f\n'%(refine_est_growth)
    fh.write(buf)
    buf = 'gamma/omega*_max   = %f\n'%(refine_est_growth/omega_star_max)
    fh.write(buf)
    buf = 'gamma/omega*_point = %f\n'%(refine_est_growth/omega_star_point)
    fh.write(buf)
    buf = 'gamma/omega*_spline  = %f\n'%(refine_est_growth/omega_star_spline)
    fh.write(buf)
    buf = 'gamma/omega*_max_1_chi2  = %f\n'%(refine_est_growth/omega_star_max*(1.0+chi*chi))
    fh.write(buf)
    buf = 'gamma/omega*_point_1_chi2 = %f\n'%(refine_est_growth/omega_star_point*(1.0+chi*chi))
    fh.write(buf)
    buf = 'gamma/omega*_spline_1_chi2 = %f\n'%(refine_est_growth/omega_star_spline*(1.0+chi*chi))
    fh.write(buf)
    buf = 'gamma/omega*_spread_1_chi2 = %f\n'%(refine_est_growth/omega_star_spread*(1.0+chi*chi))
    fh.write(buf)
    buf = 'gamma/omega_fit = %f\n'%(refine_est_growth/abs(est_freq))
    fh.write(buf)
    buf = 'omega/kpar = %f\n'%((abs(est_freq))/k_par)
    fh.write(buf)


print "te = " , (units_temperature*electron_temperature)
print "ti = " , (units_temperature*t0_grid_func)
print "omega_star_max    = " , (omega_star_max)
print "omega_star_point  = " , (omega_star_point)
print "omega_star_spline = " , (omega_star_spline)
if (ispread_width !=1):
    print "omega_star_spread = " , (omega_star_spread)
#print 'omega_star_FFT/omega*        = ', abs(freqmax)/omega_star_max
#print 'omega_star_FFT/omega*_point  = ', abs(freqmax)/ omega_star_point
#print 'omega_star_FFT/omega*_spline = ', abs(freqmax)/omega_star_spline
print 'omega_star_fit/omega*_max    = ',( abs(est_freq)/omega_star_max )
print 'omega_star_fit/omega*_point  = ',( abs(est_freq)/omega_star_point )
print 'omega_star_fit/omega*_spline = ',( abs(est_freq)/omega_star_spline )
if (ispread_width !=1):
    print 'omega_star_fit/omega*_spread = ',( abs(est_freq)/omega_star_spread )
print 'omega_star_fit/omega*_max_1_chi2        = ',( abs(est_freq)/omega_star_max*(1.0+chi*chi) )
print 'omega_star_fit/omega*_point_1_chi2  = ',( abs(est_freq)/omega_star_point*(1.0+chi*chi) )
print 'omega_star_fit/omega*_spline_1_chi2 = ',( abs(est_freq)/omega_star_spline*(1.0+chi*chi) )
print 'omega_star_fit/omega*_spread_1_chi2 = ',( abs(est_freq)/omega_star_spread*(1.0+chi*chi) )
print 'gamma  = ',(refine_est_growth)
print 'gamma/omega*_max     = ',(refine_est_growth/omega_star_max)
print 'gamma/omega*_point   = ',(refine_est_growth/omega_star_point)
print 'gamma/omega*_spline  = ',(refine_est_growth/omega_star_spline)

print 'gamma/omega*_max_1_chi2        = ',(refine_est_growth/omega_star_max*(1.0+chi*chi))
print 'gamma/omega*_point_1_chi2  = ',(refine_est_growth/omega_star_point*(1.0+chi*chi))
print 'gamma/omega*_spline_1_chi2 = ',(refine_est_growth/omega_star_spline*(1.0+chi*chi))
print 'gamma/omega*_spread_1_chi2 = ',(refine_est_growth/omega_star_spread*(1.0+chi*chi))
print 'gamma/omega_fit = ',(refine_est_growth/abs(est_freq))
print 'omega/kpar= ',((abs(est_freq))/k_par)


with open('finish_kparhat_chi_gamma_omega2.txt', 'wb') as fh:
    buf = "k_par_hat\tk_perp_yz*rho_s\tk_perp*rho_s\tgamma/omega*\tomega/omega*\tomega/omega*chi2\tomega/kpar\n" 
    fh.write(buf)
    buf = "%f\t%f\t%f\t%f\t%f\t%f\t%g\n" % (k_par_hat, k_perp_yz*rho_s, k_perp*rho_s ,(refine_est_growth/omega_star_spline), ( abs(est_freq)/omega_star_spline), ( abs(est_freq)/omega_star_spline*(1.0+chi*chi) ),((abs(est_freq))/k_par))
    fh.write(buf)

with open('finish_freq_growth.txt', 'wb') as fh:
    buf = "k_par_hat\tk_perp_yz*rho_s\tk_perp*rho_s\tgamma\tomega\tomega\tomega/kpar\n" 
    fh.write(buf)
    buf = "%f\t%f\t%f\t%f\t%f\t%f\t%g\n" % (k_par_hat, k_perp_yz*rho_s, k_perp*rho_s ,(refine_est_growth), ( abs(est_freq)), ( abs(est_freq) ),((abs(est_freq))/k_par))
    fh.write(buf)


import pylab
import Image
init_plotting('2x3')
f = pylab.figure()
for n, fname in enumerate(('foo1.png', 'foo2.png', 'foo3.png', 'foo4.png', 'foo5.png','foo6.png')):
     image=Image.open(fname)#.convert("L")
     arr=np.asarray(image)
     ax=f.add_subplot(2, 3, n+1)
     ax.axis('off')
     pylab.imshow(arr)
pylab.tight_layout()
pylab.savefig('foo0.png')
pylab.show()



