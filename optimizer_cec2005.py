# @Author: zorin
# @Date:   2018-03-11T15:03:55+05:30
# @Last modified by:   zorin
# @Last modified time: 2018-03-14T00:47:54+05:30



import propfa_cec2005 as propfa
import CS_cec2005 as CS
import csv
import numpy
import time




popSize=30
Iter=100000
fnum=1
dim=30

ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv"
Flag=False
CnvgHeader=[]
It=int(Iter/100)

for y in range(14,25):
    x=propfa.PFA(y+1,popSize,dim,Iter)
    # x=CS.CS(y+1,dim,popSize,Iter)
    for l in range(0,It):
    	CnvgHeader.append("Iter"+str((l+1)*100))
    with open(ExportToFile, 'a',newline='\n') as out:
        writer = csv.writer(out,delimiter=',')
        if (Flag==False): # just one time to write the header of the CSV file
            header= numpy.concatenate([["Optimizer","objfname","startTime","EndTime","ExecutionTime"],CnvgHeader])
            writer.writerow(header)
        a=numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime],x.convergence])
        writer.writerow(a)
    out.close()
    Flag=True # at least one experiment
