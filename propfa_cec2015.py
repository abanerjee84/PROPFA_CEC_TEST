# -*- coding: utf-8 -*-
"""
Created on Sun May 29 00:49:35 2016

@author: hossam
"""

#% ======================================================== %
#% Files of the Matlab programs included in the book:       %
#% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
#% Second Edition, Luniver Press, (2010).   www.luniver.com %
#% ======================================================== %
#
#% -------------------------------------------------------- %
#% Firefly Algorithm for constrained optimization using     %
#% for the design of a spring (benchmark)                   %
#% by Xin-She Yang (Cambridge University) Copyright @2009   %
#% -------------------------------------------------------- %

import numpy
import math
import time
from solution import solution
# from cec2013lsgo.cec2013 import Benchmark
from cec2013single.cec2013 import Benchmark



def alpha_new(alpha,NGen):
    #% alpha_n=alpha_0(1-delta)^NGen=10^(-4);
    #% alpha_0=0.9
    delta=1-(10**(-4)/0.9)**(1/NGen);
    alpha=(1-delta)*alpha
    return alpha



def PFA(objf,n,MaxGeneration):

    dim=30
    bench = Benchmark()
    info = bench.get_info(objf,dim)
    # dim = info['dimension']
    ub = info['upper']
    lb = info['lower']
    optimum = info['best']
    #General parameters

    #n=50 #number of fireflies
    # dim=10000 #dim
    #lb=-50
    #ub=50
    #MaxGeneration=500

    #FFA parameters
    alpha=0.50  # Randomness 0--1 (highly random)
    betamin=0.50  # minimum value of beta
    gamma=1   # Absorption coefficient



    zn=numpy.ones(n)
    zn.fill(float("inf"))


    #ns(i,:)=Lb+(Ub-Lb).*rand(1,d);
    ns=numpy.random.uniform(0,1,(n,dim)) *(ub-lb)+lb
    Lightn=numpy.ones(n)
    Lightn.fill(float("inf"))
    Lightnprev=numpy.ones(n)
    Lightnprev.fill(float("inf"))

    #[ns,Lightn]=init_ffa(n,d,Lb,Ub,u0)

    convergence=[]
    s=solution()


    print("PFA is optimizing F"+str(objf))

    timerStart=time.time()
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for k in range (0,MaxGeneration):     # start iterations

        #% This line of reducing alpha is optional
        #alpha=alpha_new(alpha,MaxGeneration);
        Lightnprev=Lightn
        #% Evaluate new solutions (for all n fireflies)
        fun_fitness = bench.get_function(objf)
        for i in range(0,n):
            zn[i]=fun_fitness(ns[i,:])
            Lightn[i]=zn[i]





        # Ranking fireflies by their light intensity/objectives


        Lightn=numpy.sort(zn)
        Index=numpy.argsort(zn)
        ns=ns[Index,:]


        #Find the current best
        nso=ns
        Lighto=Lightn
        nbest=ns[0,:]
        Lightbest=Lightn[0]

        #% For output only
        fbest=Lightbest;

        #% Move all fireflies to the better locations
    #    [ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,...
    #          Lightbest,alpha,betamin,gamma,Lb,Ub);
        scale=numpy.ones(dim)*abs(ub-lb)
        for i in range (0,n):
            # The attractiveness parameter beta=exp(-gamma*r)
            for j in range(0,n):
                # r=numpy.sqrt(numpy.sum((ns[i,:]-ns[j,:])**2));
                # r2=numpy.sqrt(numpy.sum((ns[i,:]-ns[0,:])**2));
                r=numpy.sum((ns[i,:]-ns[j,:]))
                r2=numpy.sum((ns[0,:]-ns[j,:]))
                #r=1
                # Update moves
                if Lightn[i]>Lighto[j]: # Brighter and more attractive
                   # PropFA parameters
                   per=((k/MaxGeneration)*100)/75
                   per2=numpy.heaviside(per-1,0.5)
                   ratA=(numpy.absolute(Lightn[i])-numpy.absolute(Lightnprev[i]))/max(numpy.absolute(Lightn[i]),numpy.absolute(Lightnprev[i]))
                   ratB=(numpy.absolute(Lightn[j])-numpy.absolute(Lightn[i]))/max(numpy.absolute(Lightn[j]),numpy.absolute(Lightn[i]))
                   ratC=(numpy.absolute(fbest)-numpy.absolute(Lightn[i]))/max(numpy.absolute(fbest),numpy.absolute(Lightn[i]))
                   ratAvg=(ratA+ratB+ratC)/3
                   scale2=numpy.absolute(ub-lb)

                   if (Lightnprev[i]==Lightn[i]):
                       alpha=10
                   else:
                       r3=numpy.sum((ns[0,:]-ns[n-1,:]))
                       alpha=(r2/10000)*ratAvg*numpy.exp(-k*per2)

                   if (Lightnprev[i]==Lightn[i]):
                       gamma=1
                   else:
                       gamma=(ratB/ratC)

                   beta0=1
                   beta=(beta0-betamin)*numpy.exp(-gamma*r**2)+betamin
                   beta2=(beta0-betamin)*numpy.exp(-gamma*r2**2)+betamin
                   tmpf=alpha*(numpy.random.rand(dim)-0.5)*scale2

                   #ns[i,:]=ns[i,:]*(1-beta)+nso[j,:]*beta+tmpf
                   ns[i,:]=ns[i,:]+(beta*(nso[j,:]-ns[i,:]))+(beta2*(nso[0,:]-ns[i,:]))+tmpf
        #ns=numpy.clip(ns, lb, ub)



        IterationNumber=k
        BestQuality=fbest

        if (k%1==0):
               print(['At iteration '+ str(k)+ ' the best fitness is '+ str(BestQuality)+": PFA"+" :"+str(objf)])
        if (k%1000==0):
               convergence.append(fbest)
    #
       ######5################# End main loop
    convergence.append(Lightn[0])
    convergence.append(Lightn[int(n/2)])
    convergence.append(Lightn[n-1])
    convergence.append(numpy.sum(Lightn)/n)
    convergence.append(numpy.std(Lightn))
    timerEnd=time.time()
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence
    s.optimizer="PFA"
    s.objfname="F"+str(objf)

    return s
