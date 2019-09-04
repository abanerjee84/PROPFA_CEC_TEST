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





def alpha_new(alpha,NGen):
    #% alpha_n=alpha_0(1-delta)^NGen=10^(-4);
    #% alpha_0=0.9
    delta=1-(10**(-4)/0.9)**(1/NGen);
    alpha=(1-delta)*alpha
    return alpha



def PFA(objf,lb,ub,dim,n,MaxGeneration):
    pop=n
    #General parameters

    #n=50 #number of fireflies
    dim=30 #dim
    #lb=-50
    #ub=50
    #MaxGeneration=500

    #FFA parameters
    alpha=0.50  # Randomness 0--1 (highly random)
    betamin=0.50  # minimum value of beta
    gamma=1   # Absorption coefficient
    delta=0.01
    # delta2=(ub-lb)/MaxGeneration

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


    print("FFA is optimizing  \""+objf.__name__+"\"")

    timerStart=time.time()
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for k in range (0,MaxGeneration):     # start iterations

        #% This line of reducing alpha is optional
        #alpha=alpha_new(alpha,MaxGeneration);
        Lightnprev=Lightn
        #% Evaluate new solutions (for all n fireflies)
        for i in range(0,n):
            zn[i]=objf(ns[i,:])
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
        if (k%10!=0):
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
                       per=((k/MaxGeneration)*100)/50
                       per2=numpy.heaviside(per-1,0.5)
                       ratA=(numpy.absolute(Lightn[i])-numpy.absolute(Lightnprev[i]))/max(numpy.absolute(Lightn[i]),numpy.absolute(Lightnprev[i]))
                       ratB=(numpy.absolute(Lightn[j])-numpy.absolute(Lightn[i]))/max(numpy.absolute(Lightn[j]),numpy.absolute(Lightn[i]))
                       ratC=(numpy.absolute(fbest)-numpy.absolute(Lightn[i]))/max(numpy.absolute(fbest),numpy.absolute(Lightn[i]))
                       ratAvg=(ratA+ratB+ratC)/3
                       scale2=numpy.absolute(ub-lb)
                       delta=r2/10
                       if (Lightnprev[i]==Lightn[i]):
                           alpha=1
                       else:
                           alpha=(delta)*ratAvg*numpy.exp(-k*per2)
                        #    alpha=1*ratAvg*1

                       if (Lightnprev[i]==Lightn[i]):
                           gamma=1
                       else:
                           gamma=1*(ratB/ratC)

                       beta0=1
                       beta=(beta0-betamin)*numpy.exp(-gamma*r**2)+betamin
                       beta2=(beta0-betamin)*numpy.exp(-gamma*r2**2)+betamin
                       tmpf=alpha*(numpy.random.rand(dim)-0.5)*1

                       #ns[i,:]=ns[i,:]*(1-beta)+nso[j,:]*beta+tmpf

                       ns[i,:]=ns[i,:]+(beta*(nso[j,:]-ns[i,:]))+(beta2*(nso[0,:]-ns[i,:]))+tmpf
                    #    ns=numpy.clip(ns, lb, ub)
        else:
            bet=3/2;
            sigma=(math.gamma(1+bet)*math.sin(math.pi*bet/2)/(math.gamma((1+bet)/2)*bet*2**((bet-1)/2)))**(1/bet);
            u=numpy.random.randn(dim)*sigma
            v=numpy.random.randn(dim)
            step=u/abs(v)**(1/bet)
            stepsize=0.001*(step*(ns[i,:]-ns[0,:]))
            lastn=n-int(pop/2)
            for t in range(lastn,n):
                ran2=numpy.random.random_sample()
                for y in range (dim):
                    ns[t,y]=ns[t,y]+stepsize[y]*numpy.random.random_sample()
                    # delta2=(ns[0,y] +ns[1,y])*0.5
                    # # delta2=delta
                    # # print (ns[0,y],ns[lastn,y],delta2)
                    # ran=numpy.random.uniform(0, delta2)
                    # if (ran2<0.5):
                    #     ns[t,y]=ns[0,y]-ran
                    #     # ns[t,y]=numpy.random.uniform(lb,ub)
                    # else:
                    #     ns[t,y]=ns[0,y]+ran
                    #     # ns[t,y]=numpy.random.uniform(lb,ub)



        ns=numpy.clip(ns, lb, ub)
        IterationNumber=k
        BestQuality=fbest

        if (k%1==0):
               print(['At iteration '+ str(k)+ ' the best fitness is '+ str(BestQuality)+": PFA"+" :"+str(objf)])
        if (k%100==0):
               convergence.append(fbest)
    #
       ####################### End main loop
    convergence.append(Lightn[0])
    convergence.append(Lightn[6])
    convergence.append(Lightn[12])
    convergence.append(Lightn[18])
    convergence.append(Lightn[24])
    timerEnd=time.time()
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence
    s.optimizer="PFA"
    s.objfname=objf.__name__

    return s
