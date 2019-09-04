# -*- coding: utf-8 -*-
"""
Created on Thu May 26 02:00:55 2016

@author: hossam
"""
import math
import numpy
import random
import time
from solution import solution


def BAT(objf,lb,ub,dim,N,Max_iteration):

    n=N;      # Population size
    #lb=-50
    #ub=50
    N_gen=Max_iteration  # Number of generations

    A=0.5;      # Loudness  (constant or decreasing)
    r=0.5;      # Pulse rate (constant or decreasing)

    Qmin=0         # Frequency minimum
    Qmax=2         # Frequency maximum


    d=dim           # Number of dimensions

    # Initializing arrays
    Q=numpy.zeros(n)  # Frequency
    v=numpy.zeros((n,d))  # Velocities
    Convergence_curve=[];

    # Initialize the population/solutions
    Sol=numpy.random.rand(n,d)*(ub-lb)+lb
    S=numpy.zeros((n,d))
    S=numpy.copy(Sol)
    Fitness=numpy.zeros(n)


    # initialize solution for the final results
    s=solution()
    print("BAT is optimizing  \""+objf.__name__+"\"")

    # Initialize timer for the experiment
    timerStart=time.time()
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")

    #Evaluate initial random solutions
    for i in range(0,n):
      Fitness[i]=objf(Sol[i,:])


    # Find the initial best solution
    fmin = min(Fitness)
    I=numpy.argmin(fmin)
    best=Sol[I,:]


    # Main loop
    for t in range (0,N_gen):

        # Loop over all bats(solutions)
        for i in range (0,n):
          Q[i]=Qmin+(Qmin-Qmax)*random.random()
          v[i,:]=v[i,:]+(Sol[i,:]-best)*Q[i]
          S[i,:]=Sol[i,:]+v[i,:]

          # Check boundaries
          Sol=numpy.clip(Sol,lb,ub)

          # Pulse rate
          if random.random()>r:
              S[i,:]=best+0.001*numpy.random.randn(d)


          # Evaluate new solutions
          Fnew=objf(S[i,:])

          # Update if the solution improves
          if ((Fnew<=Fitness[i]) and (random.random()<A) ):
                Sol[i,:]=numpy.copy(S[i,:])
                Fitness[i]=Fnew;


          # Update the current best solution
          if Fnew<=fmin:
                best=S[i,:]
                fmin=Fnew



        if (t%1==0):
            print(['At iteration '+ str(t)+ ' the best fitness is '+ str(fmin)+": BAT"])
        if (t%100==0):
            #update convergence curve
            Convergence_curve.append(fmin)
    timerEnd=time.time()
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=Convergence_curve
    s.optimizer="BAT"
    s.objfname=objf.__name__



    return s
