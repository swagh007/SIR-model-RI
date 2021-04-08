#Importing Libraries 
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

data = pd.read_csv("sirdata.csv") #Importing Data

#Initial Conditions
N = data['Susceptible'].iloc[0]
I0 , R0 = 1 , 0
S0 = N -I0 -R0
gamma =1.0/10
Ro_Universal = 2.15     # Mean of values of Ro from the initial phase of epidemic
Ro = data['Ro'].iloc[-1]   # Values from the data collected
beta = Ro_Universal* gamma


#Derivative Fuction
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * I * S / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt #obtaining the derivatives

t = np.linspace(0,350,350)
y0 = S0, I0 , R0
ret = odeint(deriv, y0, t, args=(N,beta,gamma))  #Solving the differential equation
S, I, R = ret.T

### Figure-1 ###

fig = plt.figure(figsize=(14,14),edgecolor='Black')  
fig.suptitle("When NO LOCKDOWN", fontsize=30)

## Defining all the Sub-Plots ##
ax1 = fig.add_subplot(2,3,(1,3))
ax2 = fig.add_subplot(2,3,4)
ax3 = fig.add_subplot(2,3,5)
ax4 = fig.add_subplot(2,3,6)

# Ploting Subplot1 or The SIR Model #
ax1.plot(t,S ,label='Susceptible',color='Blue')
ax1.grid()
ax1.plot(t,I,label='Infected',color='Red')
ax1.plot(t,R,label='Recovered',color='Green')
ax1.set_ylabel('No. of People',fontsize=15)
ax1.legend(loc="center right")

# For maximum no. of Infections and Recovered #
style = dict(size=8, color='black',ha ='left',va ='top')

Imax = max(I)
i=I.tolist()
indexI = i.index(Imax)
ImaxSTR = str(np.int(Imax))

Rmax = max(R)
r=R.tolist()
indexR = r.index(Rmax)
RmaxSTR = str(np.int(Rmax))

ax1.text(indexI,Imax,"  Maximum Infections : " + ImaxSTR , size=9.5, color='black',ha ='left') 
ax1.plot(indexI,Imax,color='black',marker='o',markersize=3)

ax1.text(indexR,Rmax," Total Recovered : " + RmaxSTR , size=9.5, color='black',ha ='center',va ='bottom') 
ax1.plot(indexR,Rmax,color='black',marker='o',markersize=3)

## Redefining Initial conditions  ##
t = np.linspace(0,81,81)
y0 = S0, I0 , R0
ret = odeint(deriv, y0, t, args=(N,beta,gamma))
S, I, R = ret.T

# Ploting Subplot2 #
ax2.plot(t,S ,label='SIR model',color='Blue')
data.plot(kind='line' ,x='Date', y='Susceptible' , ax=ax2 , label='Data',color='Orange')
ax2.set_xlabel('Susceptible', fontsize=15)
ax2.grid()
ax2.legend(loc="center left")
ax2.set_ylabel('No. of People',fontsize=15)

# Ploting Subplot3 #
ax3.plot(t,I,label='SIR model',color='Red')
data.plot(kind='line' ,x='Date', y='Infected' , ax=ax3 , label='Data',color='Orange')
ax3.set_xlabel('Infected', fontsize=15)
ax3.grid()
ax3.legend(loc="center left")

# Ploting Subplot4 #
ax4.plot(t,R,label='SIR model',color='Green')
data.plot(kind='line' ,x='Date', y='Recovered' , ax=ax4 , label='Data',color='Orange')
ax4.set_xlabel('Recovered', fontsize=15)
ax4.grid()
ax4.legend(loc="center left")


### Figure-2 ###

fig1 = plt.figure(figsize=(14,14) ,facecolor='w',edgecolor='Black')
fig1.suptitle("LOCKDOWN Imposed(on 28 March 2020)", fontsize=25)

## Defining the Subplots ##

ax1 = fig1.add_subplot(2,3,1)
ax2 = fig1.add_subplot(2,3,2)
ax3 = fig1.add_subplot(2,3,3)
ax4 = fig1.add_subplot(2,3,4)
ax5 = fig1.add_subplot(2,3,5)
ax6 = fig1.add_subplot(2,3,6)

## Defining New Initial Conditions when LOCKDOWN imposed ##
N = data['Susceptible'].iloc[0]
I0 , R0 = 1 , 0
S0 = N - I0 -R0
gamma =1.0/10

def Deriv(y, t, N, Beta, gamma):
    
    S, I, R = y
    dSdt = -Beta(t) * I * S / N
    dIdt = Beta(t) * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def R_0(t):
 
    return Ro_Universal if t < L else Ro             

def Beta(t):
    """
    Time dependent Beta value
    """
    return R_0(t) * gamma

# Ploting Curves Without Lockdown #
L=600
t = np.linspace(0,332,332)
y0 = S0, I0 , R0
ret = odeint(Deriv, y0, t, args=(N,Beta,gamma))
S, I, R = ret.T

ax1.plot(t,S ,label='Susceptible Before',color='Blue',dashes=[1,1])
ax2.plot(t,I,label='Infected Before',color='Red',dashes=[1,1])
ax3.plot(t,R,label='Recovered Before',color='Green',dashes=[1,1])

# For maximum no. of Infections and Recovered #
Imax = max(I)
i=I.tolist()
indexI = i.index(Imax)
ImaxSTR = str(np.int(Imax))

ax2.text(indexI,Imax,"Maximum Infections : " + ImaxSTR , size=8, color='black',va='bottom') 
ax2.plot(indexI,Imax,color='black',marker='o',markersize=3)

# Ploting Curves With Lockdown #
L=28
t = np.linspace(0,488,488)
y0 = S0, I0 , R0
ret = odeint(Deriv, y0, t, args=(N,Beta,gamma))
S, I, R = ret.T

ax1.plot(t,S ,label='Susceptible After',color='Blue')
ax2.plot(t,I,label='Infected After',color='Red')
ax3.plot(t,R,label='Recovered After',color='Green')

ax1.legend(loc="center right")
ax2.legend(loc="center right")
ax3.legend(loc="center right")

ax1.set_ylabel('No. of People',fontsize=15)

ax1.grid()
ax2.grid()
ax3.grid()

# For maximum no. of Infections #
Imax = max(I)
i=I.tolist()
indexI = i.index(Imax)
ImaxSTR = str(np.int(Imax))

Rmax = max(R)
r=R.tolist()
indexR = r.index(Rmax)
RmaxSTR = str(np.int(Rmax))

ax2.text(indexI,Imax,"Maximum Infections : " + ImaxSTR , size=8, color='black', va ='bottom',ha='left') 
ax2.plot(indexI,Imax,color='black',marker='o',markersize=3)

ax3.text(indexR,Rmax," Total Recovered : " + RmaxSTR , size=9.5, color='black',ha ='center',va ='bottom') 
ax3.plot(indexR,Rmax,color='black',marker='o',markersize=3)

## Redefining Initial conditions again to fit the model ##
L=28
t = np.linspace(0,70,70)
y0 = S0, I0 , R0
ret = odeint(Deriv, y0, t, args=(N,Beta,gamma))
S, I, R = ret.T

# Ploting Subplot2 #
ax4.plot(t,S ,label='SIR Model',color='Blue')
data.plot(kind='line' ,x='Date', y='Susceptible' , ax=ax4 , label='Data',color='Orange')
ax4.set_xlabel('Susceptible', fontsize=15)
ax4.grid()
ax4.legend(loc="center left")
ax4.set_ylabel('No. of People',fontsize=15)

# Ploting Subplot3 #
ax5.plot(t,I,label='SIR Model',color='Red')
data.plot(kind='line' ,x='Date', y='Infected' , ax=ax5 , label='Data',color='Orange')
ax5.set_xlabel('Infected', fontsize=15)
ax5.grid()
ax5.legend(loc="center left")

# Ploting Subplot4 #
ax6.plot(t,R,label='SIR Model',color='Green')
data.plot(kind='line' ,x='Date', y='Recovered' , ax=ax6 , label='Data',color='Orange')
ax6.set_xlabel('Recovered', fontsize=15)
ax6.grid()
ax6.legend(loc="center left")


plt.show()
