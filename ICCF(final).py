#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:51:20 2022

@author: jamiemcdoanld
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
from random import sample
from scipy.stats import norm



def mean(sample):
    mean = (sum(sample))/len(sample)
    return mean
def standard_dev(sample, mean):
    standlist = []
    
    for i in range(len(sample)):
        standlist.append((sample[i]-mean)**2)
    stdev=(sum(standlist)/len(sample))**0.5
   
    return stdev


bestlaglist=[]
bestccflist=[]

nbestlaglist=[]
nbestccflist=[]
for k in range(100):

    date_1, price_1=np.loadtxt('UK_cases.csv', delimiter=",", unpack=True, skiprows=1, usecols=(0,1), comments='#', dtype=str)

    date_2, price_2=np.loadtxt('TSCO.L.csv', delimiter=",", unpack=True, skiprows=1, usecols=(0,1), comments='#', dtype=str)

    

    price_1=price_1.tolist()
    
    price_2=price_2.tolist()
    
    date_1=date_1.tolist()

    date_2=date_2.tolist()

    for i in range(len(price_1)):
        price_1[i]=float(price_1[i])
    
    for i in range(len(price_2)):
        price_2[i]=float(price_2[i])


    
    x_axis1=[]    

    x_axis2=[] 
    
    for i in range(len(date_1)):
        date_string = date_1[i]
        date1 = datetime.datetime.strptime(date_string, "%d/%m/%Y")
        timestamp = datetime.datetime.timestamp(date1)
    
        x_axis1.append(round(timestamp/86400))
        
    for i in range(len(date_2)):
        date_string = date_2[i]
        date2 = datetime.datetime.strptime(date_string, "%d/%m/%Y")
        timestamp = datetime.datetime.timestamp(date2)
    
        x_axis2.append(round(timestamp/86400))
    
    
    #x_axis1=x_axis1[int(len(x_axis1)*0.4):int(len(x_axis1)*0.6)]
    
    #x_axis2=x_axis2[int(len(x_axis2)*0.4):int(len(x_axis2)*0.6)]
    
    #price_1=price_1[int(len(price_1)*0.4):int(len(price_1)*0.6)]
    
    #price_2=price_2[int(len(price_2)*0.4):int(len(price_2)*0.6)]
    
    samplerange1=int(len(x_axis1)*0.7)
    samplerange2=int(len(x_axis2)*0.7)
    
    randomdates1=sample(x_axis1, samplerange1)
    randomdates2=sample(x_axis2, samplerange2)
   
    randomdates1.sort()
    randomdates2.sort()
    
    
   
    index1=[]
    index2=[]
    
    
    for i in range(len(randomdates1)):
        index1.append(x_axis1.index(randomdates1[i]))
    for i in range(len(randomdates2)):
        index2.append(x_axis2.index(randomdates2[i]))
    
    randomprice1=[]
    randomprice2=[]
    for i in range(len(index1)):   
        j=index1[i]
        randomprice1.append(price_1[j])
    for i in range(len(index2)):
        j=index2[i]
        randomprice2.append(price_2[j])
    
    price_1=[]
    for i in range(len(randomprice1)):
        
        price_1.append(randomprice1[i])
    price_2=[]
    for i in range(len(randomprice2)):
        
        price_2.append(randomprice2[i])
    x_axis1=[]
    for i in range(len(randomdates1)):
       
        x_axis1.append(randomdates1[i])
    x_axis2=[]
    for i in range(len(randomdates2)):
        
        x_axis2.append(randomdates2[i])
    
    
    y_axis1=[]
    for i in range(len(price_1)):
        y_axis1.append(float(price_1[i]))
    
    y_axis2=[]
    for i in range(len(price_2)):
        y_axis2.append(float(price_2[i]))

    
    total1 = sum(y_axis1)
    
    mean1=total1/len(y_axis1)

    total2 = sum(y_axis2)
    
    mean2=total2/len(y_axis2)
    



    sd_list1 = []

    sd_list2 = []

    for i in range(len(y_axis1)):
        sd_list1.append((float(y_axis1[i])-mean1)**2)

    
    sd1 = ((sum(sd_list1))/len(y_axis1))**0.5

    for i in range(len(y_axis2)):
        sd_list2.append((float(y_axis2[i])-mean2)**2)


    sd2 = ((sum(sd_list2))/len(y_axis2))**0.5


    
   
    
    
    for i in range(int(len(date_1)*0.95)):
        if x_axis1[i]+1 != x_axis1[i+1]:
            x_axis1.insert(i+1,x_axis1[i]+1)
       
            
            random_num=np.random.normal(loc=0.0, scale=1)
            local1=[]
            for j in range(5):
                local1.append(y_axis1[i+j])
                local1.insert(0, y_axis1[i-j])
            localmean1=mean(local1)
            lsd1=standard_dev(local1, localmean1)    
            y_axis1.insert(i+1,(float(y_axis1[i])+(x_axis1[i+1]-x_axis1[i])*((float(y_axis1[i+1])-float(y_axis1[i]))/(x_axis1[i+2]-x_axis1[i])))+(lsd1*random_num))

   


   
    
    for i in range(int(len(date_2)*0.95)):
        if x_axis2[i]+1 != x_axis2[i+1]: 
            x_axis2.insert(i+1,x_axis2[i]+1) 
            
           
            random_num=np.random.normal(loc=0.0, scale=1)
            local2=[]
            for j in range(5):
                local2.append(y_axis2[i+j])
                local2.insert(0, y_axis2[i-j])
            localmean2=mean(local2)
            lsd2=standard_dev(local2, localmean2)  
            y_axis2.insert(i+1,(float(y_axis2[i])+(x_axis2[i+1]-x_axis2[i])*((float(y_axis2[i+1])-float(y_axis2[i]))/(x_axis2[i+2]-x_axis2[i])))+lsd2*random_num)

        
    if len(x_axis1) > len(x_axis2):
        x_axis1n=[]
        for i in range(len(x_axis2)):
            x_axis1n.append(x_axis1[i])
        x_axis1=[]
        for i in range(len(x_axis1n)):
            x_axis1.append(x_axis1n[i])
            
        y_axis1n=[]
        for i in range(len(y_axis2)):
            y_axis1n.append(y_axis1[i])
        y_axis1=[]
        for i in range(len(y_axis1n)):
            y_axis1.append(y_axis1n[i])
    
    if len(x_axis1) < len(x_axis2):
        x_axis2n=[]
        for i in range(len(x_axis1)):
            x_axis2n.append(x_axis2[i])
        x_axis2=[]
        for i in range(len(x_axis2n)):
            x_axis2.append(x_axis2n[i])
        
        y_axis2n=[]
        for i in range(len(y_axis1)):
            y_axis2n.append(y_axis2[i])
        y_axis2=[]
        for i in range(len(y_axis2n)):
            y_axis2.append(y_axis2n[i])
    
    
    
    
    
    


    total1 = sum(y_axis1)
    
    mean1=total1/len(y_axis1)

    total2 = sum(y_axis2)
    
    mean2=total2/len(y_axis2)
    


    sd_list1 = []

    sd_list2 = []

    for i in range(len(y_axis1)):
        sd_list1.append((y_axis1[i]-mean1)**2)

    
    sd1 = ((sum(sd_list1))/len(y_axis1))**0.5

    for i in range(len(y_axis2)):
        sd_list2.append((y_axis2[i]-mean2)**2)


    sd2 = ((sum(sd_list2))/len(y_axis2))**0.5

    def CCFp(price1, price2, mean1, mean2, sd1, sd2, delay):
    
    
        ccf_list=[]
        for a in range(len(price1)-delay):
            ccf_list.append((float(price1[a]) - mean1)*(float(price2[a+delay])-mean2))
        ccf=sum(ccf_list)/(sd1*sd2*(len(price1)-delay-1))
    
        return ccf

    def CCFn(price1, price2, mean1, mean2, sd1, sd2, delay):
    
        ccf_list=[]
        for a in range(len(price1)-delay):
            ccf_list.append((float(price2[a]) - mean2)*(float(price1[a+delay])-mean1))
        ccf=sum(ccf_list)/(sd1*sd2*(len(price1)-delay-1))
    
        return ccf

    

    ccf_y_axis=[]
    ccf_x_axis=[]
    for i in range(int(len(y_axis1)*0.9)):
        ccf_y_axis.append(CCFp(y_axis1,y_axis2,mean1,mean2,sd1,sd2,i))
        ccf_x_axis.append(i)

    for i in range(int(len(y_axis1)*0.9)):
        ccf_y_axis.insert(0,CCFn(y_axis1,y_axis2,mean1,mean2,sd1,sd2,i))
        ccf_x_axis.insert(0,-i)
    
    ccf_x_axis=ccf_x_axis[int(len(ccf_x_axis)*0.2):int(len(ccf_x_axis)*0.8)]
    
    ccf_y_axis=ccf_y_axis[int(len(ccf_y_axis)*0.2):int(len(ccf_y_axis)*0.8)]
    
    #plt.plot(ccf_x_axis, ccf_y_axis)
    #plt.xlabel('Offset/Time')
    #plt.ylabel('correlation coefficient')
    #plt.title('Auto-correlation EasyJet')
    #plt.show()
    
    
    ccfs=[]
    for i in range(len(ccf_y_axis)):
        ccfs.append(round(ccf_y_axis[i],3))

    seventymax=[]
    for i in range(len(ccfs)):
        if round(ccfs[i],2)==round(max(ccf_y_axis)*0.7, 2):
            seventymax.append(i)
    
    
    leftside=[]
    rightside=[]

    for i in range(len(seventymax)):
        if seventymax[i]<ccf_y_axis.index(max(ccf_y_axis)):
            leftside.append(seventymax[i])
        elif seventymax[i]>ccf_y_axis.index(max(ccf_y_axis)):
            rightside.append(seventymax[i])

    

    seventymaxccf=round(max(ccf_y_axis)*0.75, 2)+0.01

    if len(rightside)<1 or len(leftside)<1:
        while len(rightside)<1 or len(leftside)<1:
            seventymax= []
            seventymaxccf=seventymaxccf-0.01
            if seventymaxccf>0:
                for i in range(len(ccfs)):
                    if round(ccfs[i],2)==seventymaxccf:
                        seventymax.append(i)
                
        
                leftside=[]
                rightside=[]

                for i in range(len(seventymax)):
                    if seventymax[i]<ccf_y_axis.index(max(ccf_y_axis)):
                        leftside.append(seventymax[i])
                    elif seventymax[i]>ccf_y_axis.index(max(ccf_y_axis)):
                            rightside.append(seventymax[i])
        
            elif seventymaxccf<=0:
                leftside=[-10]
                rightside=[-10]
        
    
    leftmean = sum(leftside)/len(leftside)
    rightmean = sum(rightside)/len(rightside)
        
    bestlag=round((leftmean+rightmean)/2)
    if bestlag!=-10:
        
        bestlaglist.append(ccf_x_axis[bestlag])
        bestccflist.append(ccf_y_axis[bestlag])
        
        
    seventymin=[]
    for i in range(len(ccfs)):
        if round(ccfs[i],2)==round(min(ccf_y_axis)*0.7, 2):
            seventymin.append(i)
    
    
    leftside=[]
    rightside=[]

    for i in range(len(seventymin)):
        if seventymin[i]<ccf_y_axis.index(min(ccf_y_axis)):
            leftside.append(seventymin[i])
        elif seventymin[i]>ccf_y_axis.index(min(ccf_y_axis)):
            rightside.append(seventymin[i])

    

    seventyminccf=round(min(ccf_y_axis)*0.75, 2)+0.01

    if len(rightside)<1 or len(leftside)<1:
        while len(rightside)<1 or len(leftside)<1:
            seventymin= []
            seventyminccf=seventyminccf+0.01
            if seventyminccf<0:
                for i in range(len(ccfs)):
                    if round(ccfs[i],2)==seventyminccf:
                        seventymin.append(i)
                
        
                leftside=[]
                rightside=[]

                for i in range(len(seventymin)):
                    if seventymin[i]<ccf_y_axis.index(min(ccf_y_axis)):
                        leftside.append(seventymin[i])
                    elif seventymin[i]>ccf_y_axis.index(min(ccf_y_axis)):
                            rightside.append(seventymin[i])
        
            elif seventyminccf>=0:
                leftside=[-10]
                rightside=[-10]
        
    
    leftmean = sum(leftside)/len(leftside)
    rightmean = sum(rightside)/len(rightside)
        
    bestlag=round((leftmean+rightmean)/2)
    if bestlag!=-10:
        
        nbestlaglist.append(ccf_x_axis[bestlag])
        nbestccflist.append(ccf_y_axis[bestlag])
'''
b_yaxis=[1]
bestlaglist.sort()
for i in range(len(bestlaglist)-1):
    if bestlaglist[i]==bestlaglist[i+1]:
        b_yaxis[len(b_yaxis)-1]=b_yaxis[len(b_yaxis)-1]+1
    elif bestlaglist[i]!=bestlaglist[i+1]:
        b_yaxis.append(1)
sum_by=sum(b_yaxis)
for i in range(len(b_yaxis)):
    b_yaxis[i]=b_yaxis[i]/sum_by


b_xaxis=[]
for i in range(len(bestlaglist)-1):
    if bestlaglist[i]!=bestlaglist[i+1]:
        b_xaxis.append(bestlaglist[i])
b_xaxis.append(bestlaglist[len(bestlaglist)-1])

    
plt.bar(b_xaxis, b_yaxis)
bestlag=mean(bestlaglist)
lagerror=(standard_dev(bestlaglist, bestlag))/(len(bestlaglist))**0.5


bestccf=mean(bestccflist)
ccferror=(standard_dev(bestccflist, bestccf))/(len(bestccflist))**0.5

x_axis = np.arange(b_xaxis[0], b_xaxis[len(b_xaxis)-1], 0.01)
  
  
plt.plot(x_axis, norm.pdf(x_axis, bestlag, (lagerror*(len(bestlaglist))**0.5)), color='red')
plt.xlabel('offset')
plt.ylabel('probability distribution')
plt.show()



print("lag=", bestlag,"error=", lagerror)
print("ccf=", bestccf, "error=",ccferror )
'''

nb_yaxis=[1]
nbestlaglist.sort()
for i in range(len(nbestlaglist)-1):
    if nbestlaglist[i]==nbestlaglist[i+1]:
        nb_yaxis[len(nb_yaxis)-1]=nb_yaxis[len(nb_yaxis)-1]+1
    elif nbestlaglist[i]!=nbestlaglist[i+1]:
        nb_yaxis.append(1)
nsum_by=sum(nb_yaxis)
for i in range(len(nb_yaxis)):
    nb_yaxis[i]=nb_yaxis[i]/nsum_by


nb_xaxis=[]
for i in range(len(nbestlaglist)-1):
    if nbestlaglist[i]!=nbestlaglist[i+1]:
        nb_xaxis.append(nbestlaglist[i])
nb_xaxis.append(nbestlaglist[len(nbestlaglist)-1])

    
plt.bar(nb_xaxis, nb_yaxis)
nbestlag=mean(nbestlaglist)
nlagerror=(standard_dev(nbestlaglist, nbestlag))/(len(nbestlaglist))**0.5


nbestccf=mean(nbestccflist)
nccferror=(standard_dev(nbestccflist, nbestccf))/(len(nbestccflist))**0.5

nx_axis = np.arange(nb_xaxis[0], nb_xaxis[len(nb_xaxis)-1], 0.01)
  
  
plt.plot(nx_axis, norm.pdf(nx_axis, nbestlag, (nlagerror*(len(nbestlaglist))**0.5)), color='red')
plt.xlabel('offset')
plt.ylabel('probability distribution')
plt.show()

print("lag=", nbestlag,"error=", nlagerror)
print("ccf=", nbestccf, "error=",nccferror )
