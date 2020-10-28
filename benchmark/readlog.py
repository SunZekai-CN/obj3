# %%
from matplotlib import pyplot as plt
import numpy as np

x='epochs'
# 'epochs' 'processes' 'batch' 'reduce'
y='right' 
# 'loss' 'right' 'accuracy' 
way='average'
# %%
def preprocessing(line):
    new=line.replace('\n','').replace(',','').replace('=','').replace(':','').replace('/','').replace('(','').replace(')','').replace('%','')
    lineData=new.strip().split(' ')
    return lineData
def line1(lineData):
    setting={}
    setting['arch']=lineData[2] 
    setting['epochs']=lineData[4]  
    setting['processes']=lineData[7]
    setting['batch']=lineData[14]
    setting['order']=lineData[20]
    setting['reduce']=lineData[23]
    return setting
def line2(lineData):
    training={}
    training['sum']=float(lineData[3])
    training['average']=float(lineData[7])
    training['max']=float(lineData[11])
    training['min']=float(lineData[15])
    training['median']=float(lineData[19])
    return training
def line3(lineData,test):
    test['loss']=float(lineData[4])
    test['right']=int(lineData[6])
    test['cases']=int(lineData[8])
    test['accuracy']=float(lineData[9])
    return test
f = open("log.txt")                
line = f.readline()        
setting=[]
test=[]        
while line:               
    lineData=preprocessing(line)
    setting.append(line1(lineData))
    line = f.readline() 
    lineData=preprocessing(line)
    training=line2(lineData)
    line = f.readline()
    lineData=preprocessing(line)
    test.append(line3(lineData,training))
    line = f.readline()
    line = f.readline() 
f.close()  
# %%
def find_same_setting(setting,idx):
    i = idx
    cases=[]
    key1='arch'
    key2='order'
    while(i<len(setting)):
        flag = False
        for key,value in setting[i].items():
            if value != setting[idx][key]:
                flag = True
        if flag:
            break   
        cases.append(i)
        i+=1
    while(i<len(setting)):
        if (setting[idx][key1]==setting[i][key1])&(setting[idx][key2]==setting[i][key2]):
            break
        i += 1
    return i,np.array(cases)
def calculate(all_data,key,setting,value1,value2):
    results=[]
    for idx in range(len(all_data)):
        if (setting[idx]['arch']==value1)&(setting[idx]['order']==value2):
            break
    if (setting[idx]['arch']!=value1)|(setting[idx]['order']!=value2):
        return results
    next=idx
    while next<len(all_data):
        result={}
        for key1,value in setting[next].items():
            result[key1]=value
        next,cases=find_same_setting(setting,next)
        data=[]
        for p in cases:
            data.append(all_data[p][key])
        result['max']=max(data)
        result['min']=min(data)
        data=np.array(data)
        result['average']=np.mean(data)
        result['median']=np.median(data)
        results.append(result)
    return results    
def clasify(test,setting):
    lines=[]
    lines.append(calculate(test,y,setting,'ff-net','y'))
    lines.append(calculate(test,y,setting,'ff-net','n'))
    lines.append(calculate(test,y,setting,'conv-net','y'))
    lines.append(calculate(test,y,setting,'conv-net','n'))
    all_casee=[]
    for line in lines:
        cases=[]
        for case in line:
            example={}
            for key,value in case.items():
                if (key=='arch')|(key=='order')|(key=='max')|(key=='min')|(key=='average')|(key=='median')|(key ==x):
                    continue
                example[key]=value
            if example not in cases:
                cases.append(example)
        all_casee.append(cases)
    return lines,all_casee
def each_line(line,case):
    xs=[]
    ys=[]
    for data in line:
        flag = False
        for key,value in case.items():
            if data[key]!=value:
                flag=True
        if flag:
            continue
        xs.append(data[x])
        ys.append(data[way])
    return np.array(xs),np.array(ys)
def draw(test,setting):
    lines,cases=clasify(test,setting)
    for case in cases[0]:
        title=''
        for key,value in case.items():
            if title == '':
                title+=(key+'='+value)  
            else:
                title+=(';'+key+'='+value)
        plt.title(title)
        for i in range(4):
            xs,ys=each_line(lines[i],case)
            if i == 0:
                plt.plot(xs,ys,label="arch=ff,in order")
            if i == 1:
                plt.plot(xs,ys,label="arch=ff,random")
            if i == 2:
                plt.plot(xs,ys,label="arch=conv,in order")
            if i == 3:
                plt.plot(xs,ys,label="arch=conv,random")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.ylim(0,test[0]['cases'])
        plt.legend()
        plt.show()
draw(test,setting)

# %%
