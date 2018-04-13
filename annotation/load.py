import os

f=open('classic.txt','r')
train_list=open('train.list','a+')

dict_ucf={}
dict={}
list_label=[]
for i in f:
    pre_list = open('c3dtrain.list', 'r')
    classic=i.split()[0]
    pre_class=i.split()[1]
    name=i.split()[2]
    dict[name]=classic
    dict_ucf[pre_class]=classic
    list_label.append(pre_class)

pre_list = open('c3dtrain.list', 'r')
for j in pre_list:
     path=j.split()[0]
     label=j.split()[1]
     if label in list_label:
         if int(dict_ucf[label])>=0 and int(dict_ucf[label])<=19 :
              train_list.write(path + ','+dict_ucf[label]+'\n')

f.close()
path=os.getcwd()

sum=0
data='/data1/home/linzhikun2/THUMOS14/train'
for i in os.listdir(os.path.join(path,'train_list')):
    classic_name=i.split('_')[0]
    classic_num=dict[classic_name]
    if classic_name!='Ambiguous':
        last_name=1
        list=open(os.path.join(path,'train_list','%s'%i),'r')
        for i in list:
            video_name= i.split()[0]
            start_time=i.split()[1]
            end_time=i.split()[2]
            if  last_name==1:
                train_list.write(os.path.join(data, video_name) + ','+classic_num+','+ start_time + ' ' + end_time)
                last_name=video_name
                sum+=1
            elif video_name==last_name :
                train_list.write(' '+start_time+' '+end_time)
                last_name = video_name
                sum+=1
            else:
                train_list.write('\n'+os.path.join(data,video_name)+ ','+classic_num+','+start_time+' '+end_time)
                last_name = video_name
                sum+=1
        train_list.write('\n')
        
print sum