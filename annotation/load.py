import os

f = open('classic.txt','r')
train_list = open('test.list','a+')

dict_ucf = {}
dict = {}
list_label = []
for i in f:
    pre_list = open('c3dtrain.list', 'r')
    classic = i.split()[0]
    pre_class = i.split()[1]
    name = i.split()[2]
    dict[name] = classic
    dict_ucf[pre_class] = classic
    list_label.append(pre_class)

f.close()
path = os.getcwd()

sum = 0
data = '/data1/home/linzhikun2/THUMOS14/test'
for i in os.listdir(os.path.join(path,'test_list')):
    classic_name = i.split('_')[0]
    classic_num = dict[classic_name]
    if classic_name!='Ambiguous':
        list = open(os.path.join(path,'test_list','%s'%i),'r')
        for i in list:
            video_name = i.split()[0]
            start_time = i.split()[1]
            end_time = i.split()[2]
            train_list.write(os.path.join(data, video_name) + ','+classic_num+','+ start_time + ' ' + end_time + '\n')
            sum += 1
            print (sum)
        
print sum