import os

file_path='/home/zhangc/file/projects/pycharm_files/SSL/CaGenCFReg/data/office-home/Art_list.txt'
category_file='category.txt'

with open(file_path,'r') as f:
    lines=f.readlines()

cat_f=open(category_file,'w')
cat_dict=[]
for line in lines:

    tmp_c=line.strip().split(' ')
    cat_name=tmp_c[0].split('/')[-2]
    if cat_name not in cat_dict:
        cat_dict.append(cat_name)
        cat_f.write(cat_name+'\n')
cat_f.close()
