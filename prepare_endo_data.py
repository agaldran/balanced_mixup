import os
import os.path as osp
from PIL import Image
from torchvision.transforms import Resize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

rsz = Resize((512,640))
rootDir = 'data/labeled-images/'
outDir = 'data/images/'
os.makedirs(outDir, exist_ok=True)

for dirName, subdirList, fileList in os.walk(rootDir):
    print(dirName)
    for fname in fileList:
        if 'jpg' in fname:
            img = Image.open(osp.join(dirName, fname))
            img_res = rsz(img)
            img_res.save(osp.join(outDir,fname))

df_all = pd.read_csv('data/labeled-images/image-labels.csv')
findings = df_all['Finding'].values

findings_list = list(np.unique(findings))
findings_to_class = dict(zip(findings_list, np.arange(len(findings_list))))
class_to_findings = dict(zip(np.arange(len(findings_list)),findings_list))

images = df_all['Video file'].values
df_all['category'] = df_all.Finding.replace(findings_to_class)
df_all.drop(['Organ', 'Classification'], axis=1, inplace=True)
df_all.columns = ['image_id', 'finding_name', 'finding']

num_ims = len(df_all)
meh, df_val1 = train_test_split(df_all, test_size=num_ims//5, random_state=0, stratify=df_all.finding)
meh, df_val2 = train_test_split(meh,    test_size=num_ims//5, random_state=0, stratify=meh.finding)
meh, df_val3 = train_test_split(meh,    test_size=num_ims//5, random_state=0, stratify=meh.finding)
df_val5, df_val4 = train_test_split(meh,test_size=num_ims//5, random_state=0, stratify=meh.finding)

df_train1 = pd.concat([df_val2,df_val3,df_val4,df_val5], axis=0)
df_train2 = pd.concat([df_val1,df_val3,df_val4,df_val5], axis=0)
df_train3 = pd.concat([df_val1,df_val2,df_val4,df_val5], axis=0)
df_train4 = pd.concat([df_val1,df_val2,df_val3,df_val5], axis=0)
df_train5 = pd.concat([df_val1,df_val2,df_val3,df_val4], axis=0)

df_train1.to_csv('data/train_endo1.csv', index=None)
df_val1.to_csv('data/val_endo1.csv', index=None)

df_train2.to_csv('data/train_endo2.csv', index=None)
df_val2.to_csv('data/val_endo2.csv', index=None)

df_train3.to_csv('data/train_endo3.csv', index=None)
df_val3.to_csv('data/val_endo3.csv', index=None)

df_train4.to_csv('data/train_endo4.csv', index=None)
df_val4.to_csv('data/val_endo4.csv', index=None)

df_train5.to_csv('data/train_endo5.csv', index=None)
df_val5.to_csv('data/val_endo5.csv', index=None)

