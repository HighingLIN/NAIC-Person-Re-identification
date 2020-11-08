#%%
import pandas as pd
import numpy as np
import os, sys, glob, shutil, json
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2

from PIL import Image
import numpy as np

from tqdm import tqdm, tqdm_notebook

get_ipython().run_line_magic('pylab', 'inline')

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset



#%%
label=pd.read_csv('./train/label.txt',names=['label'])

# %%
png_id=[]
png_label=[]
for i in range(len(label)):
    png,png_label_t=label.loc[i,'label'].split(":")
    png_label.append(int(png_label_t))
    png_id_t,pn=png.split(".")
    png_id.append(png_id_t)

#%%
label['png_id']=png_id
label['png_label']=png_label
#%%
label.sort_values(by="png_label",axis=0,ascending=True,inplace=True)
#%%
label.reset_index(drop=True,inplace=True)
#%%
label=label[:1000].copy()
#%%
label['png_index']=label['png_id'].astype('int')
#%%
index=list(label['png_index'])
#%%
label['png_label'].value_counts()
# %%
# 定义读取数据集
class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label 
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        lbl = np.int(self.img_label[index])
        return img, torch.from_numpy(np.array(lbl))

    def __len__(self):
        return len(self.img_path)

# # 定义分类模型
# 这里使用ResNet18的模型进行特征提取
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
                
        model_conv = models.resnet18(pretrained=False)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv
        
        self.fc1 = nn.Linear(512, 259)
    
    def forward(self, img):        
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        return c1

# %%
def train(train_loader, val_input, val_target, model, criterion, optimizer):
    # 切换模型为训练模式
    model.train()
    train_loss = []
    
    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = target.cuda().long()
            
        c0 = model(input)
        # print(c0)
        loss = criterion(c0, target)
        val_output = model(val_input)
        val_loss = criterion(val_output, val_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

 

        train_loss.append(loss.item())
    return np.mean(train_loss)

def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda().long()
            
            c0 = model(input)
            loss = criterion(c0, target)
            val_loss.append(loss.item())
    return np.mean(val_loss)

def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None
    
    # TTA 次数
    for _ in range(tta):
        test_pred = []
    
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()
                
                c0 = model(input)
                if use_cuda:
                    output = np.concatenate([
                        c0.data.cpu().numpy()], axis=1)
                else:
                    output = np.concatenate([
                        c0.data.numpy()], axis=1)
                
                test_pred.append(output)
        
        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
    
    return test_pred_tta



# %%
all_path = glob.glob('./train/images/*.png')
all_path.sort()
all_path=all_path[:72824]
all_path=np.array(all_path)
all_path=all_path[index]

#%%
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
skf = KFold(n_splits=5, random_state=1, shuffle=True)
for flod_idx, (train_idx, val_idx) in enumerate(skf.split(all_path, all_path)):
    break



#%%
train_path=all_path[train_idx]
train_label=list(label.loc[train_idx,'png_label'].values)
print(len(train_path), len(train_label))

train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                        transforms.Resize((256, 128)),
                        transforms.RandomCrop((60, 120)),
                        transforms.ColorJitter(0.3, 0.3, 0.2),
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    batch_size=40,
    shuffle=True,
    num_workers=0,
)


#%%
val_path=all_path[val_idx]
val_label=label.loc[val_idx,'png_label'].values
val_label=val_label.tolist()
print(len(val_path), len(val_label))

val_loader = torch.utils.data.DataLoader(
    SVHNDataset(val_path, val_label,
                transforms.Compose([
                    transforms.Resize((256, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])), 
    batch_size=40, 
    shuffle=False, 
    num_workers=0,
)




# %%
# 训练与验证
model = SVHN_Model1()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)
best_loss = 1000.0

use_cuda = True
if use_cuda:
    model = model.cuda()

for j,(val_input, val_target) in enumerate(val_loader):
        val_input = val_input.cuda()
        val_target = val_target.cuda().long()
        break

for epoch in range(100):
    print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
    
    train_loss = train(train_loader, val_input, val_target, model, criterion, optimizer)
    val_loss = validate(val_loader, model, criterion)
    val_predict_label = predict(val_loader, model, 1)
    val_predict_label = val_predict_label[:, :259].argmax(1)
    
    val_char_acc = np.mean(np.array(val_predict_label) == np.array(val_label))
    
    print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, val_loss))
    print('Val Acc', val_char_acc)
    # 记录下验证集精度
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), './model.pt')



# %% 
# # 预测并生成提交文件
test_id_path = glob.glob('./image_A/query/*.png')
test_id_path.sort()
test_id_label = [1] * len(test_id_path)
print(len(test_id_path), len(test_id_label))

test_id_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_id_path, test_id_label,
                transforms.Compose([
                    transforms.Resize((128, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])), 
    batch_size=128, 
    shuffle=False, 
    num_workers=0,
)


# %%
# 加载保存的最优模型
model = SVHN_Model1()
use_cuda = True
if use_cuda:
    model = model.cuda()
model.load_state_dict(torch.load('model.pt'))

test_predict_id = predict(test_id_loader, model, 1)
print(test_predict_id.shape)

test_predict_id = test_predict_id[:, :19658].argmax(1)


# %%
test_path = glob.glob('./image_A/gallery/*.png')
test_path.sort()
test_label = [1] * len(test_path)
print(len(test_path), len(test_label))

test_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path, test_label,
                transforms.Compose([
                    transforms.Resize((128, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])), 
    batch_size=128, 
    shuffle=False, 
    num_workers=0,
)


# %%
# 加载保存的最优模型
test_predict = predict(test_loader, model, 1)
print(test_predict.shape)
#%%
test_predict1=np.sort(test_predict,axis=1)[::-1]
#%%
test_predict1[:10]
#%%
test_predict[0]
#%%
tempres_label=[]
for i in range(test_predict.shape[1]):
    tempres=np.argsort(test_predict[:,i],axis=0)[::-1]
    tempres=tempres[:200]
    tempres_str=[]
    for i in tempres:
        i=str(i)
        i=i.rjust(8,'0')
        i+='.png'
        tempres_str.append(i)
    tempres_label.append(tempres_str)
#%%
id_list=[]
for i in test_id_path:
    id_list.append(i.split("\\")[1])
#%%
sub={}
for i in range(len(id_list)):
    sub[id_list[i]]=tempres_label[test_predict_id[i]]
#%%
with open('sub10.json', 'w') as f:
    json.dump(sub, f)