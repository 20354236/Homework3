import torch.nn as nn

class P_Net(nn.Module):
    ''' PNet '''
    def __init__(self):
        super(P_Net,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(10)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=16,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(16)
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv4_1=nn.Conv2d(32,1,kernel_size=1,stride=1)
        self.conv4_2=nn.Conv2d(32,4,kernel_size=1,stride=1)
        self.conv4_3=nn.Conv2d(32,10,kernel_size=1,stride=1)
        self.Sigmoid=nn.Sigmoid()
        

    def forward(self,x):
        x=self.layer1(x)      # 12*12*3 -> 5*5*10
        x=self.layer2(x)      # 5*5*10 ->3*3*16
        x=self.layer3(x)      # 3*3*16 ->1*1*32
        label=self.Sigmoid(self.conv4_1(x))
        bounding_box=self.conv4_2(x)
        landmark=self.conv4_3(x)
        return label,bounding_box

class R_Net(nn.Module):
    ''' RNet '''
    def __init__(self):
        super(R_Net,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=28,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(28)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=28,out_channels=48,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(48)
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=48,out_channels=64,kernel_size=2,stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(64)
        )
        self.fc1=nn.Linear(3*3*64,128)
        self.fc2=nn.Linear(128,1)
        self.fc3=nn.Linear(128,4)
        self.fc4=nn.Linear(128,10)
        self.prelu=nn.PReLU()
        self.Sigmoid=nn.Sigmoid()
    def forward(self,x):
        x=self.layer1(x)       #24*24*3 -> 11*11*28
        x=self.layer2(x)       #11*11*28 -> 4*4*48
        x=self.layer3(x)       #4*4*48 ->3*3*64
        x=x.view(x.shape[0],-1)
        x=self.fc1(x)         # 3*3*64 -> 128
        x=self.prelu(x)
        label=self.Sigmoid(self.fc2(x)) #128 ->1
        bounding_box=self.fc3(x)      #128->4
        landmark=self.fc4(x)    #128->10
        return label,bounding_box

class O_Net(nn.Module):
    def __init__(self):
        super(O_Net, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(32)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.BatchNorm2d(64)
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64)
        )
        self.layer4=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=2,stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(128)
        )
        self.fc1=nn.Linear(3*3*128,256)
        self.fc2=nn.Linear(256,1)
        self.fc3=nn.Linear(256,4)
        self.fc4=nn.Linear(256,10)
        self.prelu=nn.PReLU()
        self.Sigmoid=nn.Sigmoid()
    def forward(self,x):
        x=self.layer1(x)       #48*48*3 -> 23*23*32
        x=self.layer2(x)       #23*23*32 -> 10*10*64
        x=self.layer3(x)       #10*10*64 ->4*4*64
        x=self.layer4(x)       #4*4*64 -> 3*3*128
        x=x.view(x.shape[0],-1)
        x=self.fc1(x)         # 3*3*128 -> 256
        x=self.prelu(x)
        label=self.Sigmoid(self.fc2(x)) #256 ->1
        bounding_box=self.fc3(x)      #256->4
        landmark=self.fc4(x)    #256->10
        return label,bounding_box,landmark


















