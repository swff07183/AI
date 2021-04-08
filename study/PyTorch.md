# PyTorch

[PyTorch Tutorials](https://pytorch.org/tutorials/)

## 신경망(Neural Networks)



![convnet](https://tutorials.pytorch.kr/_images/mnist.png)

숫자 이미지를 분류하는 신경망.



#### 신경망의 학습과정

- 학습 가능한 매개변수(또는 weight)를 갖는 신경망을 정의한다.
- 데이터셋(dataset) 입력을 반복한다.
- 입력을 신경망에서 전파(process)한다
- 손실(loss)을 계산한다.
- 변화도(gradient)를 신경망의 매개변수들에 역으로 전파한다.
- 신경망의 가중치를 갱신한다(`새로운 가중치(weight)` = `가중치(weight)` - `학습률(learning rate)` * `변화도(gradient)`)



#### 신경망 정의하기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 입력 이미지 채널 1개, 출력 채널 6개, 3x3의 정사각 컨볼루션 행렬
        # 컨볼루션 커널 정의
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 아핀(affine) 연산: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6은 이미지 차원에 해당
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # (2, 2) 크기 윈도우에 대해 맥스 풀링(max pooling)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 크기가 제곱수라면 하나의 숫자만을 특정
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 배치 차원을 제외한 모든 차원
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
```

`forward` 함수를 정의하면, `backward` 함수는 autograd를 사용하여 자동으로 정의된다.



##### *Conv2d 함수

```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
```

`in_channels` : input image의 channel 수

`out_channels` : convolution을 통하여 생성된 channel의 수

`kernel_size` : kernel(filter)의 사이즈. 정사각 행렬.



##### *Max Pooling

특징만 뽑아내서 input size를 줄이기 위하여 사용. Channel 수에는 영향 X

![img](https://blog.kakaocdn.net/dn/cyEnmu/btqCAQzvxul/yrK2N14qZc7OvZNFw2bc30/img.png)



##### 임의의 입력값 넣고 역전파 하기

```python
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
```

```python
net.zero_grad() # 변화도 버퍼(gradient buffer)를 0으로 설정
out.backward(torch.randn(1, 10))
```

신경망을 학습 시킬때 누적되는 변화도를 정확하게 추적하기 위하여  `zero_grad()` 함수를 사용하여 변화도를 0으로 설정해준다.



#### 손실함수(Loss Function)

손실함수는 (output, target)을 입력으로 받아, output이 target으로부터 얼마나 떨어져있는지를 추정하는 값을 계산한다.

`nn.MSEloss` : output과 target 간의 평균제곱오차(mean-squared error) 계산

```python
output = net(input)
target = torch.randn(10)  # 예시를 위한 임의의 정답
target = target.view(1, -1)  # 출력과 같은 shape로 만듦
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```



### 역전파(Backpropagation)

```
net.zero_grad()     # 모든 매개변수의 변화도 버퍼를 0으로 만듦

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```



### 가중치 갱신

확률적 경사하강법(Stochastic Gradient Descent) 사용.

`새로운 가중치(weight)` = `가중치(weight)` - `학습률(learning rate)` * `변화도(gradient)` 

```python
import torch.optim as optim

# Optimizer를 생성합니다.
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 학습 과정(training loop)에서는 다음과 같습니다:
optimizer.zero_grad()   # 변화도 버퍼를 0으로
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 업데이트 진행
```

