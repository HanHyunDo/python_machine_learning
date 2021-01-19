```python
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import mglearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
```


```python
from matplotlib import font_manager
import matplotlib
font_fname = 'C:/Windows/Fonts/malgun.ttf'
font_family = font_manager.FontProperties(fname=font_fname).get_name()
matplotlib.rcParams['axes.unicode_minus'] = False
print(font_family)
```

    Malgun Gothic
    


```python
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_fname = 'C:/Windows/Fonts/malgun.ttf'

font_family = font_manager.FontProperties(fname=font_fname).get_name()

plt.rcParams["font.family"] = font_family
```

# 3.1 비지도 학습의 종류



#### 1. 비지도 변환 - 데이터를 새롭게 표현하여 사람이나 다른 머신러닝 알고리즘이 원래 데이터보다 쉽게 해석할 수 있도록 만드는 알고리즘


#### 2. 군집 알고리즘 - 데이터를 비슷한 것끼리 그룹으로 묶는 것



# 3.2 비지도 학습의 도전 과제

비지도 학습은 뭔가 유용한 것을 학습했는지 평가하는 것이 제일 어렵습니다. 그래서 결과를 평가하기 위해서는 직접 확인하는 것이 유일한 방법일 때가 많습니다.

비지도 학습 알고리즘은 데이터 과학자가 데이터를 더 잘 이해하고 싶을 때 탐색적 분석 단계에서 많이 사용합니다. 그리고 지도 학습의 전처리 단계에서도 사용합니다.


# 3.2 데이터 전처리와 스케일 조정



```python
mglearn.plots.plot_scaling()
```


    
![png](output_4_0.png)
    


## 3.3.1 여러 가지 전처리 방법

위의 첫번째 그래프는 두 개의 특성을 인위적으로 만든 이진 분류 데이터셋입니다.

StandardScaler - 각 특성의 평균을 0, 분산을 1로 변경하여 모든 특성이 같은 크기를 가지게 됩니다.

RobustScaler - 특성들이 같은 스케일을 갖게 되지만 평균과 분산 대신 중간 값과 사분위 값을 사용

MinMaxScaler - 모든 특성이 정확하게 0과 1 사이에 위치하도록 데이터를 변경

Normalizer - 특성 벡터의 유클리디안 길이가 1이 되도록 데이터 포인트를 조정


## 3.3.2 데이터 변환 적용하기

MinMaxSacaler 사용하여 cancer 데이터 처리하기


```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=1)

print(X_train.shape)
print(X_test.shape)
```

    (426, 30)
    (143, 30)
    


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
```




    MinMaxScaler()




```python
# 데이터 변환
X_train_scaled = scaler.transform(X_train)
# 스케일이 조정된 후 데이터셋의 속성을 출력합니다.
print("변환된 후 크기:", X_train_scaled.shape)
print("스케일 조정 전 특성별 최소값:\n", X_train.min(axis=0))
print("스케일 조정 전 특성별 최대값:\n", X_train.max(axis=0))
print("스케일 조정 후 특성별 최소값:\n", X_train_scaled.min(axis=0))
print("스케일 조정 후 특성별 최대값:\n", X_train_scaled.max(axis=0))
```

    변환된 후 크기: (426, 30)
    스케일 조정 전 특성별 최소값:
     [6.981e+00 9.710e+00 4.379e+01 1.435e+02 5.263e-02 1.938e-02 0.000e+00
     0.000e+00 1.060e-01 5.024e-02 1.153e-01 3.602e-01 7.570e-01 6.802e+00
     1.713e-03 2.252e-03 0.000e+00 0.000e+00 9.539e-03 8.948e-04 7.930e+00
     1.202e+01 5.041e+01 1.852e+02 7.117e-02 2.729e-02 0.000e+00 0.000e+00
     1.566e-01 5.521e-02]
    스케일 조정 전 특성별 최대값:
     [2.811e+01 3.928e+01 1.885e+02 2.501e+03 1.634e-01 2.867e-01 4.268e-01
     2.012e-01 3.040e-01 9.575e-02 2.873e+00 4.885e+00 2.198e+01 5.422e+02
     3.113e-02 1.354e-01 3.960e-01 5.279e-02 6.146e-02 2.984e-02 3.604e+01
     4.954e+01 2.512e+02 4.254e+03 2.226e-01 9.379e-01 1.170e+00 2.910e-01
     5.774e-01 1.486e-01]
    스케일 조정 후 특성별 최소값:
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0.]
    스케일 조정 후 특성별 최대값:
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1.]
    


```python
# 테스트 데이터 변환
X_test_scaled = scaler.transform(X_test)
# 스케일이 조정된 후 테스트 데이터셋의 속성을 출력합니다.
print("스케일 조정 후 특성별 최소값:\n", X_test_scaled.min(axis=0))
print("스케일 조정 후 특성별 최대값:\n", X_test_scaled.max(axis=0))
```

    스케일 조정 후 특성별 최소값:
     [ 0.0336031   0.0226581   0.03144219  0.01141039  0.14128374  0.04406704
      0.          0.          0.1540404  -0.00615249 -0.00137796  0.00594501
      0.00430665  0.00079567  0.03919502  0.0112206   0.          0.
     -0.03191387  0.00664013  0.02660975  0.05810235  0.02031974  0.00943767
      0.1094235   0.02637792  0.          0.         -0.00023764 -0.00182032]
    스케일 조정 후 특성별 최대값:
     [0.9578778  0.81501522 0.95577362 0.89353128 0.81132075 1.21958701
     0.87956888 0.9333996  0.93232323 1.0371347  0.42669616 0.49765736
     0.44117231 0.28371044 0.48703131 0.73863671 0.76717172 0.62928585
     1.33685792 0.39057253 0.89612238 0.79317697 0.84859804 0.74488793
     0.9154725  1.13188961 1.07008547 0.92371134 1.20532319 1.63068851]
    

## 3.3.3 QuantileTransformer와 PowerTransformer

QuantileTransformer는 기본적으로 1000개의 분위를 사용하여 데이터를 균등하게 분포시킵니다.


```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import QuantileTransformer, StandardScaler, PowerTransformer

X, y = make_blobs(n_samples=50, centers=2, random_state=4, cluster_std=1)
X += 3

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors='black')
plt.xlim(0, 16)
plt.xlabel('x0')
plt.ylim(0, 10)
plt.ylabel('x1')
plt.title("Original Data")
plt.show()
```


    
![png](output_11_0.png)
    



```python
<div class = "girk" >
# QuantileTransformer를 적용해보자

scaler = QuantileTransformer()
X_trans = scaler.fit_transform(X)

plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y, s=30, edgecolors='black')
plt.xlim(0, 5)
plt.xlabel('x0')
plt.ylim(0, 5)
plt.ylabel('x1')
plt.title(type(scaler).__name__)
plt.show()
```


    
![png](output_12_0.png)
    



```python
# 히스토그램으로 균등 분포 확인

plt.hist(X_trans)
plt.show()
```


    
![png](output_13_0.png)
    



```python
print(scaler.quantiles_.shape)
```

    (50, 2)
    


```python
x = np.array([[0], [5], [8], [9], [10]])
print(np.percentile(x[:, 0], [0, 25, 50, 75, 100]))
```

    [ 0.  5.  8.  9. 10.]
    


```python
x_trans = QuantileTransformer().fit_transform(x)
print(np.percentile(x_trans[:, 0], [0, 25, 50, 75, 100]))
```

    [0.   0.25 0.5  0.75 1.  ]
    


```python
scaler = QuantileTransformer(output_distribution='normal')
X_trans = scaler.fit_transform(X)

plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y, s=30, edgecolors='black')
plt.xlim(-5, 5)
plt.xlabel('x0')
plt.ylim(-5, 5)
plt.ylabel('x1')
plt.title(type(scaler).__name__)
plt.show()
```


    
![png](output_17_0.png)
    



```python
plt.hist(X)
plt.title('Original Data')
plt.show()

X_trans = QuantileTransformer(output_distribution='normal').fit_transform(X)
plt.hist(X_trans)
plt.title('QuantileTransformer')
plt.show()

X_trans = StandardScaler().fit_transform(X)
plt.hist(X_trans)
plt.title('StandardScaler')
plt.show()

X_trans = PowerTransformer(method='box-cox').fit_transform(X)
plt.hist(X_trans)
plt.title('PowerTransformer box-cox')
plt.show()

X_trans = PowerTransformer(method='yeo-johnson').fit_transform(X)
plt.hist(X_trans)
plt.title('PowerTransformer yeo-johnson')
plt.show()
```


    
![png](output_18_0.png)
    



    
![png](output_18_1.png)
    



    
![png](output_18_2.png)
    



    
![png](output_18_3.png)
    



    
![png](output_18_4.png)
    


## 3.3.4 훈련 데이터와 테스트 데이터의 스케일을 같은 방법으로 조정하기

지도 학습 모델에서 테스트 세트를 사용하려면 훈련 세트와 테스트 세트에 같은 변환을 적용해야 합니다. 이와 반대로 테스트 세트의 최솟값과 범위를 사용했을 때 어떤 일이 일어날까?


```python
from sklearn.datasets import make_blobs
# 인위적인 데이터셋 생성
X, _, make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# 훈련 세트와 테스트 세트로 나눕니다.
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

# 훈련 세트와 테스트 세트의 산점도를 그립니다.
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=mglearn.cm2.colors[0],
                label="훈련 세트", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], c=mglearn.cm2.colors[1],
                label="테스트 세트", s=60, marker='^')
axes[0].legend(loc='upper left')
axes[0].set_title("원본 데이터")

# MinMaxScaler를 사용해 스케일 조정
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 스케일이 조정된 데이터의 산점도를 그립니다.
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2.colors[0],
                label="훈련 세트", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=mglearn.cm2.colors[1],
                label="테스트 세트", s=60, marker='^')
axes[1].set_title("스케일이 조정된 데이터")

# 테스트 세트의 스케일을 따로 조정합니다
# 테스트 세트의 최솟값은 0, 최댓값은 1이 됩니다.
# 이는 예제를 위한 것으로, 절대로 이렇게 사용해서는 안됩니다.
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaler_badly = test_scaler.transform(X_test)

# 잘못 조정된 데이터의 산점도를 그립니다.
axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2.colors[0],
                label="훈련 세트", s=60)
axes[2].scatter(X_test_scaler_badly[:, 0], X_test_scaler_badly[:, 1], c=mglearn.cm2.colors[1],
                label="테스트 세트", s=60, marker='^')
axes[2].set_title("잘못 조정된 데이터")

for ax in axes:
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
```


    
![png](output_20_0.png)
    


## 3.3.5 지도 학습에서 데이터 전처리 효과



```python
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

svm = SVC(C=100)
svm.fit(X_train, y_train)
print("테스트 세트 정확도: {:.2f}".format(svm.score(X_test, y_test)))
```

    테스트 세트 정확도: 0.94
    


```python
<div class="girk">
# 학습 시키기 전에 MinMaxScaler을 사용해 데이터의 스케일 조정

# 0~1 사이로 스케일 조정
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 조정된 데이터로 SVM 학습
svm.fit(X_train_scaled, y_train)

# 스케일 조정된 테스트 세트의 정확도
print("스케일 조정된 테스트 세트 정확도: {:.2f}".format(svm.score(X_test_scaled, y_test)))</div><i class="fa fa-lightbulb-o "></i>
```

    스케일 조정된 테스트 세트 정확도: 0.97
    


```python
# 평균 0, 분산 1을 갖도록 스케일 조정
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 조정된 데이터로 SVM 학습
svm.fit(X_train_scaled, y_train)

# 스케일 조정된 테스트 세트의 정확도
print("SVM test accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))
```

    SVM test accuracy: 0.96
    
