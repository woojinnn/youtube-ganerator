# YTGAN: YouTube thumbnail GANerator
---
## Introduction  
### KAIST CS376 Machine Learning  
### Final Project - Youtube GANerator
#### by Woojin Lee, Dongwon Choi, Jeongha Seo, Sihyun Ahn
---
|Final Report | https://docs.google.com/document/d/1qYt5e0qHzGD1t92SkOKZ5pQtOtsITWwEfoodB8Gf288/edit?usp=sharing|
|:---:|:---|
|**PPT**         | https://docs.google.com/presentation/d/1pspdktUKwSx3tf10Y9hvMtDlkbq1szGpZ3Xk47uVWtQ/edit?usp=sharing|
---
### Generated Images By Final Model During Training
|Autos&Vehicles|News&Politics|
|------|---|
|![](/readme_img/A%26V4_animation.gif)|![](/readme_img/animation.gif)|
---

## Adjust Model

#### 1. Vanilla DCGAN (News&Politics) 

![](/readme_img/NnP/vanilla/fake_real.png)

|epoch 5|epoch 100|
|:-------:|:---------:|
|![](/readme_img/NnP/vanilla/yt_5.png)|![](/readme_img/NnP/vanilla/yt_100.png)|
|**epoch 200** |**epoch 300**|
|![](/readme_img/NnP/vanilla/yt_200.png)|![](/readme_img/NnP/vanilla/yt_300.png)|
|**epoch 400**|**epoch 495**|
|![](/readme_img/NnP/vanilla/yt_400.png)|![](/readme_img/NnP/vanilla/yt_495.png)|
---
#### 2. Noise to Discriminator Input 0.05 (News&Politics)

![](/readme_img/NnP/noise/0.05/fake_real.png)

|epoch 5|epoch 100|
|:-------:|:---------:|
|![](/readme_img/NnP/noise/0.05/yt_5.png)|![](/readme_img/NnP/noise/0.05/yt_100.png)|
|**epoch 200**|**epoch 300**|
|![](/readme_img/NnP/noise/0.05/yt_200.png)|![](/readme_img/NnP/noise/0.05/yt_300.png)|
|**epoch 400**|**epoch 495**|
|![](/readme_img/NnP/noise/0.05/yt_400.png)|![](/readme_img/NnP/noise/0.05/yt_495.png)|
---
#### 3. Noise 0.1 (News&Politics)

![](/readme_img/NnP/noise/0.1/fake_real.png)

|**epoch 5**|**epoch 100**|
|:-------:|:---------:|
|![](/readme_img/NnP/noise/0.1/yt_5.png)|![](/readme_img/NnP/noise/0.1/yt_100.png)|
|**epoch 200**|**epoch 300**|
|![](/readme_img/NnP/noise/0.1/yt_200.png)|![](/readme_img/NnP/noise/0.1/yt_300.png)|
**epoch 400**|**epoch 495**|
|![](/readme_img/NnP/noise/0.1/yt_400.png)|![](/readme_img/NnP/noise/0.1/yt_495.png)|
---
#### 4. Noise 0.15 (News&Politics)

![](/readme_img/NnP/noise/0.15/fake_real.png)

|**epoch 5**|**epoch 100**|
|:-------:|:---------:|
|![](/readme_img/NnP/noise/0.15/yt_5.png)|![](/readme_img/NnP/noise/0.15/yt_100.png)|
|**epoch 200**|**epoch 300**|
|![](/readme_img/NnP/noise/0.15/yt_200.png)|![](/readme_img/NnP/noise/0.15/yt_300.png)|
**epoch 400**|**epoch 495**|
|![](/readme_img/NnP/noise/0.15/yt_400.png)|![](/readme_img/NnP/noise/0.15/yt_495.png)|
---
#### 5. Noisy Label (News&Politics)

![](/readme_img/NnP/lnoise/fake_real.png)

|**epoch 5**|**epoch 100**|
|:-------:|:---------:|
|![](/readme_img/NnP/lnoise/yt_5.png)|![](/readme_img/NnP/lnoise/yt_100.png)|
|**epoch 200**|**epoch 300**|
|![](/readme_img/NnP/lnoise/yt_200.png)|![](/readme_img/NnP/lnoise/yt_300.png)|
**epoch 400**|**epoch 495**|
|![](/readme_img/NnP/lnoise/yt_400.png)|![](/readme_img/NnP/lnoise/yt_495.png)|
---
#### 6. Dropout (News&Politics)

![](/readme_img/NnP/dropout/fake_real.png)

|**epoch 5|****epoch 100**|
|:-------:|:---------:|
|![](/readme_img/NnP/dropout/yt_5.png)|![](/readme_img/NnP/dropout/yt_100.png)|
|**epoch 200**|**epoch 300**|
|![](/readme_img/NnP/dropout/yt_200.png)|![](/readme_img/NnP/dropout/yt_300.png)|
**epoch 400**|**epoch 495**|
|![](/readme_img/NnP/dropout/yt_400.png)|![](/readme_img/NnP/dropout/yt_495.png)|

---
### Loss Graph (News&Politics)

|Vanilla DCGAN|Noise 0.05|
|:---:|:---:|
|![](/readme_img/NnP/vanilla/G%26D_Loss.png)|![](/readme_img/NnP/noise/0.05/G%26D_Loss.png)
|**Noise 0.1**|**Noise 0.15**|
|![](/readme_img/NnP/noise/0.1/G%26D_Loss.png)|![](/readme_img/NnP/noise/0.15/G%26D_Loss.png)
|**Noisy Label**|**Dropout**|
|![](/readme_img/NnP/lnoise/G%26D_Loss.png)|![](/readme_img/NnP/dropout/G%26D_Loss.png)

---
## Adjust Learning Rate

### Dropout (News&Politics)
#### 1. Learning Rate = 0.001 
![](/readme_img/A%26VD_001_animation.gif)
![](/readme_img/A%26VD_001_fake_real.png)
#### 2. learning rate = 0.0005 
![](/readme_img/A%26VD_0005_animation.gif)
![](/readme_img/A%26VD_0005_fake_real.png)
#### 3. learning rate = 0.0002 
![](/readme_img/A%26VD_animation.gif)
![](/readme_img/A%26VD_fake_real.png)
#### 4. learning rate = 0.0001 
![](/readme_img/A%26VD_0001_animation.gif)
![](/readme_img/A%26VD_0001_fake_real.png)
### Loss & Score
|Learning Rate = 0.001|Learning Rate = 0.0005|
|:---:|:---:|
|![](/readme_img/A%26VD_001_G%26D_Loss.png)|![](/readme_img/A%26VD_0005_G%26D_Loss.png)|
|![](/readme_img/A%26VD_001_G%26D_score.png)|![](/readme_img/A%26VD_0005_G%26D_score.png)|

|Learning Rate = 0.0002|Learning Rate = 0.0001|
|:---:|:---:|
|![](/readme_img/A%26VD_G%26D_Loss.png)|![](/readme_img/A%26VD_0001_G%26D_Loss.png)|
|![](/readme_img/A%26VD_G%26D_score.png)|![](/readme_img/A%26VD_0001_G%26D_score.png)|
---
### Dropout & Noisy Label (News&Politics)
#### 1. Learning Rate = 0.001 
![](/readme_img/A%26V4_001_animation.gif)
![](/readme_img/A%26V4_001_fake_real.png)
#### 2. learning rate = 0.0005 
![](/readme_img/A%26V4_0005_animation.gif)
![](/readme_img/A%26V4_0005_fake_real.png)
#### 3. learning rate = 0.0002 
![](/readme_img/A%26V4_animation.gif)
![](/readme_img/A%26V4_fake_real.png)
#### 4. learning rate = 0.0001 
![](/readme_img/A%26V4_0001_animation.gif)
![](/readme_img/A%26V4_0001_fake_real.png)
### Loss & Score
|Learning Rate = 0.001|Learning Rate = 0.0005|
|:---:|:---:|
|![](/readme_img/A%26V4_001_G%26D_Loss.png)|![](/readme_img/A%26V4_0005_G%26D_Loss.png)|
|![](/readme_img/A%26V4_001_G%26D_score.png)|![](/readme_img/A%26V4_0005_G%26D_score.png)|

|Learning Rate = 0.0002|Learning Rate = 0.0001|
|:---:|:---:|
|![](/readme_img/A%26V4_G%26D_Loss.png)|![](/readme_img/A%26V4_0001_G%26D_Loss.png)|
|![](/readme_img/A%26V4_G%26D_score.png)|![](/readme_img/A%26V4_0001_G%26D_score.png)|
---
# Results
#### Autos&Vehicles
![](/readme_img/A%26V_fake_real3.png) 
#### Film&Animation
![](/readme_img/F%26A_fake_real3.png) 
#### News&Politics
![](/readme_img/N%26P_fake_real3.png) 
