
### Generated Images By Final Model During Training
|Autos&Vehicles|News&Politics|
|------|---|
|![](/readme_img/A%26V4_animation.gif)|model4 n/p 0002|
---

## Adjust Model

#### 1. Vanilla DCGAN (News&Politics) 

![](/readme_img/NnP/vanilla/fake_real.png)

|epoch 5|epoch 100|
|-------|---------|
|![](/readme_img/NnP/vanilla/yt_5.png)|![](/readme_img/NnP/vanilla/yt_100.png)|

|epoch 200|epoch 300|
|---------|---------|
|![](/readme_img/NnP/vanilla/yt_200.png)|![](/readme_img/NnP/vanilla/yt_300.png)|

epoch 400|epoch 495|
|---------|---------|
|![](/readme_img/NnP/vanilla/yt_400.png)|![](/readme_img/NnP/vanilla/yt_495.png)|
---
#### 2. Noise 0.05 (News&Politics)

![](/readme_img/NnP/noise/0.05/fake_real.png)

|epoch 5|epoch 100|
|-------|---------|
|![](/readme_img/NnP/noise/0.05/yt_5.png)|![](/readme_img/NnP/noise/0.05/yt_100.png)|

|epoch 200|epoch 300|
|---------|---------|
|![](/readme_img/NnP/noise/0.05/yt_200.png)|![](/readme_img/NnP/noise/0.05/yt_300.png)|

epoch 400|epoch 495|
|---------|---------|
|![](/readme_img/NnP/noise/0.05/yt_400.png)|![](/readme_img/NnP/noise/0.05/yt_495.png)|
---
#### 3. Noise 0.1 (News&Politics)

![](/readme_img/NnP/noise/0.1/fake_real.png)

|epoch 5|epoch 100|
|-------|---------|
|![](/readme_img/NnP/noise/0.1/yt_5.png)|![](/readme_img/NnP/noise/0.1/yt_100.png)|

|epoch 200|epoch 300|
|---------|---------|
|![](/readme_img/NnP/noise/0.1/yt_200.png)|![](/readme_img/NnP/noise/0.1/yt_300.png)|

epoch 400|epoch 495|
|---------|---------|
|![](/readme_img/NnP/noise/0.1/yt_400.png)|![](/readme_img/NnP/noise/0.1/yt_495.png)|
---
#### 4. Noise 0.15 (News&Politics)

![](/readme_img/NnP/noise/0.15/fake_real.png)

|epoch 5|epoch 100|
|-------|---------|
|![](/readme_img/NnP/noise/0.15/yt_5.png)|![](/readme_img/NnP/noise/0.15/yt_100.png)|

|epoch 200|epoch 300|
|---------|---------|
|![](/readme_img/NnP/noise/0.15/yt_200.png)|![](/readme_img/NnP/noise/0.15/yt_300.png)|

epoch 400|epoch 495|
|---------|---------|
|![](/readme_img/NnP/noise/0.15/yt_400.png)|![](/readme_img/NnP/noise/0.15/yt_495.png)|
---
#### 5. Noisy label (News&Politics)

![](/readme_img/NnP/lnoise/fake_real.png)

|epoch 5|epoch 100|
|-------|---------|
|![](/readme_img/NnP/lnoise/yt_5.png)|![](/readme_img/NnP/lnoise/yt_100.png)|

|epoch 200|epoch 300|
|---------|---------|
|![](/readme_img/NnP/lnoise/yt_200.png)|![](/readme_img/NnP/lnoise/yt_300.png)|

epoch 400|epoch 495|
|---------|---------|
|![](/readme_img/NnP/lnoise/yt_400.png)|![](/readme_img/NnP/lnoise/yt_495.png)|
---
#### 6. Dropout (News&Politics)

![](/readme_img/NnP/dropout/fake_real.png)

|epoch 5|epoch 100|
|-------|---------|
|![](/readme_img/NnP/dropout/yt_5.png)|![](/readme_img/NnP/dropout/yt_100.png)|

|epoch 200|epoch 300|
|---------|---------|
|![](/readme_img/NnP/dropout/yt_200.png)|![](/readme_img/NnP/dropout/yt_300.png)|

epoch 400|epoch 495|
|---------|---------|
|![](/readme_img/NnP/dropout/yt_400.png)|![](/readme_img/NnP/dropout/yt_495.png)|

---
### Loss Graph (News&Politics)

|Vanilla DCGAN|Noise 0.05|
|---|---|
|![](/readme_img/NnP/vanilla/G%26D_Loss.png)|![](/readme_img/NnP/noise/0.05/G%26D_Loss.png)

|Noise 0.1|Noise 0.15|
|---|---|
|![](/readme_img/NnP/noise/0.1/G%26D_Loss.png)|![](/readme_img/NnP/noise/0.15/G%26D_Loss.png)

|Noisy Label|Dropout|
|---|---|
|![](/readme_img/NnP/lnoise/G%26D_Loss.png)|![](/readme_img/NnP/dropout/G%26D_Loss.png)

---
## Adjust Learning Rate

### Dropout
#### Learning Rate = 0.001 
![](/readme_img/A%26VD_001_animation.gif)
![](/readme_img/A%26VD_001_fake_real.png)
#### learning rate = 0.0005 
![](/readme_img/A%26VD_0005_animation.gif)
![](/readme_img/A%26VD_0005_fake_real.png)
#### learning rate = 0.0002 
![](/readme_img/A%26VD_animation.gif)
![](/readme_img/A%26VD_fake_real.png)
#### learning rate = 0.0001 
![](/readme_img/A%26VD_0001_animation.gif)
![](/readme_img/A%26VD_0001_fake_real.png)
### Loss and Score
|Learning Rate = 0.001|Learning Rate = 0.0005|
|---|---|
|![](/readme_img/A%26VD_001_G%26D_Loss.png)|![](/readme_img/A%26VD_0005_G%26D_Loss.png)|
|![](/readme_img/A%26VD_001_G%26D_score.png)|![](/readme_img/A%26VD_0005_G%26D_score.png)|

|Learning Rate = 0.0002|Learning Rate = 0.0001|
|---|---|
|![](/readme_img/A%26VD_G%26D_Loss.png)|![](/readme_img/A%26VD_0001_G%26D_Loss.png)|
|![](/readme_img/A%26VD_G%26D_score.png)|![](/readme_img/A%26VD_0001_G%26D_score.png)|
