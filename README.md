# ♻ 재활용 쓰레기 이미지 데이터 객체 탐지 

<br/>

## 👨‍👩‍👧‍👦 팀 구성
<div align="center">
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/SeoJinHyoung">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003813%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>서진형</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/andantecode">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003899%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>함로운</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/sihari-1115">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00004046%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>이시하</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/IronNote">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00004085%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>김명철</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/ruka030809">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00004086%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>김형준</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/alexminyoungpark">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00004104%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>박민영</b></sub><br />
      </a>
    </td>
  </tr>
</table>
</div>
<br />

## 📃 프로젝트 개요
분리수거는 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 재활용 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다.

문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.
<br/>

## ✔ 평가 지표
`mAP50(Mean Average Precision)로 평가`

- Object Detection에서 사용하는 대표적인 성능 측정 방법

- Ground Truth 박스와 Prediction 박스간 IoU가 50이 넘는 예측에 대해 True라고 판단

<br/>

## 📅 프로젝트 일정
2024/09/30 ~ 2024/10/24
<br/>

## 💻 개발 환경
```bash
- Language : Python
- Environment
  - CPU : Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz
  - GPU : Tesla V100-SXM2 32GB × 4
- Framework : PyTorch, pl, MMDetection, MMYOLO
- Collaborative Tool : Git, Notion
```
<br/>

## 🔆 프로젝트 결과

<br/>

## 📁 프로젝트 구조
```
📦level2-objectdetection-cv-15
 ┣ 📂.github
 ┃ ┗ 📄.keep
 ┣ 📂eda
 ┃ ┣ 📄eda.ipynb
 ┣ 📂huggingface
 ┃ ┣ 📂configs
 ┃ ┣ 📂src
 ┃ ┣ 📂tests
 ┃ ┣ 📄train.py
 ┣ 📂mmdetection
 ┃ ┣ 📂cascade_rcnn
 ┃ ┣ 📂dino
 ┃ ┣ 📂retinanet
 ┣ 📂mmyolo
 ┣ 📂prototype
 ┃ ┣ 📂utils
 ┃ ┣ 📄augmentation_demo.py
 ┣ 📂utils
 ┃ ┣ 📂configs
 ┃ ┣ 📄crop_images.ipynb
 ┃ ┣ 📄ensemble_inference.py
 ┃ ┣ 📄pseudo_labeling.py
 ┣ 📄.gitignore
 ┣ 📄README.md
 ```
<br/>
## ⚙️ requirements
```
```
<br/>
 
#### 1) `eda` 
- 재활용 쓰레기 이미지 데이터셋 분석 노트북

#### 2) `huggingface` 
- huggingface에 업로드 되어있는 DETR 계열 모델을 활용한 학습 pl 코드

#### 3) `mmdetection`
- mmdetection의 3개의 모델 backbone 변경 및 학습 방법론 교차 실험
    
    - cascade_rcnn
    - dino
    - retinanet

#### 4) `mmyolo`
- mmyolo의 yolox_x 모델 backbone 변경 실험

#### 5) `prototype`
- albumentation 증강 기법 적용 확인을 위한 데모

#### 6) `utils`
- ensemble 추론 및 pseudo_labeling을 위한 코드
<br/>

## 📃 Wrap-Up 리포트

<br/>
<br/>
<br/>



    - 각 프레임워크의 실험 방법은 해당 폴더 README 참조