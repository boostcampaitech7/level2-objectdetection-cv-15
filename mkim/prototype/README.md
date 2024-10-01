## 🐣 데이터 증강 실험 데모 (feat. streamlit) 🐣

### 사용 방법
---

```bash
# streamlit 라이브러리 설치
pip install streamlit

# prototype 디렉토리 이동
cd prototype

# streamlit 실행
streamlit run augmentation_demo.py

# browser 열기
http://localhost:8501/
```

> 이후, 원하는 이미지 선택 (50개로 제한해둠)   
원하는 증강 기법 사이드 바에서 선택   
적용하기 버튼 클릭하면, 원본 이미지 옆에 증강된 이미지와 바뀐 bbox 좌표가 함께 plot됨

### 실험 가능한 증강 기법 [albumentation으로 구현]
---
1. Brightness (밝기 조정): limit값 설정
2. Contrast (대비 조정): limit값 설정
3. HorizontalFlip (수평 뒤집기): T/F
4. VerticalFlip (수직 뒤집기): T/F
5. Rotate (회전): 회전 각도 설정
6. RandomCrop (랜덤하게 해당 사이즈로 크롭): 크롭할 사이즈 (height, width)
7. Affine (affine 변환): angle, x, y 이동 좌표, scale (크기 조정), shear (기울임 정도)
8. Elastic Transform (비선형적 변환): alpha (강도), sigma (부드러움), alpha_affine (affine)
9. Gaussian Blur (가우시안 노이즈 추가): kernel size
10. Random Gamma (랜덤 감마값 변경): gamma 수치
11. Hue Saturation Value (Hue, saturation 조정): *TODO 한계값 다시 지정해야함* 
12. Random snow (랜덤으로 눈 온 환경 조성): 신기함
13. Channel Shuffle (채널 무작위 셔플)
14. Motion Blur (모션 블러)
15. Cutout (특정 부분 잘라내기)


### 결과

* 대부분의 증강에 대해 bbox도 함께 잘 변환되는 것을 확인할 수 있었음
> 특히, Random Crop, Affine 변환 등에도 굳건하게 잘 조정됨을 확인

* 굳이, Albumentation transform을 사용할 때, bbox 좌표를 커스텀해서 변경해줄 필요는 없을 것 같음
* Cutmix 구현 시에 추가 좌표 조정이 필요할 것 같음
* 합성 데이터를 만들어 증강하는 방식도 고려할 수 있을 것 bbox 좌표만 잘 레이블링한다면?




