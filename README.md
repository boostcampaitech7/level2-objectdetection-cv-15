## Repository Clone 및 데이터 다운로드

다음 명령어를 사용하여 레포지토리를 클론하고 데이터 파일을 다운로드한 후, 압축을 해제할 수 있습니다.

```bash
# 1. 레포지토리 클론
git clone https://github.com/boostcampaitech7/level2-objectdetection-cv-15.git

# 2. 클론한 디렉토리로 이동
cd level2-objectdetection-cv-15

# 3. 데이터 다운로드
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000325/data/data.tar.gz

# 4. 다운로드한 데이터 파일 압축 해제
tar -xzvf data.tar.gz

# 5. 압축 파일 삭제
rm data.tar.gz

# 6. 필요 패키지 다운로드
cd baseline
source activate base
pip install -r requirements.txt