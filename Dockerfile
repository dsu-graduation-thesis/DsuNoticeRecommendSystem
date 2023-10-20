# 베이스 이미지 설정
FROM python:3.11

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 설치
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# 소스 코드 복사
COPY . .

# 애플리케이션 실행
CMD ["python", "app.py"]
