//venv 설치 및 실행(가상환경 설정)

python -m venv venv
.\venv\Scripts\Activate.ps1

에러시
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1

venv\Scripts\activate
pip install Flask
pip install flask-cors

// 서버 실행

cd server
python server.py

---

// 클라이언트 실행

cd navi-client
npm i
npm start

---

git push 과정

각자 branch에서 push
merge는 한명이 진행

클라이언트 수정 시
클라이언트에서

git add .
git commit -m ""
git push

진행 후
navi에서

git add .
git commit -m ""
git push

진행
