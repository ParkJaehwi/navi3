# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import openai
# import os
# from dotenv import load_dotenv
# import requests

# # .env 파일 로드
# load_dotenv()

# # OpenAI API 키 설정
# openai.api_key = os.getenv('OPENAI_API_KEY')

# app = Flask(__name__)
# CORS(app)

# @app.route('/api/data', methods=['GET'])
# def get_data():
#     # GPT-3.5 모델에 대한 요청 생성
#     response = openai.Completion.create(
#       model="gpt-3.5-turbo",  # 사용할 모델 지정
#       messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Hello, world!"}
#       ],
#       max_tokens=50  # 생성할 텍스트의 최대 길이
#     )

#     # GPT 응답에서 텍스트 추출
#     gpt_response = response.choices[0].message['content'].strip()

#     data = {"message": gpt_response}
#     return jsonify(data)

# if __name__ == '__main__':
#     app.run(debug=True)
# app.py

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import openai
# import os
# from dotenv import load_dotenv
# import requests

# app = Flask(__name__)
# CORS(app)  # CORS 설정

# load_dotenv()  # .env 파일에서 환경 변수 로드

# # OpenAI API 키 설정
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# # GPT-3.5 API 호출 함수
# def get_gpt3_response(prompt):
#     headers = {
#         'Authorization': f'Bearer {OPENAI_API_KEY}',
#         'Content-Type': 'application/json',
#     }
#     data = {
#         'model': 'gpt-3.5-turbo',
#         'messages': [{'role': 'user', 'content': prompt}],
#     }
#     response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
#     return response.json()

# # 기본 라우트 설정
# @app.route('/')
# def home():
#     return "Hello, this is a GPT-3.5 API integration with Flask!"

# # GPT-3.5 API를 호출하는 엔드포인트 설정
# @app.route('/ask', methods=['POST'])
# def ask_gpt3():
#     data = request.get_json()
#     prompt = data.get('prompt', '')
#     if prompt:
#         gpt3_response = get_gpt3_response(prompt)
#         return jsonify(gpt3_response)
#     return jsonify({'error': 'No prompt provided'}), 400

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import openai
# import os
# from dotenv import load_dotenv
# import requests

# app = Flask(__name__)
# CORS(app)

# load_dotenv()
# GPT_API_KEY=os.getenv('GPT_API_KEY')

# client = openai.OpenAI(api_key = GPT_API_KEY)

# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "지금 응답 테스트 중이야. content내용 '응답완료'로 대답해"},
#     {"role": "user", "content": "응답완료"}
#   ]
# )

# print(completion.choices[0].message.content)


# if __name__ == '__main__':
#     app.run(debug=True)
    
    
# @app.route('/ask', methods=['GET'])
# def get_data():
#   completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#       {"role": "system", "content": "지금 응답 테스트 중이야. content내용 '응답완료'로 대답해"},
#       {"role": "user", "content": "응답완료"}
#     ]
#   )
#   data = completion.choices[0].message.content
#   return jsonify({'data': data})
  


from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
load_dotenv()
GPT_API_KEY = os.getenv('GPT_API_KEY')
client = openai.OpenAI(api_key=GPT_API_KEY)

def gpt_api():
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "너는 여행 동선을 짜주는 ai비서야"},
        {"role": "user", "content": "아쿠아플라넷 제주: 제주 서귀포시 성산읍 섭지코지로 95 아쿠아플라넷 제주 산방산 탄산온천: 제주 서귀포시 안덕면 사계북로41번길 192 제주항공우주박물관: "
  + "제주 서귀포시 안덕면 녹차분재로 218 제주항공우주박물관 대포주상절리: 제주 서귀포시 이어도로 36-24 더본 호텔 제주 :제주 서귀포시 색달로 18"+"내가 준 데이터 중에서 3개만 뽑아서 제주도 1박 2일 일정 짜주고 리턴 데이터는 json으로"}
    ],
    # max_tokens=100
  )
  data = completion.choices[0].message.content
  print(completion.choices[0].message.content)
  return jsonify({'data': data})

@app.route('/ask', methods=['GET'])
def ask_gpt():
    return gpt_api()

if __name__ == '__main__':
    app.run(debug=True)
