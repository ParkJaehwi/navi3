# class TLSAdapter(HTTPAdapter):
#   def init_poolmanager(self, *args, **kwargs):
#     context = ssl.create_default_context()
#     context.set_ciphers('DEFAULT:@SECLEVEL=1')  # Adjust SSL security level if necessary
#     kwargs['ssl_context'] = context
#     return super(TLSAdapter, self).init_poolmanager(*args, **kwargs)

# session = requests.Session()
# session.mount('https://', TLSAdapter())

# url = "https://apis.data.go.kr/B551011/KorService1/searchStay1?serviceKey=DsUAduFiNXosF4vwy50AMcCRk9sYGUOEuVAXTab4UL%2BxhoqapEvU33LLFozTQYcRSVmIkGqX4Xke5XheHQIAwg%3D%3D&MobileOS=ETC&MobileApp=TestApp&_type=json&numOfRows=3747"

# response = session.get(url)

# print(f"Status Code: {response.status_code}")
# if response.status_code == 200:
#     data = response.json()
#     item = data.get('response', {}).get('body', {}).get('items', {}).get('item', {})
#     csv_file_path = os.path.join(os.getcwd(), 'output2.csv')
#     with open(csv_file_path, mode='w', newline='', encoding='utf-8-sig') as csv_file:
#         fieldnames = item[0].keys()  # CSV 파일의 헤더는 item의 키 값들로 설정
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(item)
# else:
#     print('error')


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import openai
# import os
# from dotenv import load_dotenv
# import requests
# import csv
# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.poolmanager import PoolManager
# import ssl

# app = Flask(__name__)
# CORS(app)
# load_dotenv()
# GPT_API_KEY = os.getenv('GPT_API_KEY')
# client = openai.OpenAI(api_key=GPT_API_KEY)

# def gpt_api():
#   completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "너는 여행 동선을 짜주는 ai비서야"},
#         {"role": "user", "content": "아쿠아플라넷 제주: 제주 서귀포시 성산읍 섭지코지로 95 아쿠아플라넷 제주 산방산 탄산온천: 제주 서귀포시 안덕면 사계북로41번길 192 제주항공우주박물관: "
#   + "제주 서귀포시 안덕면 녹차분재로 218 제주항공우주박물관 대포주상절리: 제주 서귀포시 이어도로 36-24 더본 호텔 제주 :제주 서귀포시 색달로 18"+"내가 준 데이터 중에서 3개만 뽑아서 제주도 1박 2일 일정 짜주고 리턴 데이터는 json으로"}
#     ],
#     # max_tokens=100
#   )
#   data = completion.choices[0].message.content
#   print(completion.choices[0].message.content)
#   return jsonify({'data': data})

# @app.route('/ask', methods=['GET'])
# def ask_gpt():
#     return gpt_api()

# import pandas as pd
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain.schema import Document
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import openai
# import os
# from dotenv import load_dotenv
# from openai import OpenAI
# from tqdm import tqdm

# app = Flask(__name__)
# CORS(app)
# load_dotenv()

# print('시작')

# # 1. CSV 파일 읽기 및 데이터 전처리
# file_path = './spot_data2.csv'  # 파일 경로를 지정하세요
# df = pd.read_csv(file_path)

# # 데이터 전처리: 필요한 열 선택 및 결합
# df['full_address'] = df['addr1'].fillna('') + ' ' + df['addr2'].fillna('')
# df['text'] = df['title'].fillna('') + ' - ' + df['full_address'].fillna('') + ' (' + df['tel'].fillna('').astype(str) + ')'
# df['doc_text'] = 'title: ' + df['title'] + ' addr1: ' + df['addr1'] + ' cat1: ' + df['cat1'] + ' ' + df['text']  # cat1 정보를 포함한 문서 텍스트 생성
# print('전처리 완료')

# # 중복 및 결측값 제거
# df.dropna(subset=['doc_text'], inplace=True)
# df.drop_duplicates(subset=['doc_text'], inplace=True)
# documents = df['doc_text'].tolist()
# print('중복 및 결측값 제거 완료')


# chunk_size = 300  # 청크 크기 설정
# chunk_overlap = 100  # 청크 오버랩 설정
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# # 데이터 분할
# split_docs = []
# for doc in tqdm(documents, desc="데이터 분할 중"):
#     split_docs.extend(text_splitter.create_documents([doc]))
# print('데이터 분할 완료')

# # 문서 객체 생성
# doc_objects = [Document(page_content=doc) for doc in documents]

# # 2. 벡터화 및 인덱싱
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # 상태바를 사용하여 벡터화
# doc_vectors = []
# for doc in tqdm(split_docs, desc="벡터화 중"):
#     doc_vectors.append(embeddings.embed_documents([doc.page_content])[0])
# print('벡터화 완료')

# # 상태바를 사용하여 인덱싱
# docstore = InMemoryDocstore(dict(zip(range(len(doc_objects)), doc_objects)))
# index_to_docstore_id = {i: i for i in tqdm(range(len(doc_objects)), desc="인덱싱 중")}
# vector_store = FAISS.from_documents(documents=doc_objects, embedding=embeddings)
# print('인덱싱 완료')

# # 3. 질의 처리 및 검색
# query = "cat1이 A03 인 가게 검색해줘"  # 사용자의 질의


# # 유사 문서 검색
# related_docs = vector_store.similarity_search(query, k=5)  # 여기에서 문자열을 전달합니다.
# print('유사 문서 검색 완료')

# # 4. 생성 모델 적용
# # OpenAI API 키 설정
# # GPT_API_KEY = os.getenv('GPT_API_KEY')

# # OpenAI API 키 설정
# # openai.api_key = GPT_API_KEY

# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=os.environ.get("GPT_API_KEY"),
# )

# # ChatGPT API를 통해 응답 생성
# def generate_chat_response(prompt):
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content

# # 관련 문서에서 텍스트만 추출
# related_texts = [doc.page_content for doc in tqdm(related_docs, desc="관련 문서 텍스트 추출 중")]
# print('GPT작동중...')

# # 생성 모델을 통해 응답 생성
# combined_texts = "\n".join(related_texts)
# prompt = (
#     "다음 텍스트를 바탕으로 너가 쿼리로 찾은 가게들의 이름(title), 주소(addr1), 코드(cat1) JSON 형식으로 출력해줘. key값은 name, addr1, cat1 으로 지정해줘:\n\n"
#     f"{combined_texts}\n\n"
#     "응답은 반드시 JSON 형식이어야 합니다. 예:\n"
#     '[{"title": "상점이름", "addr1": "부산광역시 부산진구 중앙번영로 6", "cat1": "A03"}]'
# )
# response = generate_chat_response(prompt)

# # 5. 결과 제공
# if __name__ == '__main__':
#     print(response)

# if __name__ == '__main__':
#     app.run(debug=True)


# import pandas as pd
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain.schema import Document
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from tqdm import tqdm
# import os
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFacePipeline



# load_dotenv()

# print('시작')

# # 1. CSV 파일 읽기 및 데이터 전처리
# file_path = './spot_data2.csv'  # 파일 경로를 지정하세요
# df = pd.read_csv(file_path)

# # 데이터 전처리: 필요한 열 선택 및 결합
# df['full_address'] = df['addr1'].fillna('') + ' ' + df['addr2'].fillna('')
# df['text'] = df['title'].fillna('') + ' - ' + df['full_address'].fillna('') + ' (' + df['tel'].fillna('').astype(str) + ')'
# df['doc_text'] = 'title: ' + df['title'] + ' addr1: ' + df['addr1'] + ' cat1: ' + df['cat1'] + ' ' + df['text']  # cat1 정보를 포함한 문서 텍스트 생성
# print('전처리 완료')

# # 중복 및 결측값 제거
# df.dropna(subset=['doc_text'], inplace=True)
# df.drop_duplicates(subset=['doc_text'], inplace=True)
# documents = df['doc_text'].tolist()
# print('중복 및 결측값 제거 완료')

# chunk_size = 300  # 청크 크기 설정
# chunk_overlap = 100  # 청크 오버랩 설정
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# # 데이터 분할
# split_docs = []
# for doc in tqdm(documents, desc="데이터 분할 중"):
#     split_docs.extend(text_splitter.create_documents([doc]))
# print('데이터 분할 완료')

# # 문서 객체 생성
# doc_objects = [Document(page_content=doc) for doc in documents]

# # 2. 벡터화 및 인덱싱
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # 상태바를 사용하여 벡터화
# doc_vectors = []
# for doc in tqdm(split_docs, desc="벡터화 중"):
#     doc_vectors.append(embeddings.embed_documents([doc.page_content])[0])
# print('벡터화 완료')

# # 상태바를 사용하여 인덱싱
# docstore = InMemoryDocstore(dict(zip(range(len(doc_objects)), doc_objects)))
# index_to_docstore_id = {i: i for i in tqdm(range(len(doc_objects)), desc="인덱싱 중")}
# vector_store = FAISS.from_documents(documents=doc_objects, embedding=embeddings)
# print('인덱싱 완료')

# # 3. 새로운 모델 로드 및 설정
# model_id = "kyujinpy/Ko-PlatYi-6B"
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# if torch.cuda.is_available():
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )
#     model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
#     device = "cuda"
# else:
#     model = AutoModelForCausalLM.from_pretrained(model_id)
#     device = "cpu"

# text_generation_pipeline = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=tokenizer,
#     task="text-generation",
#     temperature=0.2,
#     return_full_text=True,
#     max_new_tokens=300,
#     device=0 if device == "cuda" else -1  # GPU를 사용하도록 설정
# )

# prompt_template = """
# ### [INST]
# Instruction: Answer the question based on your knowledge.
# Here is context to help:

# {context}

# ### QUESTION:
# {question}

# [/INST]
#  """

# koplatyi_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# # Create prompt from prompt template
# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template=prompt_template,
# )

# # Create llm chain
# llm_chain = LLMChain(llm=koplatyi_llm, prompt=prompt)

# # 4. 질의 처리 및 검색
# query = "addr1에 전라남도가 들어간 가게 검색해줘"  # 사용자의 질의

# # 유사 문서 검색
# related_docs = vector_store.similarity_search(query, k=5)
# print('유사 문서 검색 완료')

# # 검색된 문서 확인
# print("검색된 문서:")
# for doc in related_docs:
#     print(doc.page_content)

# # 관련 문서에서 텍스트만 추출
# related_texts = [doc.page_content for doc in related_docs]
# print("관련 텍스트:")
# for text in related_texts:
#     print(text)

# # 검색된 문서들을 결합하여 컨텍스트로 제공
# combined_context = "\n".join(related_texts)

# print("Combined context:")
# print(combined_context)

# # LLMChain을 사용하여 질문을 처리하고 응답 생성
# try:
#     result = llm_chain.run(context=combined_context, question=query)
#     print(f"\n답변: {result}")
# except Exception as e:
#     print(f"LLMChain 실행 중 오류 발생: {e}")
#     print(f"오류 타입: {type(e)}")
#     import traceback
#     print(traceback.format_exc())


import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from tqdm import tqdm


# OpenAI API 키 설정
load_dotenv()
GPT_API_KEY = os.getenv('GPT_API_KEY')
os.environ["OPENAI_API_KEY"] = GPT_API_KEY

try:
    df = pd.read_csv("spot_data.csv")
except FileNotFoundError:
    print("CSV 파일을 찾을 수 없습니다.")
    exit()

df = df[['addr1', 'cat2', 'title', 'tel']]
df = df.fillna('')

# 각 행을 문자열로 변환
texts = []
for text in tqdm(df.itertuples(index=False), desc="텍스트 변환 중"):
    texts.append(f"지역: {text.addr1}, 카테고리: {text.cat2}, 이름: {text.title}, 전화번호: {text.tel}")

# 임베딩 생성 및 벡터 데이터베이스 생성
print("임베딩 생성 중...")
embeddings = OpenAIEmbeddings()
vectors = []

for text in tqdm(texts, desc="텍스트 임베딩 중"):
    vectors.append(embeddings.embed_query(text))

vectorstore = FAISS.from_texts(texts, embeddings)


# 프롬프트 템플릿 생성
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    사용자가 {location}에 위치한 {category} 카테고리의 관광지를 찾고 있습니다. 
    이 정보를 바탕으로 적절한 관광지를 추천해주세요. 
    관광지의 이름과 전화번호도 함께 제공해주세요.

    {context}
    """
)

# RetrievalQA 체인 생성
print("RetrievalQA 체인 생성 중...")
retriever = vectorstore.as_retriever()
llm = OpenAI()
qa_chain = RetrievalQA(llm=llm, retriever=retriever, return_source_documents=True, prompt=prompt_template)

# 질문 예시
query = "서울에 위치한 박물관을 추천해줘"

# 질문에 대한 답변 생성
print("질문에 대한 답변 생성 중...")
result = qa_chain({"question": query})

# 결과 출력
print("결과 출력 중...")
print(result['answer'])