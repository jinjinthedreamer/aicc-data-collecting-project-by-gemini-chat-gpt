import os
import pandas as pd
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# ================================
# 1. .env 파일에서 API 키 로딩
# ================================
load_dotenv("API_KEY.env")
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not HF_TOKEN:
    raise ValueError("❌ Hugging Face API 키를 .env 파일에서 불러오지 못했습니다!")

# ================================
# 2. 모델 설정 (무료 사용 가능한 모델)
# ================================
client = InferenceClient(
    model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",  # ✅ 무료로 안정적인 모델
    token=HF_TOKEN
)

# ================================
# 3. 챗봇 로그 데이터 분석
# ================================
try:w
    logs = pd.read_csv("chatbot_logs.csv")
    handover_rate = logs["handover"].mean()
    print(f"\n📊 Human Handover Rate: {handover_rate:.2%}")
    
    prompt_logs = f"""
아래는 챗봇 대화 로그의 통계입니다:
- 총 대화 수: {len(logs)}
- 인간 이관 비율: {handover_rate:.2%}

1) 이 지표가 의미하는 바를 기획자 관점에서 요약해 주세요.
2) 개선을 위한 3가지 제안을 해주세요.
"""
    response_logs = client.chat_completion(
        messages=[{"role": "user", "content": prompt_logs}],
        temperature=0.7,
        max_tokens=500
    )
    print("\n📈 챗봇 로그 분석 요약:")
    print(response_logs.choices[0].message["content"])

except Exception as e:
    print("\n⚠️ 챗봇 로그 분석 실패:", e)

# ================================
# 4. 통화 녹취 분석
# ================================
try:
    with open("call_transcript.txt", "r", encoding="utf-8") as f:
        transcript = f.read()

    prompt_transcript = f"""
다음은 고객 통화 녹취를 텍스트로 변환한 내용입니다:
\"\"\"{transcript[:1000]}...\"\"\"

1) 고객의 주요 불만/요구를 3문장으로 요약하세요.
2) 해당 이슈를 해결하기 위해 어떤 기능 또는 프로세스를 기획할 수 있을지 제안해주세요.
"""
    response_transcript = client.chat_completion(
        messages=[{"role": "user", "content": prompt_transcript}],
        temperature=0.7,
        max_tokens=500
    )
    print("\n📞 통화 분석 결과:")
    print(response_transcript.choices[0].message["content"])

except Exception as e:
    print("\n⚠️ 통화 녹취 분석 실패:", e)

# ================================
# 5. PPT 슬라이드용 요약 생성
# ================================
try:
    ppt_prompt = """
위에서 얻은 제안 내용을 바탕으로, 
PPT 한 슬라이드 분량의 마크다운 형식(제목 + 핵심 3개 bullet)으로 출력해 주세요.
"""
    response_ppt = client.chat_completion(
        messages=[{"role": "user", "content": ppt_prompt}],
        temperature=0.7,
        max_tokens=300
    )
    print("\n🖼 PPT 슬라이드용 마크다운:")
    print(response_ppt.choices[0].message["content"])

except Exception as e:
    print("\n⚠️ PPT 요약 생성 실패:", e)


#테스트 데이터 및 API 키 활용 시, 활용가능