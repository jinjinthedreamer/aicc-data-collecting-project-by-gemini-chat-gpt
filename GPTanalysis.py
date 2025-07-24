import os
import json
from dotenv import load_dotenv
import pandas as pd
from PIL import Image
import pytesseract
from openai import OpenAI
from typing import Dict, Any

# --- 1. 설정 (Configuration) ---
def setup_environment():
    """
    .env 파일에서 환경 변수를 로드하고 OpenAI 클라이언트를 초기화합니다.
    """
    load_dotenv()
    client = OpenAI()
    # API 키가 제대로 로드되었는지 확인
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY가 환경 변수에 설정되지 않았습니다. .env 파일을 확인하세요.")
    return client

# --- 2. 핵심 기능 함수 (Core Functions) ---
def extract_text_from_image(image_path: str) -> str:
    """
    주어진 이미지 경로에서 Tesseract OCR을 사용하여 텍스트를 추출합니다.
    """
    try:
        img = Image.open(image_path)
        # 한국어와 영어를 함께 인식하도록 설정
        raw_text = pytesseract.image_to_string(img, lang="kor+eng").strip()
        return raw_text
    except Exception as e:
        print(f"❗️ OCR 처리 중 오류 발생 ({image_path}): {e}")
        return "" # 오류 발생 시 빈 문자열 반환

def analyze_chat_with_gpt(client: OpenAI, text: str, session_id: str) -> Dict[str, Any]:
    """
    추출된 텍스트를 GPT-4o를 사용하여 분석하고, 구조화된 JSON 데이터를 반환합니다.
    """
    # GPT가 스크린샷 텍스트만으로 '현실적으로' 추출할 수 있는 지표들로 재구성
    prompt = f"""
    당신은 AICC(AI Contact Center)의 QA 분석 전문가입니다.
    아래 고객 상담 내용 텍스트를 분석하여, 요청된 8가지 항목에 대해 JSON 객체 형식으로만 응답해주세요.
    다른 부가 설명은 절대 추가하지 마세요.

    --- 상담 내용 텍스트 ---
    \"\"\"{text}\"\"\"
    ---

    요청 분석 항목 (JSON 형식):
    1.  "session_id": 제공된 세션 ID. (예: "{session_id}")
    2.  "phase": 대화의 현재 단계를 추정. (옵션: "초기상담", "문제파악", "해결방안제시", "마무리", "알수없음")
    3.  "inquiry_type": 고객의 핵심 문의 유형. (예: "배송지연", "결제오류", "상품정보문의", "단순문의")
    4.  "agent_evaluation": 상담사 응대 품질. (옵션: "S", "A", "B", "C", "D", "F")
    5.  "is_resolved_in_session": 이 대화 내에서 고객 문의가 해결되었는지 여부. (boolean: true / false)
    6.  "customer_sentiment": 고객의 주된 감정 상태. (옵션: "매우부정", "부정", "중립", "긍정", "매우긍정")
    7.  "is_escalated": 다른 팀으로 이관 또는 상급자 개입이 있었는지 여부. (boolean: true / false)
    8.  "system_errors": 텍스트에서 언급된 시스템 오류 코드나 메시지의 빈도. (객체 형식, 없으면 빈 객체 []. 예: {{"ERR_LOGIN_01": 1, "Timeout": 2}})
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"}, # JSON 모드 활성화
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, # 약간의 유연성을 부여하되 일관성 유지
            max_tokens=1024 # 지표가 많아져서 토큰 여유롭게 설정
        )
        # JSON 모드를 사용했으므로 바로 json.loads() 사용 가능
        analysis_result = json.loads(response.choices[0].message.content)
        return analysis_result
    except Exception as e:
        print(f"❗️ GPT API 호출 또는 JSON 파싱 중 오류 발생 (세션 ID: {session_id}): {e}")
        # API 실패 시 반환할 기본 오류 구조
        return {"session_id": session_id, "error_message": str(e)}

# --- 3. 메인 실행 로직 (Main Execution Logic) ---
def main():
    """
    메인 로직을 실행합니다: 이미지 폴더를 순회하며 분석하고 결과를 CSV로 저장합니다.
    """
    try:
        client = setup_environment()
    except ValueError as e:
        print(f"❌ 치명적 오류: {e}")
        return

    image_folder = "ocr_images"
    all_records = []

    print("🚀 AICC 로그 분석을 시작합니다...")

    # 폴더 존재 여부 확인
    if not os.path.isdir(image_folder):
        print(f"❌ 오류: '{image_folder}' 폴더를 찾을 수 없습니다.")
        return

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not image_files:
        print(f"🤷‍♀️ '{image_folder}' 폴더에 분석할 이미지 파일이 없습니다.")
        return

    for filename in image_files:
        session_id = os.path.splitext(filename)[0]
        image_path = os.path.join(image_folder, filename)
        print(f"📄 '{filename}' 분석 중...")

        # 1. OCR로 텍스트 추출
        raw_text = extract_text_from_image(image_path)
        if not raw_text:
            # OCR 실패 시, 오류 기록 남기고 다음 파일로
            record = {"session_id": session_id, "raw_text": "OCR_FAILED", "error_message": "텍스트 추출 실패"}
            all_records.append(record)
            continue

        # 2. GPT로 텍스트 분석
        analysis_data = analyze_chat_with_gpt(client, raw_text, session_id)

        # 3. 최종 결과에 원본 텍스트 추가
        analysis_data['raw_text'] = raw_text
        all_records.append(analysis_data)

    # 4. DataFrame 생성 및 CSV 저장
    df = pd.DataFrame(all_records)
    
    # 컬럼 순서 정리 (가독성을 위해)
    ordered_columns = [
        'session_id', 'inquiry_type', 'is_resolved_in_session', 'customer_sentiment',
        'agent_evaluation', 'is_escalated', 'phase', 'system_errors',
        'raw_text', 'error_message'
    ]
    # 존재하는 컬럼만 필터링하여 순서 재배치
    final_columns = [col for col in ordered_columns if col in df.columns]
    df = df[final_columns]

    output_filename = "aicc_analysis_results.csv"
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"\n✅ 분석 완료! 총 {len(all_records)}개의 세션이 처리되었습니다.")
    print(f"결과가 '{output_filename}' 파일에 저장되었습니다.")
    print("\n--- 최종 결과 미리보기 ---")
    print(df.head())

if __name__ == "__main__":
    main()


#GPT 버전, 별도 API Key는 유료이기에 코드만 작성됨