import os
import requests
import base64
import json
import yaml
import urllib3
from typing import List, Dict, Any

# LangChain 관련 임포트
from dotenv import load_dotenv


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1) config.yaml 불러오기
with open("config.yaml", "r", encoding="utf-8") as yf:
    config = yaml.safe_load(yf)

API_KEY = config["api_key"]
PWC_MODEL = config["pwc_model"] 

class PwcChatModel:
    """
    PwC Proxy를 통한 chat/completions API 호출용 클래스.
    streaming=True 로 설정되어 있으면, 서버의 응답을
    yield하여 실시간으로 출력할 수 있습니다.
    """
    def __init__(self, api_key: str, model: str = None, temperature: float = 0):
        self.api_key = api_key
        self.api_url = "https://ngc-genai-proxy-stage.pwcinternal.com/chat/completions"
        # config.yaml에서 정의된 모델이 있으면 기본 값을 사용
        if not model:
            # openai 키에 해당하는 값을 기본 사용
            model = PWC_MODEL.get("openai", "azure.gpt-3.5-turbo")
        self.model = model
        self.temperature = temperature

    def get_response(self, input_text: List[Dict[str, str]]):
        """
        input_text: [{"role": "system", "content": ...}, {"role": "user", "content": ...}, ...]
        """
        headers = {
            "User-Agent": "curl/8.9.1",
            "accept": "application/json",
            "accept-encoding": "gzip, deflate, br",
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
            "Connection": "keep-alive",
            "x-request-type": "sync",
            "Accept-Encoding": "identity",
        }

        payload = {
            "temperature": self.temperature,
            "top_p": 1,
            "n": 1,
            "stream": True,      # 스트리밍 응답
            "messages": input_text,
            "model": self.model,
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            verify=False,  # SSL 이슈가 있으면 인증서 등록 고려
            stream=True
        )

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            json_str = line[6:]  # 'data: ' 제거
                            if json_str.strip() == "[DONE]":
                                break
                            json_data = json.loads(json_str)

                            # JSON 응답 구조 확인 및 텍스트 추출
                            if "choices" in json_data and len(json_data["choices"]) > 0:
                                choice = json_data["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    yield {"choices": [{"delta": {"content": choice["delta"]["content"]}}]}
                    except Exception as e:
                        print(f"Error processing line: {e}")
                        continue
        else:
            yield {"error": f"Error: {response.status_code}, {response.text}"}


###################################################
# 이미지 base64 인코딩 (필요 시 사용)
###################################################
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


###################################################
# 예제) Chat 모델 호출 테스트
###################################################
def text_input_test():
    client = PwcChatModel(api_key=API_KEY, model=PWC_MODEL["openai"])
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "2022년 월드컵 우승팀은?"}
    ]

    print("[INFO] Chat 모델 호출을 시작합니다.")
    for chunk in client.get_response(messages):
        if isinstance(chunk, dict):
            if "error" in chunk:
                print("Error:", chunk["error"])
            elif "choices" in chunk and len(chunk["choices"]) > 0:
                content = chunk["choices"][0].get("delta", {}).get("content", "")
                print(content, end="", flush=True)
    print("\n[INFO] Chat 모델 호출 종료")


###################################################
# 메인 실행부
###################################################
if __name__ == "__main__":
    """
    python rag_module.py 로 실행 시, 
    1) Chat API 호출 테스트
    2) 샘플 RAG 테스트
    """
    # 1) Chat 테스트
    text_input_test()