import requests
import base64
import json
import yaml
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# YAML 파일에서 설정 읽기
with open("config.yaml", "r") as yaml_file:
    config = yaml.safe_load(yaml_file)

key = config['api_key']
pwc_model = config['pwc_model']

class PwcChatModel:
    def __init__(self, api_key, model=pwc_model['openai'], temperature=0, stream=True):
        self.api_key = api_key
        self.api_url = "https://ngc-genai-proxy-stage.pwcinternal.com/chat/completions"
        self.model = model
        self.temperature = temperature
        self.stream = stream

    def get_response(self, input_text):
        headers = {
            "User-Agent": "curl/8.9.1",
            "accept": "application/json",
            "accept-encoding": "gzip, deflate, br",
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
            "Connection": "keep-alive",  
            "x-request-type": "sync",
            'Accept-Encoding': 'identity',
        }

        payload = {
            "temperature": self.temperature,
            "top_p": 1,
            "n": 1,
            "stream": self.stream,
            "messages": input_text,
            "model": self.model,
        }
        
        response = requests.post(
            self.api_url, 
            headers=headers, 
            json=payload, 
            verify=False,
            stream=self.stream
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            json_str = line[6:]  # 'data: ' 제거
                            if json_str.strip() == '[DONE]':
                                break
                            json_data = json.loads(json_str)
                            
                            # JSON 응답 구조 확인 및 텍스트 추출
                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                choice = json_data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    yield {'choices': [{'delta': {'content': choice['delta']['content']}}]}
                    except Exception as e:
                        print(f"Error processing line: {e}")
                        continue
        else:
            yield {"error": f"Error: {response.status_code}, {response.text}"}
        


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def text_input_test():
    client = PwcChatModel(api_key=key, model=pwc_model['openai'])

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "2022년 월드컵 우승팀은?"}
    ]
    
    # 응답 스트림 처리
    for chunk in client.get_response(messages):
        print("Chunk received:", chunk)  # 실제 응답 형식 확인
        if isinstance(chunk, dict):
            if "error" in chunk:
                print("Error:", chunk["error"])
            elif "choices" in chunk and len(chunk["choices"]) > 0:
                content = chunk["choices"][0].get("delta", {}).get("content", "")
                print("Content:", content, end="", flush=True)

def img_input_test(image_path,question):
    client = PwcChatModel(api_key=key)
    
    base64_image = encode_image(image_path)
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{question}",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                
            ],
        }
    ]    

    # 응답 받기
    output_text = client.get_response(messages)
    print(output_text)




if __name__ == '__main__':
    text_input_test()
    # image_path = "Vision.png"
    # question = "이 이미지는 무슨내용인가요??"

    # img_input_test(image_path, question)



