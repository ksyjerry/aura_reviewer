from pwc_chat_model import PwcChatModel 
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

class ExcelDocumentQA:
    def __init__(self, config_path: str = "config.yaml"):
        """설정 파일을 읽고 LLM 클라이언트 초기화"""
        with open(config_path, "r", encoding='utf-8') as yaml_file:
            config = yaml.safe_load(yaml_file)
        
        self.client = PwcChatModel(
            api_key=config['api_key'],
            model=config['pwc_model']['openai']
        )
        
    def _load_json_data(self, json_path: str) -> Dict:
        """JSON 파일 읽기"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _create_system_prompt(self, json_data: Dict) -> str:
        """시스템 프롬프트 생성"""
        # 여러 파일 리를 위한 메타데이터 구성
        if 'files_data' in json_data:
            # 여러 파일이 있는 경우
            files_info = "\n".join([
                f"- 파일명: {file_data['metadata']['file_name']}" 
                for file_data in json_data['files_data']
            ])
            total_files = len(json_data['files_data'])
            
            system_prompt = f"""당신은 20년 이상의 경력을 가진 숙련된 회계감사 Manager입니다. 
검토중인 감사조서 정보:
- 총 파일 수: {total_files}
검토 대상 파일:
{files_info}

"""
        else:
            # 단일 파일인 경우
            metadata = json_data['metadata']
            system_prompt = f"""당신은 20년 이상의 경력을 가진 숙련된 회계감사 Manager입니다. 회계감사조서를 읽고 질문에 대한 대답을 답변해주세요. 질문이 회계감사와 관련이 없는 경우 회계감사와 관련된 질문에 대한 대답만 제공한다고 알려주세요. 절대로 회계감사 이외의 질문에 대한 대답은 하지 않습니다.
검토중인 감사조서 정보:
- 자료명: {metadata.get('file_name', '문서')}
"""

        system_prompt += """
전문성:
- 회계감사기준서(GAAS)에 대한 깊은 이해
- 한국채택국제회계기준(K-IFRS)의 전문적 지식
- 내부통제 및 위험평가 전문가
- 분석적 절차 수행 경험 풍부
- 표본감사, 문서검증, 재계산 등 감사절차 전문가

검토 스타일:
- 추상적인 검토 코멘트를 제시하지 말고 실제 데이터를 기반으로 구체적인 피드백을 제공해주세요.
- 금액간 대사가 일치하는지 철저하게 검토해주세요. 
- 감사 결론에 어긋나는 발견사항을 철저하게 검토해주세요.
- 앞뒤 문맥에 맞지 않거나 절차가 맞지 않는 부분을 철저하게 검토해주세요.
- 수치 분석 시에는 반드시 구체적인 숫자를 언급하며 검토해주세요.
- 발견된 이슈는 중요성 수준을 고려하여 평가해주세요.

검토 시 중점 사항:
1. 감사증거
  - 충분하고 적합한 감사증거 획득 여부
  - 문서화의 품질과 완전성
  - 외부 증빙과의 대사 여부
  - 금액의 정확성과 완전성
  - 증빙 문서의 신뢰성

2. 절차의 수행여부
  - 절차의 충분성
  - 절차의 적절성
  - 절차의 효과성
  - 절차의 효율성


응답 형식:
1. 구체적인 수치를 포함한 분석 결과 제시
2. 발견된 이슈의 중요도 명시 (상/중/하)
3. 필요한 경우 추가 감사절차 제안
4. 명확한 개선 권고사항 제시

제시된 데이터를 바탕으로, 회계감사조서 검토자로서 전문적이고 구체적인 피드백을 제공해주세요. 
특히 감사기준과의 부합성, 문서화의 적절성, 그리고 추가 검토가 필요한 영역을 중점적으로 파악해주시기 바랍니다.
**회계감사 이외의 질문에는 대답할 수 없다고 명시해주세요** """

        return system_prompt

    def ask(self, json_path: str, question: str) -> str:
        """JSON 데이터에 대한 질문하기"""
        try:
            # JSON 데이터 로드
            json_data = self._load_json_data(json_path)
            
            # 메시지 구성
            messages = [
                {
                    "role": "system",
                    "content": self._create_system_prompt(json_data)
                },
                {
                    "role": "user", 
                    "content": f"다음은 엑셀 파일의 JSON 데이터입니다:\n{json.dumps(json_data, ensure_ascii=False, indent=2)}\n\n질문: {question}"
                }
            ]
            
            # LLM에 질문하고 응답 받기
            response = self.client.get_response(messages)
            
            # 스트림 응답을 하나의 문자열로 결합
            full_response = ""
            for chunk in response:
                if isinstance(chunk, dict):
                    if "error" in chunk:
                        return f"오류: {chunk['error']}"
                    elif "choices" in chunk and len(chunk["choices"]) > 0:
                        content = chunk["choices"][0].get("delta", {}).get("content", "")
                        full_response += content
            
            return full_response
            
        except Exception as e:
            return f"오류 발생: {str(e)}"

def main():
    # 설정 및 초기화
    qa = ExcelDocumentQA()
    
    # JSON 파일 경로
    json_path = "./workpaper/매출채권조서.json" # 또는 다른 JSON 파일 경로
    
    # 대화형 인터페이스
    print(f"\nJSON 파일 '{json_path}'에 대해 질문해주세요. (종료하려면 'quit' 입력)")
    
    while True:
        question = input("\n질문: ").strip()
        
        if question.lower() in ['quit', 'exit', '종료']:
            break
            
        if not question:
            continue
            
        print("\n답변:")
        response = qa.ask(json_path, question)
        print(response)
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()



