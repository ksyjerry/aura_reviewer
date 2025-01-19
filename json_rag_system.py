from typing import List, Dict, Any
import json
import os
import requests
import yaml
import urllib3
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from pwc_chat_model import PwcChatModel, API_KEY, PWC_MODEL

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# config.yaml 불러오기
with open("config.yaml", "r", encoding="utf-8") as yf:
    config = yaml.safe_load(yf)

class PwcEmbeddings(Embeddings):
    def __init__(
        self,
        api_url: str = "https://ngc-genai-proxy-stage.pwcinternal.com/v1/embeddings",
        api_key: str = API_KEY,
        model_name: str = PWC_MODEL.get("openai_embedding", "azure.text-embedding-3-small")
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for t in texts:
            emb = self._embed_single_text(t)
            embeddings.append(emb)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single_text(text)

    def _embed_single_text(self, text: str) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        payload = {
            "input": text,
            "model": self.model_name
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            verify=False
        )

        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
            else:
                raise ValueError(f"응답 데이터에 embedding이 없습니다: {data}")
        else:
            raise ValueError(f"Embedding 요청 실패: {response.status_code}, {response.text}")

class JsonRAGSystem:
    def __init__(self):
        self.embeddings = PwcEmbeddings()
        self.vector_store = None
        self.json_cache = {}
        self.initialize_system()

    def initialize_system(self):
        """메타데이터만 임베딩하여 벡터 스토어 생성"""
        documents = []
        
        try:
            # output.json에서 메타데이터 매핑 생성
            with open('json/output.json', 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
                
            # Link를 키로 하는 메타데이터 매핑 생성
            metadata_map = {item['Link']: item for item in metadata_list}
            print(f"\n메타데이터 매핑 수: {len(metadata_map)}")
            
            # json 폴더 내의 모든 파일 목록 가져오기
            json_files = [f.replace('.json', '') for f in os.listdir('json') 
                         if f.endswith('.json') and f != 'output.json']
            
            print(f"처리할 JSON 파일 수: {len(json_files)}")
            
            # 각 파일의 메타데이터 처리
            for file_id in json_files:
                try:
                    # Link(file_id)로 메타데이터 찾기
                    metadata = metadata_map.get(file_id)
                    
                    if metadata:
                        # 검색용 메타데이터 텍스트 생성
                        metadata_text = (
                            f"FSLI: {metadata['FSLI(s)']}\n"
                            f"Primary FSLI: {metadata['Primary FSLI']}\n"
                            f"EGA: {metadata['EGA']}\n"
                            f"Type: {metadata['Type']}"
                        )
                        
                        # Document 객체 생성 (hyperlink 추가)
                        doc = Document(
                            page_content=metadata_text,
                            metadata={
                                'file_name': f"{file_id}.json",
                                'link': file_id,
                                'fsli': metadata['FSLI(s)'],
                                'primary_fsli': metadata['Primary FSLI'],
                                'ega': metadata['EGA'],
                                'type': metadata['Type'],
                                'hyperlink': metadata.get('hyperlink', '')  # hyperlink 필드 추가
                            }
                        )
                        documents.append(doc)
                        print(f"메타데이터 매핑 성공: {file_id}.json -> {metadata['EGA']}")
                    else:
                        print(f"메타데이터 매핑 없음: {file_id}.json")
                    
                except Exception as e:
                    print(f"Error processing metadata for {file_id}.json: {str(e)}")
                    continue
                
            print(f"\n총 {len(documents)}개의 메타데이터가 처리되었습니다.")
            
            if not documents:
                raise ValueError("처리된 메타데이터가 없습니다.")
            
            # FAISS 벡터 스토어 생성
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            print("메타데이터 벡터 스토어 생성이 완료되었습니다.")
            
        except Exception as e:
            print(f"초기화 중 오류 발생: {str(e)}")
            raise

    def load_json_content(self, file_name: str) -> str:
        """JSON 파일의 실제 내용을 로드"""
        if file_name not in self.json_cache:
            try:
                file_path = os.path.join('json', file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                content_parts = []
                if 'sheets' in json_data:
                    for sheet_name, sheet_data in json_data['sheets'].items():
                        content_parts.append(f"\n[{sheet_name}]\n")
                        for row in sheet_data:
                            if 'content' in row:
                                row_text = []
                                for cell_data in row['content'].values():
                                    if isinstance(cell_data, dict) and 'value' in cell_data:
                                        value = str(cell_data['value']).strip()
                                        if value:
                                            row_text.append(value)
                                if row_text:
                                    content_parts.append(" ".join(row_text))
                
                self.json_cache[file_name] = "\n".join(content_parts)
            
            except Exception as e:
                print(f"Error loading content for {file_name}: {str(e)}")
                return ""
        
        return self.json_cache[file_name]

    def get_response(self, query: str, top_k: int = 3) -> str:
        try:
            # 메타데이터 기반으로 관련 문서 검색
            search_results = self.vector_store.similarity_search(query, k=top_k)
            print(f"\n[검색된 감사조서 {len(search_results)}개]")
            for i, doc in enumerate(search_results, 1):
                print(f"{i}. {doc.metadata['ega']} ({doc.metadata['file_name']})")
                print(f"   링크: {doc.metadata['hyperlink']}")
            
            chat_model = PwcChatModel(api_key=API_KEY, model=PWC_MODEL["openai"])
            all_responses = []
            
            # 각 문서별로 개별 처리 및 출력
            for i, doc in enumerate(search_results, 1):
                print(f"\n[문서 {i} - {doc.metadata['ega']}]")
                print(f"링크: {doc.metadata['hyperlink']}")
                
                content = self.load_json_content(doc.metadata['file_name'])
                context = (
                    f"문서 정보:\n"
                    f"- FSLI: {doc.metadata['fsli']}\n"
                    f"- EGA: {doc.metadata['ega']}\n"
                    f"- Type: {doc.metadata['type']}\n\n"
                    f"문서 내용:\n{content}\n"
                )

                messages = [
                    {"role": "system", "content": """
당신은 20년 이상의 경력을 가진 숙련된 회계감사 Manager입니다. 
회계감사조서를 읽고 질문에 대한 대답을 답변해주세요. 

엄격한 검토 기준:
1. "해당사항 없음", "예외사항 없음", "모두 회수됨" 등의 결론적 서술만으로는 절차 수행을 인정하지 않습니다.
2. 반드시 다음과 같은 구체적인 증거가 문서화되어 있어야 합니다:
   - 테스트 대상의 구체적인 모집단 (예: "외부조회서 발송 30건")
   - 실제 테스트 결과 데이터 (예: "회수된 조회서 28건, 미회수 2건")
   - 차이/예외사항에 대한 구체적인 후속 절차 내용과 결과
   - 대체적 테스트의 구체적인 내용과 결과 (해당되는 경우)

답변 시 필수 확인사항:
1. 구체적인 데이터 존재 여부
   - 모집단의 크기와 특성
   - 테스트 대상의 선정 기준과 범위
   - 실제 테스트된 항목의 수와 금액
   - 발견된 예외사항의 구체적인 내용

2. 실제 수행 증거 평가
   - "수행했다", "확인했다" 등의 서술만으로는 절차 수행을 인정하지 않음
   - 반드시 구체적인 테스트 방법과 결과가 문서화되어 있어야 함
   - 테스트 결과의 수치적 증거가 필요함

답변 형식:
1. 문서화된 실제 데이터만 인용
2. 구체적인 테스트 결과가 없는 경우:
   - "해당 절차의 실제 수행 증거가 문서화되어 있지 않습니다."
   - "구체적인 테스트 결과가 문서화되어 있지 않아 절차 수행 여부를 확인할 수 없습니다."
3. 필요한 보완 사항 제시:
   - 누락된 구체적 데이터
   - 필요한 추가 문서화 항목
   - 권장되는 구체적인 테스트 방법

주의사항:
- "확인 결과 이상 없음"과 같은 결론적 서술만 있는 경우 → 증거 불충분
- 구체적인 테스트 모집단과 범위가 명시되지 않은 경우 → 증거 불충분
- 실제 테스트 결과의 구체적 수치가 없는 경우 → 증거 불충분
- 예외사항 "없음"이라는 서술만 있는 경우 → 증거 불충분

제시된 감사조서에서 실제 문서화된 구체적인 테스트 내용과 결과만을 기반으로 답변해주세요.
결론적 서술이나 일반적인 확인 문구는 절차 수행의 증거로 인정하지 마세요.
"""},
                    {"role": "user", "content": f"""
다음 감사조서 문서를 참고하여 질문에 답변해주세요:

{context}

질문: {query}
"""}
                ]

                # 각 문서별 응답을 실시간으로 출력하고 저장
                response_text = ""
                for chunk in chat_model.get_response(messages):
                    if isinstance(chunk, dict) and "choices" in chunk:
                        content = chunk["choices"][0].get("delta", {}).get("content", "")
                        print(content, end='', flush=True)
                        response_text += content
                
                print("\n")  # 각 문서의 답변 사이에 빈 줄 추가
            
            # 응답 반환 (출력은 하지 않음)
            return ""
            
        except Exception as e:
            return f"오류 발생: {str(e)}"

    def print_metadata_mapping(self):
        """현재 시스템에 로드된 모든 감사조서의 메타데이터 매핑 정보 출력"""
        try:
            if not hasattr(self.vector_store, 'docstore') or not self.vector_store.docstore._dict:
                print("\n현재 로드된 메타데이터가 없습니다.")
                return
            
            print("\n=== 감사조서 메타데이터 매핑 정보 ===")
            print(f"총 문서 수: {len(self.vector_store.docstore._dict)}\n")
            
            # 모든 문서 정보 수집
            all_docs = []
            for doc_id in self.vector_store.docstore._dict:
                doc = self.vector_store.docstore._dict[doc_id]
                all_docs.append({
                    'file_name': doc.metadata['file_name'],
                    'fsli': doc.metadata['fsli'],
                    'ega': doc.metadata['ega'],
                    'type': doc.metadata['type']
                })
            
            if not all_docs:
                print("매핑된 문서가 없습니다.")
                return
            
            # FSLI 기준으로 정렬
            all_docs.sort(key=lambda x: (x['fsli'], x['ega']))
            
            # 정보 출력
            current_fsli = None
            for doc in all_docs:
                if current_fsli != doc['fsli']:
                    current_fsli = doc['fsli']
                    print(f"\n[FSLI: {current_fsli}]")
                
                print(f"파일명: {doc['file_name']}")
                print(f"  - EGA: {doc['ega']}")
                print(f"  - Type: {doc['type']}")
                print()
            
        except Exception as e:
            print(f"메타데이터 매핑 정보 출력 중 오류 발생: {str(e)}")

def main():
    # 시스템 초기화
    print("RAG 시스템을 초기화하는 중...")
    rag_system = JsonRAGSystem()
    
    # 대화 루프
    print("\n회계감사 관련 질문을 입력해주세요")
    print("- 종료: 'quit' 입력")
    print("- 참조할 감사조서 수 변경: 'set k=숫자' 입력")
    print("- 메타데이터 매핑 정보 확인: 'meta' 입력")
    
    top_k = 3  # 기본값
    
    while True:
        query = input("\n질문: ").strip()
        if query.lower() == 'quit':
            break
        elif query.lower() == 'meta':
            rag_system.print_metadata_mapping()
            continue
        elif query.lower().startswith('set k='):
            try:
                top_k = int(query.split('=')[1])
                print(f"참조할 감사조서 수가 {top_k}개로 설정되었습니다.")
                continue
            except:
                print("올바른 숫자를 입력해주세요.")
                continue
            
        try:
            response = rag_system.get_response(query, top_k=top_k)
            print("\n답변:", response)
        except Exception as e:
            print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 