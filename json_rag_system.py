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

    def print_metadata_mapping(self) -> str:
        """메타데이터 매핑 정보를 문자열로 반환"""
        try:
            with open('json/output.json', 'r', encoding='utf-8') as f:
                docs = json.load(f)
            
            output = []
            current_fsli = None
            
            for doc in docs:
                if current_fsli != doc['FSLI(s)']:
                    current_fsli = doc['FSLI(s)']
                    output.append(f"\n[FSLI: {current_fsli}]")
                
                output.append(f"파일명: {doc.get('Link', '없음')}")
                output.append(f"  - EGA: {doc.get('EGA', '없음')}")
                output.append(f"  - Type: {doc.get('Type', '없음')}")
                output.append("")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"메타데이터 매핑 정보 출력 중 오류 발생: {str(e)}"

    def get_response(self, query: str, top_k: int = 3) -> None:
        try:
            # 벡터 검색 먼저 수행
            initial_results = self.vector_store.similarity_search(query, k=top_k*3)
            
            # 쿼리에서 주요 FSLI 키워드 추출
            fsli_keywords = {
                " Inventory": ["재고", "재고자산", "상품", "제품", "원재료", "재공품", "저장품", "inventory"],
                " Trade Receivables": ["매출채권", "수취채권", "외상매출금", "받을어음", "receivables"],
                " Revenue": ["매출", "수익", "영업수익", "매출액", "revenue"],
                " Trade Payables": ["매입", "매입채무", "지급채무", "외상매입금", "지급어음", "payables"],
                " Property, Plant and Equipment": ["유형자산", "비품", "차량운반구", "기계장치", "공구와기구", "ppe", "유형자산"],
                " Cash and Cash Equivalents": ["현금", "현금성자산", "보통예금", "당좌예금", "cash"],
                " Planning Activities": ["계획", "감사계획", "planning"],
                " Completion Activities": ["종결", "감사종결", "completion"],
            }
            
            # 쿼리와 관련된 FSLI 찾기
            target_fslis = []
            for fsli, keywords in fsli_keywords.items():
                if any(keyword in query.lower() for keyword in keywords):
                    target_fslis.append(fsli)
            
            # 검색 결과 필터링
            if target_fslis:
                # FSLI 기반 필터링
                filtered_results = [
                    doc for doc in initial_results
                    if any(target_fsli in doc.metadata.get('fsli', '') for target_fsli in target_fslis)
                ]
                
                # 필터링된 결과가 있으면 사용, 없으면 원래 결과 사용
                search_results = filtered_results[:top_k] if filtered_results else initial_results[:top_k]
            else:
                # FSLI 키워드가 없으면 원래 검색 결과 사용
                search_results = initial_results[:top_k]
            
            # 디버깅 정보 출력
            yield f"[검색 정보]\n"
            if target_fslis:
                yield f"- 관련 FSLI: {', '.join(target_fslis)}\n"
            yield f"- 검색된 감사조서: {len(search_results)}개\n\n"
            
            # 검색된 문서 정보 출력
            for i, doc in enumerate(search_results, 1):
                yield f"{i}. {doc.metadata['ega']} ({doc.metadata['file_name']}) - "
                yield f"[LINK]({doc.metadata['hyperlink']})\n"
                yield f"   FSLI: {doc.metadata['fsli']}\n"
            
            chat_model = PwcChatModel(api_key=API_KEY, model=PWC_MODEL["openai"])
            
            # 각 문서별로 개별 처리
            for i, doc in enumerate(search_results, 1):
                yield f"\n[문서 {i} - {doc.metadata['ega']}] - "
                yield f"[LINK]({doc.metadata['hyperlink']})\n"
                
                content = self.load_json_content(doc.metadata['file_name'])
                context = (
                    f"문서 정보:\n"
                    f"- FSLI: {doc.metadata['fsli']}\n"
                    f"- EGA: {doc.metadata['ega']}\n"
                    f"- Type: {doc.metadata['type']}\n\n"
                    f"문서 내용:\n{content}\n"
                )

                # 부정위험 관련 키워드 매핑
                fraud_risk_keywords = {
                    "경영진 통제무력화": ["JET", "Journal Entry Test", "journal entry", "분개", "전표", "Halo for Journals"],
                    "수익인식": ["매출", "수익", "revenue", "sales"],
                    "자산유용": ["자금", "현금", "유용", "횡령", "misappropriation"],
                }

                # 시스템 프롬프트에 추가할 내용
                """
                부정위험 관련 키워드가 포함된 경우 다음 사항을 고려하여 답변해주세요:

                1. 경영진의 통제무력화 위험
                - JET(Journal Entry Test) 관련 절차
                - Halo for Journals 분석 결과
                - 비경상적인 분개 검토 결과

                2. 수익인식 관련 위험
                - 매출/수익 인식 기준 검토
                - 비경상적 거래 분석

                3. 자산유용 관련 위험
                - 자금 통제 검토
                - 비정상 거래 패턴 분석
                """

                messages = [
                    {"role": "system", "content": """
당신은 20년 이상의 경력을 가진 숙련된 회계감사 Manager입니다. 
답변 시 다음 사항을 반드시 준수해주세요:

1. 참고 정보 명시
- 어떤 시트의 어떤 섹션을 참고했는지 구체적으로 명시
- 예: "시트 'Control Testing'의 'Test of Controls' 섹션에서..."

2. 구체적인 데이터 인용
- 계정과목명, 거래처명, 금액, 날짜 등 구체적인 데이터 포함
- 단순히 "수행했다" 대신 실제 테스트 결과 명시
- 예: "매출채권 외부조회 대상 거래처 A사(XX백만원), B사(YY백만원)에 대해..."

3. 표 형식 활용
- 데이터를 표로 나타낼 수 있는 경우 반드시 markdown 표 형식 사용
- 예시 표 형식:
| 구분 | 거래처 | 금액(백만원) | 확인결과 |
|-----|--------|-------------|----------|
| 매출채권 | A사 | XX | 일치 |

4. 검토 결과 구조화
- 발견사항을 명확한 구조로 제시
  a) 테스트 대상 및 범위
  b) 구체적인 테스트 절차
  c) 테스트 결과 (가능한 경우 표 형식)
  d) 예외사항 및 후속 조치

5. 증거 기반 결론
- "수행했다"는 서술만으로는 불충분
- 반드시 구체적인 데이터나 테스트 결과를 기반으로 결론 도출
- 데이터가 불충분한 경우 "해당 내용을 확인할 수 있는 구체적인 데이터가 문서에 없음" 명시

6. 추가 검토 필요사항
- 문서화가 불충분한 영역 지적
- 필요한 추가 데이터나 테스트 제안

답변은 항상 구체적인 데이터와 실제 테스트 결과를 중심으로 작성하며, 
데이터를 표현하는데 표 형식이 효과적인 경우 반드시 표로 정리하여 제시해주세요.
"""},
                    {"role": "user", "content": f"""
다음 감사조서 문서를 참고하여 질문에 답변해주세요:

{context}

질문: {query}
"""}
                ]

                # 각 문서별 응답을 실시간으로 생성
                for chunk in chat_model.get_response(messages):
                    if isinstance(chunk, dict) and "choices" in chunk:
                        content = chunk["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                
                yield "\n\n"
            
        except Exception as e:
            yield f"오류 발생: {str(e)}"

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
            print(rag_system.print_metadata_mapping())
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
            responses = list(rag_system.get_response(query, top_k=top_k))
            for response in responses:
                print("\n답변:", response)
        except Exception as e:
            print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 