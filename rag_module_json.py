import os
import requests
import yaml
import json
import urllib3
from typing import List, Dict, Any

# LangChain 관련 임포트
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1) config.yaml 불러오기
with open("config.yaml", "r", encoding="utf-8") as yf:
    config = yaml.safe_load(yf)

API_KEY = config["api_key"]
PWC_MODEL = config["pwc_model"]  # 예: {"openai": "azure.gpt-3.5-turbo", "openai_embedding": "azure.text-embedding-3-small"}

###################################################
# PwCEmbeddings: Embeddings 상속 커스텀 클래스
###################################################
class PwcEmbeddings(Embeddings):
    """
    PwC Proxy의 /embeddings (또는 /v1/embeddings) 엔드포인트를 직접 요청하여
    임베딩을 반환하는 커스텀 Embeddings 클래스.

    LangChain의 OpenAIEmbeddings 대체 버전.
    embed_documents, embed_query 메서드를 구현하면,
    LangChain의 Vector Store (예: FAISS)와 연동 가능.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "azure.text-embedding-3-small"
    ):
        """
        api_url 예시:
         - "https://ngc-genai-proxy-stage.pwcinternal.com/v1/embeddings"
         - "https://ngc-genai-proxy-stage.pwcinternal.com/embeddings"
         - "https://ngc-genai-proxy-stage.pwcinternal.com/openai/deployments/{model}/embeddings"
        model_name 예시:
         - "azure.text-embedding-3-small"
         - "azure.text-embedding-ada-002"
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 문장(혹은 텍스트 문서)에 대한 임베딩 배열을 반환
        """
        embeddings = []
        for t in texts:
            emb = self._embed_single_text(t)
            embeddings.append(emb)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리 텍스트에 대한 임베딩을 반환
        """
        return self._embed_single_text(text)

    def _embed_single_text(self, text: str) -> List[float]:
        """
        PwC Proxy /embeddings 엔드포인트에 POST. 응답을 파싱하여 벡터 반환.
        """
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
            verify=False  # 사내 인증서 문제 시 임시방편
        )

        if response.status_code == 200:
            data = response.json()
            # OpenAI 호환 응답 구조: {"data":[{"embedding":[...]}], "model":"..."}
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
            else:
                raise ValueError(f"응답 데이터에 embedding이 없습니다: {data}")
        else:
            raise ValueError(f"Embedding 요청 실패: {response.status_code}, {response.text}")


###################################################
# RAG 예시 함수
###################################################
def build_vector_store(json_data: Dict[str, Any],
                      embedding_model=None,
                      persist_directory: str = None,
                      chunk_size: int = 10000,
                      chunk_overlap: int = 1000
                      ) -> FAISS:
    """
    JSON 문서를 벡터 스토어에 저장
    Args:
        chunk_size: 각 청크의 최대 문자 수
        chunk_overlap: 청크 간 중복되는 문자 수
    """
    if embedding_model is None:
        default_model_name = PWC_MODEL.get("openai_embedding")
        embedding_model = PwcEmbeddings(
            api_url="https://ngc-genai-proxy-stage.pwcinternal.com/v1/embeddings",
            api_key=API_KEY,
            model_name=default_model_name
        )

    try:
        docs_for_db = []
        current_chunk = ""
        current_metadata = {}
        
        def process_json(data, path=""):
            nonlocal current_chunk, current_metadata
            
            if isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    if isinstance(value, (dict, list)):
                        process_json(value, new_path)
                    else:
                        content = f"{new_path}: {str(value)}\n"
                        if len(current_chunk) + len(content) > chunk_size:
                            # 현재 청크가 최대 크기를 초과하면 새로운 Document 생성
                            if current_chunk:
                                docs_for_db.append(Document(
                                    page_content=current_chunk,
                                    metadata=current_metadata.copy()
                                ))
                            current_chunk = content
                            current_metadata = {"paths": [new_path], "values": [str(value)]}
                        else:
                            current_chunk += content
                            current_metadata.setdefault("paths", []).append(new_path)
                            current_metadata.setdefault("values", []).append(str(value))
                            
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    new_path = f"{path}[{idx}]"
                    if isinstance(item, (dict, list)):
                        process_json(item, new_path)
                    else:
                        content = f"{new_path}: {str(item)}\n"
                        if len(current_chunk) + len(content) > chunk_size:
                            if current_chunk:
                                docs_for_db.append(Document(
                                    page_content=current_chunk,
                                    metadata=current_metadata.copy()
                                ))
                            current_chunk = content
                            current_metadata = {"paths": [new_path], "values": [str(item)]}
                        else:
                            current_chunk += content
                            current_metadata.setdefault("paths", []).append(new_path)
                            current_metadata.setdefault("values", []).append(str(item))

        process_json(json_data)
        
        # 마지막 청크 처리
        if current_chunk:
            docs_for_db.append(Document(
                page_content=current_chunk,
                metadata=current_metadata
            ))
        
        print(f"[build_vector_store] 총 {len(docs_for_db)}개의 청크를 임베딩합니다...")
        vector_store = FAISS.from_documents(docs_for_db, embedding_model)
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            vector_store.save_local(persist_directory)
        return vector_store

    except Exception as e:
        print(f"[build_vector_store] 에러 발생: {str(e)}")
        raise

def search_document(query_text: str, vector_store: FAISS, k: int = 3) -> List[Dict[str, Any]]:
    """
    문서 검색 함수
    """
    try:
        print(f"[search_document] '{query_text}'로 검색합니다. (k={k})")
        docs = vector_store.similarity_search(query_text, k=k)
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "path": doc.metadata["path"],
                "value": doc.metadata["value"]
            })
        return results
    except Exception as e:
        print(f"[search_document] 에러 발생: {str(e)}")
        return []




###################################################
# 메인 실행부
###################################################
if __name__ == "__main__":
    try:
        # 매출채권조서 JSON 파일 로드
        with open(r"C:\Users\jkim564\Documents\ai_apps\Audit Reviewer\Aura Review\workpaper\매출채권조서.json", "r", encoding="utf-8") as f:
            workpaper_data = json.load(f)

        print("[INFO] RAG: 벡터스토어 생성 시작...")
        vector_db = build_vector_store(workpaper_data, persist_directory="faiss_index")
        print("[INFO] RAG: 벡터스토어 생성 완료.")

        # 검색 예시
        query_text = "매출채권 잔액"
        results = search_document(query_text, vector_db)
        
        print("\n[검색 결과]")
        for idx, result in enumerate(results, 1):
            print(f"\n결과 {idx}:")
            print(f"경로: {result['path']}")
            print(f"내용: {result['value']}")
            
    except Exception as e:
        print("[MAIN] RAG 에러:", str(e)) 