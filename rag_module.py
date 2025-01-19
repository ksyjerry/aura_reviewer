import os
import requests
import yaml
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
def build_vector_store(ai_rcm_list: List[Dict[str, Any]],
                       embedding_model=None,
                       persist_directory: str = None) -> FAISS:
    """
    주어진 AI_RCM 목록을 벡터 스토어에 저장.
    """
    if embedding_model is None:
        # 예: v1/embeddings 를 사용하는 URL
        # PWC_MODEL 안에 'openai_embedding' 항목이 있다면 사용
        default_model_name = PWC_MODEL.get("openai_embedding")
        embedding_model = PwcEmbeddings(
            api_url="https://ngc-genai-proxy-stage.pwcinternal.com/v1/embeddings",
            api_key=API_KEY,
            model_name=default_model_name
        )

    try:
        docs_for_db = []
        for rcm_data in ai_rcm_list:
            content_str = "\n".join(filter(None, [
                rcm_data.get("process_name", ""),
                rcm_data.get("subprocess_name", ""),
                rcm_data.get("risk_description", ""),
                rcm_data.get("control_title", ""),
                rcm_data.get("control_description", ""),
            ]))

            metadata = {
                "control_code": rcm_data.get("control_code", "unknown_code"),
                "rcm_data": rcm_data
            }

            doc = Document(page_content=content_str, metadata=metadata)
            docs_for_db.append(doc)

        print(f"[build_vector_store] 총 {len(docs_for_db)}개의 문서를 임베딩합니다...")
        vector_store = FAISS.from_documents(docs_for_db, embedding_model)
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            vector_store.save_local(persist_directory)
        return vector_store

    except Exception as e:
        print(f"[build_vector_store] 에러 발생: {str(e)}")
        raise

def retrieve_ai_rcm(query_text: str, vector_store: FAISS, k: int = 1) -> Dict[str, Any]:
    """
    RAG 검색 함수 예시
    """
    try:
        print(f"[retrieve_ai_rcm] '{query_text}'로 검색합니다. (k={k})")
        docs = vector_store.similarity_search(query_text, k=k)
        if not docs:
            return {}
        top_doc = docs[0]
        return top_doc.metadata.get("rcm_data", {})
    except Exception as e:
        print(f"[retrieve_ai_rcm] 에러 발생: {str(e)}")
        return {}




###################################################
# 메인 실행부
###################################################
if __name__ == "__main__":


    

    sample_ai_rcm_list = [
        {
            "process_name": "유무형자산 취득",
            "subprocess_name": "",
            "risk_description": "자산 구매와 불일치한 전표처리 위험",
            "control_code": "C-FA-10-04",
            "control_title": "유무형자산 취득 통제",
            "control_description": "회계부서장이 증빙 대조 후 승인"
        },
        {
            "process_name": "재무제표 결산",
            "subprocess_name": "",
            "risk_description": "승인되지 않은 전표 처리 위험",
            "control_code": "C-FR-20-01",
            "control_title": "수기전표 승인 통제",
            "control_description": "작성자와 승인자 분리"
        }
    ]
    try:
        print("[INFO] RAG: 벡터스토어 생성 시작...")
        vector_db = build_vector_store(sample_ai_rcm_list, persist_directory="faiss_index")
        print("[INFO] RAG: 벡터스토어 생성 완료.")

        query_text = "기계장치 취득과 관련된 통제"
        result = retrieve_ai_rcm(query_text, vector_db)
        print("\n[검색 결과]")
        print(" - Control Code:", result.get("control_code"))
        print(" - Control Title:", result.get("control_title"))
        print(" - Control Description:", result.get("control_description"))
    except Exception as e:
        print("[MAIN] RAG 에러:", str(e)) 