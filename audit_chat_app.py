import streamlit as st
from excel import process_excel_content
from aura_reviewer import ExcelDocumentQA
import json
import time

class AuditReviewChatbot:
    def __init__(self, config_path: str = "config.yaml"):
        """감사조서 리뷰 챗봇 초기화"""
        self.qa_engine = ExcelDocumentQA(config_path)
        
    def process_excel_to_json(self, file_content: bytes, file_name: str) -> dict:
        """엑셀 파일을 JSON 구조로 변환"""
        return process_excel_content(file_content, file_name)
            
    def get_response_stream(self, json_data_list: list, question: str):
        """스트리밍 방식으로 응답 생성"""
        # 여러 파일의 데이터를 하나의 컨텍스트로 결합
        combined_context = {
            'metadata': {
                'total_files': len(json_data_list),
                'files': [data['metadata']['file_name'] for data in json_data_list],
                'sheets_info': {}
            },
            'files_data': json_data_list
        }
        
        messages = [
            {
                "role": "system",
                "content": self.qa_engine._create_system_prompt(combined_context)
            },
            {
                "role": "user",
                "content": f"다음은 엑셀 파일들의 JSON 데이터입니다:\n{json.dumps(combined_context, ensure_ascii=False, indent=2)}\n\n질문: {question}"
            }
        ]
        return self.qa_engine.client.get_response(messages)

def initialize_session_state():
    """세션 상태 초기화"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'json_data_list' not in st.session_state:
        st.session_state.json_data_list = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = set()

def main():
    st.set_page_config(
        page_title="감사조서 리뷰 챗봇",
        page_icon="📊",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("📊 감사조서 리뷰 챗봇")
    
    chatbot = AuditReviewChatbot()
    
    # 사이드바에 파일 업로더와 파일 목록 표시
    with st.sidebar:
        st.header("파일 관리")
        
        # 파일 업로더
        uploaded_files = st.file_uploader(
            "감사조서 파일을 업로드하세요 (XLSM)",
            type=['xlsm'],
            accept_multiple_files=True,
            help="여러 개의 파일을 동시에 업로드할 수 있습니다. (최대 200MB/파일)"
        )
        
        # 업로드된 파일 처리
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                
                # 새로운 파일만 처리
                if file_name not in st.session_state.uploaded_files:
                    try:
                        st.session_state.processing = True
                        
                        # 파일 크기 확인
                        file_size = uploaded_file.size / (1024 * 1024)  # MB로 변환
                        if file_size > 200:
                            st.error(f"파일 '{file_name}'이 200MB를 초과합니다.")
                            continue
                        
                        # 파일 처리
                        with st.spinner(f"'{file_name}' 처리 중..."):
                            file_content = uploaded_file.read()
                            json_data = chatbot.process_excel_to_json(file_content, file_name)
                            
                            if json_data:
                                st.session_state.json_data_list.append(json_data)
                                st.session_state.uploaded_files.add(file_name)
                                st.success(f"✅ '{file_name}' 분석 완료!")
                    
                    except Exception as e:
                        st.error(f"'{file_name}' 처리 중 오류 발생: {str(e)}")
                    
                    finally:
                        st.session_state.processing = False
        
        # 업로드된 파일 목록 표시
        if st.session_state.uploaded_files:
            st.write("### 분석된 파일 목록")
            for idx, file_name in enumerate(st.session_state.uploaded_files, 1):
                st.write(f"{idx}. {file_name}")
            
            if st.button("모든 파일 초기화"):
                st.session_state.messages = []
                st.session_state.json_data_list = []
                st.session_state.uploaded_files = set()
                st.session_state.processing = False
                st.rerun()
    
    # 메인 영역: 채팅 인터페이스
    if st.session_state.json_data_list:
        # 구분선 추가
        st.markdown("---")
        
        # 이전 메시지 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 사용자 입력
        if prompt := st.chat_input("질문을 입력하세요"):
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 어시스턴트 응답
            with st.chat_message("assistant"):
                response_container = st.empty()
                full_response = ""
                
                try:
                    # 스트리밍 응답 처리
                    for chunk in chatbot.get_response_stream(st.session_state.json_data_list, prompt):
                        if isinstance(chunk, dict) and "choices" in chunk:
                            content = chunk["choices"][0].get("delta", {}).get("content", "")
                            full_response += content
                            response_container.markdown(full_response + "▌")
                    
                    response_container.markdown(full_response)
                    # 어시스턴트 메시지 저장
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"응답 생성 중 오류가 발생했습니다: {str(e)}")
    
    else:
        st.info("👈 사이드바에서 감사조서 파일을 업로드해주세요.")

if __name__ == "__main__":
    main() 