import streamlit as st
import os
from aura_reporting import excel_to_json
from excel_to_json import extract_engagement_id, making_excel_path, convert_xlsm_folder_to_json
from json_rag_system import JsonRAGSystem
import json

def init_session_state():
    """세션 상태 초기화"""
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'output_json_created' not in st.session_state:
        st.session_state.output_json_created = False
    if 'json_files_created' not in st.session_state:
        st.session_state.json_files_created = False
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None

def main():
    st.title("감사조서 검토 시스템")
    
    init_session_state()
    
    # Step 1: Aura Reporting 엑셀 파일 업로드
    st.header("Step 1: Aura Reporting 엑셀 파일 업로드")
    excel_file = st.file_uploader("Aura Reporting 엑셀 파일을 업로드하세요", type=['xlsx', 'xls'])
    
    if excel_file is not None:
        if st.button("output.json 생성"):
            try:
                # 임시 파일로 저장
                with open("temp.xlsx", "wb") as f:
                    f.write(excel_file.getvalue())
                
                # output.json 생성
                json_data = excel_to_json("temp.xlsx")
                
                if json_data:
                    st.success("output.json 파일이 생성되었습니다!")
                    st.session_state.output_json_created = True
                    st.session_state.step = 2
                
                # 임시 파일 삭제
                os.remove("temp.xlsx")
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
    
    # Step 2: Aura URL 입력
    if st.session_state.output_json_created:
        st.header("Step 2: Aura URL 입력")
        aura_url = st.text_input("Aura URL을 입력하세요")
        
        if aura_url and st.button("감사조서 JSON 파일 생성"):
            try:
                engagement_id = extract_engagement_id(aura_url)
                if engagement_id:
                    folder_path = making_excel_path(engagement_id)
                    convert_xlsm_folder_to_json(folder_path)
                    st.success("감사조서 JSON 파일이 생성되었습니다!")
                    st.session_state.json_files_created = True
                    st.session_state.step = 3
                else:
                    st.error("올바른 Aura URL을 입력해주세요.")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
    
    # Step 3: RAG 시스템 초기화 및 질문
    if st.session_state.json_files_created:
        st.header("Step 3: 감사조서 검토")
        
        # RAG 시스템 초기화
        if st.session_state.rag_system is None:
            try:
                st.session_state.rag_system = JsonRAGSystem()
                st.success("RAG 시스템이 초기화되었습니다!")
            except Exception as e:
                st.error(f"RAG 시스템 초기화 중 오류가 발생했습니다: {str(e)}")
                return
        
        # 메타데이터 매핑 정보 표시 옵션
        if st.button("메타데이터 매핑 정보 보기"):
            mapping_info = st.session_state.rag_system.print_metadata_mapping()
            st.markdown(mapping_info)
        
        # 참조할 감사조서 수 설정
        top_k = st.slider("참조할 감사조서 수", min_value=1, max_value=5, value=3)
        
        # 질문 입력 및 답변
        question = st.text_area("질문을 입력하세요")
        if question and st.button("질문하기"):
            response_container = st.empty()
            full_response = ""
            
            # 스트리밍 응답 처리
            for response_chunk in st.session_state.rag_system.get_response(question, top_k):
                full_response += response_chunk
                response_container.markdown(full_response + "▌")
            
            # 최종 응답 표시
            response_container.markdown(full_response)

if __name__ == "__main__":
    main() 