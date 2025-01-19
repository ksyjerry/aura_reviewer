import pyodbc

def get_assignee_report():
    conn_str = (
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=gx-zsesqlp011.database.windows.net;"
        "Database=REPORT_COMMON;"
        "UID=KRAzureCommon;"
        "PWD=a=fh9+@Xw?4RgprbFD2TQ*eUgLB8R7eL;"
        "Connection Timeout=30"
    )
    
    try:
        print("데이터베이스 연결 시도 중...")
        conn = pyodbc.connect(conn_str)
        print("연결 성공!")
        cursor = conn.cursor()
        
        # TOP 10으로 제한하여 빠르게 조회
        query = """
        SELECT TOP 10
            EngagementId,
            EgaName,
            PrepName,
            PrepStatus,
            PrepDueDate,
            CURAssignee
        FROM BI_AURA_ASSIGNEEREPORT WITH (NOLOCK)
        WHERE EngagementId = '014981ed-c5b2-4bdd-b9ca-30c680e9282c'
        AND PrepDueDate IS NOT NULL
        ORDER BY PrepDueDate DESC
        """
        
        print("최근 10개 데이터 조회 중...")
        cursor.execute(query)
        results = cursor.fetchall()
        
        if results:
            columns = [column[0] for column in cursor.description]
            print(f"\n{len(results)}개의 결과가 조회되었습니다.")
            print("\n조회 결과:")
            for row in results:
                print("\n---")
                for i, col in enumerate(columns):
                    print(f"{col}: {row[i]}")
        else:
            print("\n해당 EngagementId의 데이터가 없습니다.")
        
        cursor.close()
        conn.close()
        print("\n연결 종료됨")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    get_assignee_report()
