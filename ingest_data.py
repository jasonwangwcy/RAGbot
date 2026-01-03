# ingest_data.py
import json
import os
import shutil
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def run_ingestion():
    # --- 路徑判斷邏輯 ---
    # 依照你的預設，如果看到 /datadrive 就用它，否則用本地目錄
    if os.path.exists("/datadrive"):
        base_data_path = "/datadrive/glows_data"
    else:
        base_data_path = "./data"
    
    # 定義 ChromaDB 儲存的完整路徑
    persist_db_path = os.path.join(base_data_path, "chroma_db")

    # 0. 清理舊資料庫
    if os.path.exists(persist_db_path):
        try:
            # 加上 ignore_errors=True 可以跳過一些暫時性的鎖定問題
            shutil.rmtree(persist_db_path, ignore_errors=True)
            
            # 如果還是刪不掉，我們可以嘗試第二次刪除 (針對掛載硬碟的延遲)
            if os.path.exists(persist_db_path):
                import time
                time.sleep(1) # 稍微等一下檔案系統釋放
                shutil.rmtree(persist_db_path, ignore_errors=True)
                
            print(f"已清除舊的向量資料庫: {persist_db_path}")
        except Exception as e:
            print(f"清理資料夾時發生非預期錯誤: {e}")
    # 確保基礎資料夾存在
    if not os.path.exists(base_data_path):
        os.makedirs(base_data_path)

    # 1. 初始化 Embedding 模型 (使用 BAAI/bge-m3)
    print("正在載入 Embedding 模型...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda'} # L40S 必備
    )

    # 2. 讀取 JSON (假設 123.json 跟你的程式碼放在一起)
    json_path = "/datadrive/123.json"
    if not os.path.exists(json_path):
        print(f"❌ 找不到 {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    # 3. 轉換為 LangChain Document 格式
    documents = []
    for item in qa_data:
        text_content = f"問題: {item['instruction']}\n回答: {item['output']}"
        doc = Document(
            page_content=text_content,
            metadata={"source": "123.json", "category": "QA"}
        )
        documents.append(doc)

    # 4. 存入 ChromaDB
    print(f"正在匯入 {len(documents)} 筆資料...")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_db_path  # 指向 /datadrive/glows_data/chroma_db
    )
    print(f"✅ 資料匯入成功！儲存於 {persist_db_path}")

if __name__ == "__main__":
    run_ingestion()