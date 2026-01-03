# rag_engine.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import ollama
import os

# --- 路徑判斷邏輯 ---
if os.path.exists("/datadrive"):
    base_data_path = "/datadrive/glows_data"
else:
    base_data_path = "./data"

persist_db_path = os.path.join(base_data_path, "chroma_db")

# 1. 初始化 Embedding (保留在全域，因為載入權重很重，只需做一次)
print("正在載入 RAG Embedding 模型 (L40S)...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'}
)

# 注意：原本這裡的 vector_db = Chroma(...) 已經被移走！

def ask_glows_bot(query):
    """
    輸入: 使用者問題
    輸出: (回答文字, 分類標籤, 是否成功回答Boolean)
    """
    
    # 2. 關鍵修正：每次問答時才建立連線，確保讀取到最新重建的資料庫
    # 因為 SQLite 是基於檔案的，重新連線的速度非常快，不會造成明顯延遲
    vector_db = Chroma(
        persist_directory=persist_db_path,
        embedding_function=embedding_model
    )
    
    # --- A. 檢索階段 ---
    results = vector_db.similarity_search_with_score(query, k=3)
    
    context_text = ""
    is_answered = True
    
    # 門檻值判斷 (Score 越低越相似)
    if not results or results[0][1] > 1.2:  
        is_answered = False
        context_text = "無相關資料"
    else:
        context_text = "\n\n".join([doc.page_content for doc, score in results])

    # --- B. 生成階段 (Ollama) ---
    if is_answered:
        prompt = f"""
        你是 Glows.ai 的智能助手。請根據以下參考資料回答使用者的問題。
        如果參考資料不足以回答，請說「我目前的資料庫中沒有相關資訊」。

        參考資料：
        {context_text}

        使用者問題：{query}
        """
        
        try:
            response = ollama.chat(model='qwen2.5:14b', messages=[{'role': 'user', 'content': prompt}])
            answer = response['message']['content']
        except Exception as e:
            print(f"Ollama Error: {e}")
            answer = "生成回答時出錯，請稍後再試。"
    else:
        answer = "抱歉，我目前的知識庫中沒有關於此問題的資訊，我已將此問題記錄下來，會儘快請專人補充。"

    # --- C. 自動分類階段 ---
    category = "General"
    q_lower = query.lower()
    
    if any(k in q_lower for k in ["錢", "價格", "費", "租"]):
        category = "Pricing"
    elif any(k in q_lower for k in ["安裝", "cuda", "ssh", "連線", "ip", "error"]):
        category = "Technical"
    elif any(k in q_lower for k in ["帳號", "登入", "密碼"]):
        category = "Account"
    
    if not is_answered:
        category = "Unknown"

    return answer, category, is_answered