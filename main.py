# main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
import uvicorn
import os
import json
import subprocess  # <--- 改用這個來執行外部腳本，避開鎖定

# 定義收集新知識的 JSON 檔案路徑
if os.path.exists("/datadrive"):
    # 存放在持久化硬碟中，避免 Container 重啟消失
    COLLECTED_JSON = "/datadrive/glows_data/collected_qa.json"
else:
    COLLECTED_JSON = "./data/collected_qa.json"

# 引入自定義模組
# 這些模組現在都會自動將資料導向 /datadrive/glows_data
from database import SessionLocal, ChatLog, init_db
from rag_engine import ask_glows_bot

# 確保資料庫與資料表已初始化
init_db()

app = FastAPI(title="Glows.ai RAG Backend")

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 資料庫依賴項 ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 定義請求格式 ---
class ChatRequest(BaseModel):
    message: str

class AnswerUpdate(BaseModel):
    log_id: int
    answer: str

# ===========================
# 0. 健康檢查接口
# ===========================
@app.get("/health")
async def health_check():
    return {"status": "online", "message": "Glows.ai RAG API is running"}

# ===========================
# 1. 聊天接口 (Chat API)
# ===========================
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        # 呼叫 RAG 引擎
        answer, category, is_answered = ask_glows_bot(request.message)

        # 寫入資料庫
        new_log = ChatLog(
            question=request.message,
            answer=answer,
            category=category,
            is_answered=is_answered
        )
        db.add(new_log)
        db.commit()

        return {
            "answer": answer, 
            "category": category,
            "is_answered": is_answered
        }
    except Exception as e:
        print(f"Error in /api/chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===========================
# 2. 後台儀表板接口 (Dashboard API)
# ===========================

# A: 每日統計
@app.get("/api/dashboard/daily_stats")
async def get_daily_stats(db: Session = Depends(get_db)):
    logs = db.query(ChatLog).all()
    stats = {}
    for log in logs:
        d_str = log.timestamp.strftime("%Y-%m-%d")
        stats[d_str] = stats.get(d_str, 0) + 1
    return stats

# B: 問題種類分佈
@app.get("/api/dashboard/category_stats")
async def get_category_stats(db: Session = Depends(get_db)):
    results = db.query(ChatLog.category, func.count(ChatLog.id)).group_by(ChatLog.category).all()
    return [{"name": r[0], "value": r[1]} for r in results]

# C: 未回答的問題清單
@app.get("/api/dashboard/missed_questions")
async def get_missed_questions(db: Session = Depends(get_db)):
    results = db.query(ChatLog).filter(ChatLog.is_answered == False).order_by(ChatLog.timestamp.desc()).limit(50).all()
    return [
        {
            "id": r.id,
            "date": r.timestamp.strftime("%Y-%m-%d %H:%M"),
            "question": r.question,
            "category": r.category
        }
        for r in results
    ]

# D: 所有問題清單
@app.get("/api/dashboard/all_questions")
async def get_all_questions(db: Session = Depends(get_db)):
    results = db.query(ChatLog).order_by(ChatLog.timestamp.desc()).all()
    return [
        {
            "id": r.id,
            "date": r.timestamp.strftime("%Y-%m-%d %H:%M"),
            "question": r.question,
            "answer": r.answer,
            "category": r.category,
            "is_answered": r.is_answered
        }
        for r in results
    ]

# E: 手動更新答案接口 (用於人工優化 RAG)
@app.post("/api/dashboard/update_answer")
async def update_answer(data: AnswerUpdate, db: Session = Depends(get_db)):
    # 1. 從資料庫找到該筆紀錄
    log_entry = db.query(ChatLog).filter(ChatLog.id == data.log_id).first()
    if not log_entry:
        raise HTTPException(status_code=404, detail="找不到該筆紀錄")
    
    # 2. 更新 SQLite 資料庫狀態
    log_entry.answer = data.answer
    log_entry.is_answered = True
    log_entry.category = "Manual_Fixed"  # 標記為人工修復
    
    # 3. 將資料寫入 collected_qa.json 檔案
    try:
        new_qa_pair = {
            "instruction": log_entry.question,
            "output": data.answer
        }
        
        existing_data = []
        # 如果檔案已存在，先讀取舊資料
        if os.path.exists(COLLECTED_JSON):
            with open(COLLECTED_JSON, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        
        # 將新問答追加進去
        existing_data.append(new_qa_pair)
        
        # 寫回檔案
        with open(COLLECTED_JSON, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
        print(f"✅ 已同步至收集清單: {COLLECTED_JSON}")
        
    except Exception as e:
        print(f"❌ 寫入 JSON 檔案失敗: {e}")

    db.commit()
    return {"status": "success", "message": "答案已更新並同步至 collected_qa.json"}


# F: RAG 重建接口 (使用子進程解決鎖定問題)
@app.post("/api/dashboard/rebuild_rag")
async def trigger_rebuild(background_tasks: BackgroundTasks):
    """
    前端按下「同步並更新 RAG」按鈕時觸發
    改用 subprocess 執行外部腳本，徹底解決 SQLite 鎖定與 Read-only 問題
    """
    def run_rebuild_script_isolated():
        try:
            print("⏳ 開始執行 RAG 重建子進程...")
            # 這行指令等同於你在終端機打 "python rebuild_rag.py"
            result = subprocess.run(
                ["python", "rebuild_rag.py"], 
                cwd="/backend",  # 確保在正確的專案目錄執行
                capture_output=True,
                text=True
            )
            
            # 記錄詳細日誌
            log_path = "/datadrive/glows_data/rebuild.log"
            with open(log_path, "w") as f:
                f.write(f"--- STDOUT ---\n{result.stdout}\n")
                f.write(f"--- STDERR ---\n{result.stderr}\n")

            if result.returncode == 0:
                print("✅ 背景子進程：RAG 重建成功 (Process Isolated)")
            else:
                print(f"❌ 背景子進程：RAG 重建失敗，請查看 {log_path}")
                print(result.stderr) # 也印在終端機方便除錯
                
        except Exception as e:
            print(f"❌ 呼叫子進程失敗: {e}")

    # 將任務丟進後台
    background_tasks.add_task(run_rebuild_script_isolated)

    # 立即回傳給前端
    return {
        "status": "success", 
        "message": "系統已啟動獨立進程重建 RAG，請稍候幾分鐘再查看結果。"
    }
@app.get("/api/dashboard/qa_library")
async def get_qa_library():
    """
    從持久化硬碟讀取最新的知識庫檔案 123.json
    """
    target_path = "/datadrive/123.json" 
    
    if os.path.exists(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"讀取 JSON 失敗: {e}")
    else:
        return [] # 如果檔案還沒建立，回傳空陣列

# ===========================
# 啟動伺服器
# ===========================
if __name__ == "__main__":
    # 你原本的設定 port=1111
    uvicorn.run(app, host="0.0.0.0", port=1111)