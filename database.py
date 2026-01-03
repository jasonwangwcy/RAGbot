# database.py
import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# --- 路徑判斷邏輯 ---
# 檢查主機是否存在 /datadrive 資料夾
if os.path.exists("/datadrive"):
    # 建立一個專屬資料夾存放 DB 檔案，避免直接丟在根目錄太亂
    db_folder = "/datadrive/glows_data"
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)
    DATABASE_URL = f"sqlite:///{db_folder}/chat_logs.db"
    print(f"检测到外部硬碟，資料將儲存於: {DATABASE_URL}")
else:
    # 如果沒看到 /datadrive，就用當前目錄 (開發測試用)
    DATABASE_URL = "sqlite:///./chat_logs.db"
    print("未檢測到外部硬碟，使用本地路徑儲存資料。")

# --- SQLAlchemy 設定 ---
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    question = Column(String)
    answer = Column(String)
    category = Column(String, default="General")
    is_answered = Column(Boolean, default=True)

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("資料庫初始化完成！")