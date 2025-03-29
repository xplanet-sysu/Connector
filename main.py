import uvicorn
from fastapi import FastAPI,APIRouter
from api.endpoints import router as api_router

app = FastAPI(title="My FastAPI App")

# 加载三个子接口
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000,reload=True)
