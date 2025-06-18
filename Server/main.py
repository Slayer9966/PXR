from fastapi import FastAPI
from database import Base, engine
from routes import router as objfile_router
import ml_inference
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your frontend's address for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Base.metadata.create_all(bind=engine)

app.include_router(objfile_router, prefix="/objfiles", tags=["objfiles"])

@app.on_event("startup")
async def startup_event():
    ml_inference.load_models()

@app.get("/")
async def root():
    return {"message": "API is running"}
