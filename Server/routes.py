import os
import shutil
import traceback
import uuid

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import SessionLocal
from models import ObjFile  # You may want to rename this to ModelFile
from schemas import ObjFileCreate, ObjFileResponse  # You may want to rename these
import ml_inference

router = APIRouter()

UPLOAD_DIR = "uploads"
OUTPUT_GLB_DIR = "output_glbs"  # Changed from OUTPUT_OBJ_DIR

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_GLB_DIR, exist_ok=True)  # Changed directory name

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/models/")
def list_models(user_id: str, db: Session = Depends(get_db)):
    db_models = db.query(ObjFile).filter(ObjFile.user_id == user_id).all()
    model_list = []
    for db_model in db_models:
        # Assuming db_model.filepath is like "output_glbs/xxxxxx.glb"
        filename = os.path.basename(db_model.filepath)
        model_list.append({
            "id": db_model.id,
            "title": filename.replace('_', ' ').title(),
            "glb_url": f"/models/{filename}"
        })
    return model_list

@router.get("/models/{filename}")
def get_model_file(filename: str):
    file_path = os.path.join(OUTPUT_GLB_DIR, filename)  # Changed directory
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            headers={
                "Content-Type": "model/gltf-binary",  # Changed MIME type for GLB
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")

@router.post("/upload-images/", response_model=ObjFileResponse)
async def upload_images(
    user_id: str = Form(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    # Create unique folder for user upload batch
    upload_folder = os.path.join(UPLOAD_DIR, str(uuid.uuid4()))
    os.makedirs(upload_folder, exist_ok=True)
    
    # Save uploaded files
    for file in files:
        file_path = os.path.join(upload_folder, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
    # Prepare output GLB path
    glb_filename = f"{uuid.uuid4()}.glb"  # Changed extension
    output_glb_path = os.path.join(OUTPUT_GLB_DIR, glb_filename)  # Changed path
    
    try:
        # Run inference and save GLB
        ml_inference.load_models()  # Make sure models are loaded
        glb_path = ml_inference.run_inference_and_save_glb(upload_folder, output_glb_path)  # Changed function name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    
    # Save to DB
    db_glbfile = ObjFile(user_id=user_id, filepath=glb_path)  # Consider renaming ObjFile model
    db.add(db_glbfile)
    db.commit()
    db.refresh(db_glbfile)
    
    return db_glbfile 