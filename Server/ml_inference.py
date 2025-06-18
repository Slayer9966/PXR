import torch
import os
import numpy as np
from PIL import Image
import glob
from skimage import measure
import trimesh

# Import your model definitions from ml_models folder
from ml_models.encoder import Encoder
from ml_models.decoder import Decoder
from ml_models.merger import Merger
from ml_models.refiner import Refiner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_VIEWS = 13
IMAGE_SIZE = 256

encoder = None
decoder = None
merger = None
refiner = None

def load_models():
    global encoder, decoder, merger, refiner
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    merger = Merger().to(device)
    refiner = Refiner().to(device)
    
    checkpoint = torch.load('ml_models/fixed_best_model.pth', map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    merger.load_state_dict(checkpoint['merger'])
    refiner.load_state_dict(checkpoint['refiner'])
    
    encoder.train()
    decoder.train()
    merger.train()
    refiner.train()

def load_png_images_from_directory(directory):
    image_paths = sorted(glob.glob(os.path.join(directory, "*.png")))
    if len(image_paths) == 0:
        raise ValueError("No PNG images found in directory")
    
    selected_paths = image_paths[:NUM_VIEWS]
    images = []
    for img_path in selected_paths:
        img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        images.append(img)
    while len(images) < NUM_VIEWS:
        images.append(images[-1].clone())
    images = torch.stack(images, dim=0)
    return images

def save_voxels_to_glb(voxel_grid, filepath='output.glb', threshold=0.5):
    """
    Convert voxel grid to GLB format using marching cubes and trimesh
    """
    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.squeeze().cpu().numpy()
    
    # Extract mesh using marching cubes
    verts, faces, normals, _ = measure.marching_cubes(voxel_grid, level=threshold)
    
    # Create a Trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # Optional: Apply smoothing for better quality
    mesh = mesh.smoothed()
    
    # Optional: Fix mesh issues
    mesh.fix_normals()
    mesh.fill_holes()
    
    # Export to GLB format
    mesh.export(filepath, file_type='glb')
    
    print(f"GLB model saved to: {filepath}")
    return filepath

# Keep the old OBJ function commented for reference
# def save_voxels_to_obj(voxel_grid, filepath='output.obj', threshold=0.5):
#     if isinstance(voxel_grid, torch.Tensor):
#         voxel_grid = voxel_grid.squeeze().cpu().numpy()
#     verts, faces, normals, _ = measure.marching_cubes(voxel_grid, level=threshold)
#     with open(filepath, 'w') as f:
#         for v in verts:
#             f.write(f"v {v[0]} {v[1]} {v[2]}\n")
#         for face in faces + 1:
#             f.write(f"f {face[0]} {face[1]} {face[2]}\n")

def run_inference_and_save_glb(image_dir, output_glb_path):
    """
    Run inference on images and save the result as a GLB file
    
    Args:
        image_dir (str): Directory containing input PNG images
        output_glb_path (str): Path where the GLB file will be saved
        
    Returns:
        str: Path to the saved GLB file
    """
    # Load and preprocess images
    images = load_png_images_from_directory(image_dir)
    images = images.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        features = encoder(images)
        raw_features, gen_volumes = decoder(features)
        merged = merger(raw_features, gen_volumes)
        merged_sigmoid = torch.sigmoid(merged)
        refined_logits = refiner(merged_sigmoid)
        refined_sigmoid = torch.sigmoid(refined_logits)
        
        # Convert to binary voxels
        pred_voxels = (refined_sigmoid > 0.5).float().squeeze(0)
        
        # Save as GLB
        save_voxels_to_glb(pred_voxels, filepath=output_glb_path)
    
    return output_glb_path

# Deprecated: Keep for backward compatibility if needed
def run_inference_and_save_obj(image_dir, output_obj_path):
    """
    DEPRECATED: Use run_inference_and_save_glb instead
    This function is kept for backward compatibility
    """
    print("Warning: run_inference_and_save_obj is deprecated. Use run_inference_and_save_glb instead.")
    # Convert .obj extension to .glb
    glb_path = output_obj_path.replace('.obj', '.glb')
    return run_inference_and_save_glb(image_dir, glb_path)