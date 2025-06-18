from database import engine, Base
from models import ReconstructionResult

Base.metadata.create_all(bind=engine)
print("Tables created successfully!")
