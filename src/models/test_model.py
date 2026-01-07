from src.models.full_model import build_foodvision_model

if __name__ == "__main__":
    model = build_foodvision_model()
    model.summary()
