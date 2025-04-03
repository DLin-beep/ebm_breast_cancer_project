from models.train_ebm import train_and_save_model
from evaluation.evaluate_ebm import evaluate_model

if __name__ == "__main__":
    ebm, X_test, y_test, feature_info = train_and_save_model()
    evaluate_model(ebm, X_test, y_test)
