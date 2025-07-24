from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib, json, os, numpy as np, logging
from typing import List, Dict, Any
from utils.feature_extractor import FeatureExtractor

class LocalHeadingModel:
    def __init__(self, model_dir: str = "models/labler"):
        self.model_dir = model_dir
        self.classifier, self.scaler, self.feature_names = None, None, None
        os.makedirs(model_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def train_model(self, training_data_file: str) -> Dict[str, Any]:
        with open(training_data_file, 'r', encoding='utf-8') as f: training_data = json.load(f)
        if not training_data: self.logger.error("No training data"); return {}
        
        feature_extractor = FeatureExtractor()
        features, labels = feature_extractor.extract_features(training_data), [s['label'] for s in training_data]
        if len(features) == 0: self.logger.error("No features extracted"); return {}
        
        self.feature_names, X, y = feature_extractor.get_feature_names(), features, np.array(labels)
        unique_labels = np.unique(y)
        if len(unique_labels) <= 1:
            self.logger.warning(f"Only one class in training data. Model cannot be meaningfully trained."); return {'accuracy': 1.0}

        label_counts = {label: np.sum(y == label) for label in unique_labels}
        if any(count < 2 for count in label_counts.values()):
            self.logger.warning(f"Some classes have only 1 member: {label_counts}. Training without stratification.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.classifier = LGBMClassifier(random_state=42)
        self.classifier.fit(X_train_scaled, y_train)
        
        accuracy = accuracy_score(y_test, self.classifier.predict(self.scaler.transform(X_test)))
        self.logger.info(f"Model trained with accuracy: {accuracy:.3f}")
        self._save_model()
        return {'accuracy': accuracy}
    
    def predict(self, blocks: List[Dict[str, Any]]) -> List[str]:
        if not self.classifier: self.logger.error("Model not loaded."); return ['NONE'] * len(blocks)
        features = FeatureExtractor().extract_features(blocks)
        if len(features) == 0: return ['NONE'] * len(blocks)
        return self.classifier.predict(self.scaler.transform(features)).tolist()
    
    def _save_model(self):
        joblib.dump(self.classifier, os.path.join(self.model_dir, 'heading_classifier.joblib'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'feature_scaler.joblib'))
        with open(os.path.join(self.model_dir, 'feature_names.json'), 'w') as f: json.dump(self.feature_names, f)
        self.logger.info("Model saved successfully")
    
    def load_model(self) -> bool:
        try:
            self.classifier = joblib.load(os.path.join(self.model_dir, 'heading_classifier.joblib'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'feature_scaler.joblib'))
            with open(os.path.join(self.model_dir, 'feature_names.json'), 'r') as f: self.feature_names = json.load(f)
            self.logger.info("Model loaded successfully"); return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}"); return False