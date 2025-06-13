import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import json
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import io
from sklearn.model_selection import cross_val_score, KFold

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('recommender')

# Create Flask app for API
app = Flask(__name__)
CORS(app)

class VolunteerProjectRecommender:
    def __init__(self):
        self.model = None
        self.skill_vectorizer = None
        self.skill_categories = None
        self.feature_names = None
        self.evaluation_results = {}
        
    def _preprocess_skills(self, skills_list):
        """Convert skills list to TF-IDF vector"""
        if not isinstance(skills_list, list):
            skills_list = []
        skills_text = " ".join(skills_list)
        return self.skill_vectorizer.transform([skills_text]).toarray()[0]
    
    def _calculate_location_features(self, location):
        """Extract location features including coordinates"""
        if not location:
            return np.zeros(2)
        
        try:
            lat = float(location.get('lat') or location.get('latitude', 0))
            lng = float(location.get('lng') or location.get('longitude', 0))
            return np.array([lat, lng])
        except:
            return np.zeros(2)
    
    def _calculate_location_distance(self, loc1, loc2):
        """Calculate approximate distance between two locations"""
        if not loc1 or not loc2:
            return 1000  # Large distance for missing locations
        
        try:
            # Simple Haversine distance calculation
            lat1 = float(loc1.get('lat') or loc1.get('latitude', 0))
            lon1 = float(loc1.get('lng') or loc1.get('longitude', 0))
            lat2 = float(loc2.get('lat') or loc2.get('latitude', 0))
            lon2 = float(loc2.get('lng') or loc2.get('longitude', 0))
            
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371  # Radius of earth in kilometers
            return c * r
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return 1000
    
    def _extract_features(self, volunteer, project):
        """Extract features for volunteer-project pair"""
        # Process volunteer skills
        vol_skills = []
        if 'skills' in volunteer:
            vol_skills = volunteer['skills']
        if 'interest' in volunteer and volunteer['interest']:
            vol_skills.append(volunteer['interest'])
        
        vol_skills_vec = self._preprocess_skills(vol_skills)
        
        # Process project skills
        proj_skills = project.get('skillsReq', [])
        proj_skills_vec = self._preprocess_skills(proj_skills)
        
        # Calculate skill similarity
        skill_sim = cosine_similarity([vol_skills_vec], [proj_skills_vec])[0][0]
        
        # Calculate skill overlap count
        skill_overlap = len(set(vol_skills).intersection(set(proj_skills)))
        
        # Process locations
        vol_loc = self._calculate_location_features(volunteer.get('location'))
        proj_loc = self._calculate_location_features(project.get('location'))
        
        # Calculate distance
        distance = self._calculate_location_distance(
            volunteer.get('location'),
            project.get('location')
        )
        
        # Normalize distance
        norm_distance = np.clip(1 - (distance / 100), 0, 1)  # 0 to 1, closer is higher
        
        # Create feature vector
        features = np.concatenate([
            [skill_sim],  # Skill similarity
            [skill_overlap],  # Number of overlapping skills
            [norm_distance],  # Normalized distance (0-1)
            vol_skills_vec,  # Volunteer skills vector
            proj_skills_vec,  # Project skills vector
        ])
        
        return features
    
    def fit(self, volunteer_data, project_data, interaction_data=None, test_size=0.2):
        """
        Train the recommendation model
        
        Parameters:
        - volunteer_data: list of volunteer dictionaries
        - project_data: list of project dictionaries
        - interaction_data: DataFrame with columns [volunteer_id, project_id, matched]
                            where matched=1 means successful match
        - test_size: proportion of data to use for testing (default: 0.2)
        
        Returns:
        - Dictionary with evaluation metrics if interaction_data is provided
        """
        logger.info(f"Beginning model training with {len(volunteer_data)} volunteers and {len(project_data)} projects")
        
        # Process all unique skills
        all_skills = set()
        for volunteer in volunteer_data:
            all_skills.update(volunteer.get('skills', []))
            if 'interest' in volunteer and volunteer['interest']:
                all_skills.add(volunteer['interest'])
                
        for project in project_data:
            all_skills.update(project.get('skillsReq', []))
        
        self.skill_categories = sorted(list(all_skills))
        logger.info(f"Identified {len(self.skill_categories)} unique skills")
        
        # Create skill vectorizer
        self.skill_vectorizer = TfidfVectorizer(vocabulary=self.skill_categories)
        # Fit on a dummy document with all skills to initialize the vocabulary
        self.skill_vectorizer.fit([" ".join(self.skill_categories)])
        
        # Reset evaluation results
        self.evaluation_results = {}
        
        # If we have interaction data, train the model
        if interaction_data is not None and len(interaction_data) > 0:
            logger.info(f"Training with {len(interaction_data)} interaction records")
            
            # Prepare training data
            X = []
            y = []
            
            vol_dict = {v.get('id', i): v for i, v in enumerate(volunteer_data)}
            proj_dict = {p.get('id', i): p for i, p in enumerate(project_data)}
            
            for _, row in interaction_data.iterrows():
                vol_id = row['volunteer_id']
                proj_id = row['project_id']
                
                if vol_id in vol_dict and proj_id in proj_dict:
                    volunteer = vol_dict[vol_id]
                    project = proj_dict[proj_id]
                    
                    features = self._extract_features(volunteer, project)
                    X.append(features)
                    y.append(float(row['matched']))
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            if len(X) == 0:
                logger.warning("No valid samples found for training")
                return None
            
            # Build feature names for model explainability
            skill_dim = len(self.skill_categories)
            self.feature_names = [
                'skill_similarity', 
                'skill_overlap_count', 
                'location_proximity'
            ] + [f'vol_skill_{s}' for s in self.skill_categories] + [f'proj_skill_{s}' for s in self.skill_categories]
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            logger.info(f"Training RandomForest with {X_train.shape[0]} samples and {X_train.shape[1]} features")
            
            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            eval_results = self.evaluate_model(X_train, y_train, X_test, y_test)
            self.evaluation_results = eval_results
            
            # Cross-validation
            cv_scores = self._perform_cross_validation(X, y, cv=5)
            self.evaluation_results['cross_validation'] = cv_scores
            
            logger.info("Model training and evaluation complete")
            
            return eval_results
        else:
            logger.info("No interaction data provided. Model will use similarity-based recommendations.")
            return None
    
    def _perform_cross_validation(self, X, y, cv=5):
        """Perform cross-validation and return scores"""
        if len(X) < cv:
            logger.warning(f"Not enough samples for {cv}-fold cross-validation. Skipping.")
            return None
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Convert regression problem to classification for CV scoring
        # (since we're primarily interested in binary recommendation outcomes)
        y_binary = (y >= 0.5).astype(int)
        
        # Binary classification metrics
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'rmse': []
        }
        
        for train_idx, test_idx in kf.split(X):
            X_cv_train, X_cv_test = X[train_idx], X[test_idx]
            y_cv_train, y_cv_test = y[train_idx], y[test_idx]
            y_cv_test_binary = y_binary[test_idx]
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_cv_train, y_cv_train)
            
            # Get predictions
            y_pred = model.predict(X_cv_test)
            y_pred_binary = (y_pred >= 0.5).astype(int)
            
            # Calculate metrics
            metrics['accuracy'].append(accuracy_score(y_cv_test_binary, y_pred_binary))
            metrics['precision'].append(precision_score(y_cv_test_binary, y_pred_binary, zero_division=0))
            metrics['recall'].append(recall_score(y_cv_test_binary, y_pred_binary, zero_division=0))
            metrics['f1'].append(f1_score(y_cv_test_binary, y_pred_binary, zero_division=0))
            metrics['rmse'].append(np.sqrt(mean_squared_error(y_cv_test, y_pred)))
        
        # Calculate averages
        cv_results = {
            'accuracy_mean': np.mean(metrics['accuracy']),
            'accuracy_std': np.std(metrics['accuracy']),
            'precision_mean': np.mean(metrics['precision']),
            'precision_std': np.std(metrics['precision']),
            'recall_mean': np.mean(metrics['recall']),
            'recall_std': np.std(metrics['recall']),
            'f1_mean': np.mean(metrics['f1']),
            'f1_std': np.std(metrics['f1']),
            'rmse_mean': np.mean(metrics['rmse']),
            'rmse_std': np.std(metrics['rmse']),
            'fold_scores': metrics
        }
        
        return cv_results
    
    def evaluate_model(self, X_train, y_train, X_test, y_test):
        """Evaluate the model's performance"""
        if self.model is None:
            logger.warning("Model not trained, cannot evaluate")
            return None
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Convert regression values to binary for classification metrics
        threshold = 0.5
        y_train_binary = (y_train >= threshold).astype(int)
        y_test_binary = (y_test >= threshold).astype(int)
        y_train_pred_binary = (y_train_pred >= threshold).astype(int)
        y_test_pred_binary = (y_test_pred >= threshold).astype(int)
        
        # Calculate regression metrics
        regression_metrics = {
            'train': {
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'mae': mean_absolute_error(y_train, y_train_pred),
                'r2': r2_score(y_train, y_train_pred)
            },
            'test': {
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'mae': mean_absolute_error(y_test, y_test_pred),
                'r2': r2_score(y_test, y_test_pred)
            }
        }
        
        # Calculate classification metrics
        try:
            classification_metrics = {
                'train': {
                    'accuracy': accuracy_score(y_train_binary, y_train_pred_binary),
                    'precision': precision_score(y_train_binary, y_train_pred_binary, zero_division=0),
                    'recall': recall_score(y_train_binary, y_train_pred_binary, zero_division=0),
                    'f1': f1_score(y_train_binary, y_train_pred_binary, zero_division=0),
                    'auc': roc_auc_score(y_train_binary, y_train_pred) if len(np.unique(y_train_binary)) > 1 else None
                },
                'test': {
                    'accuracy': accuracy_score(y_test_binary, y_test_pred_binary),
                    'precision': precision_score(y_test_binary, y_test_pred_binary, zero_division=0),
                    'recall': recall_score(y_test_binary, y_test_pred_binary, zero_division=0),
                    'f1': f1_score(y_test_binary, y_test_pred_binary, zero_division=0),
                    'auc': roc_auc_score(y_test_binary, y_test_pred) if len(np.unique(y_test_binary)) > 1 else None
                }
            }
            
            # Calculate ROC curve data
            fpr, tpr, _ = roc_curve(y_test_binary, y_test_pred)
            roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
            classification_metrics = {
                'train': {'error': str(e)},
                'test': {'error': str(e)}
            }
            roc_data = None
        
        # Calculate feature importance
        feature_importance = None
        if self.model is not None and hasattr(self.model, 'feature_importances_') and self.feature_names:
            importance = self.model.feature_importances_
            # Sort features by importance
            indices = np.argsort(importance)[::-1]
            
            # Create feature importance data
            feature_importance = [{
                'feature': self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}',
                'importance': float(importance[i])
            } for i in indices]
        
        # Store evaluation results
        eval_results = {
            'regression_metrics': regression_metrics,
            'classification_metrics': classification_metrics,
            'roc_data': roc_data,
            'feature_importance': feature_importance,
            'threshold': threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        return eval_results
    
    def generate_accuracy_graphs(self):
        """Generate accuracy evaluation graphs"""
        if not self.evaluation_results:
            logger.warning("No evaluation results available to plot")
            return None
            
        # Create a figure with multiple subplots
        plt.figure(figsize=(20, 15))
        
        # 1. Classification metrics comparison
        plt.subplot(2, 2, 1)
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        train_values = [self.evaluation_results['classification_metrics']['train'].get(m, 0) for m in metrics]
        test_values = [self.evaluation_results['classification_metrics']['test'].get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, train_values, width, label='Train')
        plt.bar(x + width/2, test_values, width, label='Test')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Classification Metrics Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 2. ROC Curve
        plt.subplot(2, 2, 2)
        if self.evaluation_results.get('roc_data'):
            fpr = self.evaluation_results['roc_data']['fpr']
            tpr = self.evaluation_results['roc_data']['tpr']
            plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {self.evaluation_results["classification_metrics"]["test"].get("auc", 0):.3f})')
            plt.plot([0, 1], [0, 1], 'r--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, 'ROC data not available', ha='center', va='center')
        
        # 3. Regression metrics
        plt.subplot(2, 2, 3)
        reg_metrics = ['rmse', 'mae']
        train_reg_values = [self.evaluation_results['regression_metrics']['train'].get(m, 0) for m in reg_metrics]
        test_reg_values = [self.evaluation_results['regression_metrics']['test'].get(m, 0) for m in reg_metrics]
        
        x = np.arange(len(reg_metrics))
        
        plt.bar(x - width/2, train_reg_values, width, label='Train')
        plt.bar(x + width/2, test_reg_values, width, label='Test')
        
        plt.xlabel('Metrics')
        plt.ylabel('Error')
        plt.title('Regression Error Metrics')
        plt.xticks(x, reg_metrics)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 4. Feature Importance
        plt.subplot(2, 2, 4)
        if self.evaluation_results.get('feature_importance'):
            # Get top 10 features
            top_features = self.evaluation_results['feature_importance'][:10]
            features = [f['feature'] for f in top_features]
            importance = [f['importance'] for f in top_features]
            
            # Sort for horizontal bar chart
            sorted_indices = np.argsort(importance)
            features = [features[i] for i in sorted_indices]
            importance = [importance[i] for i in sorted_indices]
            
            plt.barh(features, importance)
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.grid(True, linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, 'Feature importance data not available', ha='center', va='center')
        
        # 5. If cross-validation results available, add them
        if 'cross_validation' in self.evaluation_results and self.evaluation_results['cross_validation']:
            plt.figure(figsize=(15, 8))
            cv_metrics = ['accuracy', 'precision', 'recall', 'f1', 'rmse']
            cv_results = self.evaluation_results['cross_validation']
            
            means = [cv_results.get(f'{m}_mean', 0) for m in cv_metrics]
            stds = [cv_results.get(f'{m}_std', 0) for m in cv_metrics]
            
            x = np.arange(len(cv_metrics))
            
            plt.bar(x, means, yerr=stds, align='center', alpha=0.7, capsize=10)
            plt.xlabel('Metrics')
            plt.ylabel('Score / Error')
            plt.title('Cross-Validation Results (5-fold)')
            plt.xticks(x, cv_metrics)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save this as a separate image
            cv_img = io.BytesIO()
            plt.savefig(cv_img, format='png', bbox_inches='tight')
            cv_img.seek(0)
            plt.close()
        
        # Save the main figure to a bytes buffer
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        plt.close()
        
        return img_buf
    
    def get_recommendations(self, volunteer, projects, top_n=10):
        """
        Get project recommendations for a volunteer
        
        If model is not trained with interactions, fall back to similarity calculations
        """
        logger.info(f"Getting recommendations for volunteer {volunteer.get('name')} from {len(projects)} projects")
        
        # Calculate scores for each project
        projects_data = []
        
        # Define what constitutes a "perfect match" threshold
        # Perfect match requires high skill overlap and close proximity
        PERFECT_MATCH_SKILL_THRESHOLD = 0.7  # 70% skill similarity
        PERFECT_MATCH_DISTANCE_THRESHOLD = 10  # 10 kilometers
        
        for project in projects:
            # Extract features
            features = self._extract_features(volunteer, project)
            
            # If we have a trained model, use it
            if self.model is not None:
                # Make prediction
                match_score = float(self.model.predict([features])[0])
                
                # Ensure score is between 0 and 1
                match_score = max(0, min(1, match_score))
            else:
                # Fallback to similarity calculation if no trained model
                skill_sim = features[0]  # Skill similarity from TF-IDF
                skill_overlap = features[1]  # Number of overlapping skills
                loc_score = features[2]  # Normalized distance (0-1, closer is higher)
                
                # Calculate distance in km
                distance = self._calculate_location_distance(
                    volunteer.get('location'),
                    project.get('location')
                )
                
                # Check if this is a perfect match (high skill similarity + close proximity)
                is_perfect_match = (skill_sim >= PERFECT_MATCH_SKILL_THRESHOLD and 
                                distance <= PERFECT_MATCH_DISTANCE_THRESHOLD)
                
                if is_perfect_match:
                    # Perfect matches get a score of 1.0 + small bonus for even better matches
                    # This ensures they always appear first
                    match_score = 1.0 + (skill_sim - PERFECT_MATCH_SKILL_THRESHOLD) + (1.0 - (distance / PERFECT_MATCH_DISTANCE_THRESHOLD)) * 0.1
                else:
                    # For non-perfect matches:
                    # 60% weight to skill match (with internal weighting of similarity and overlap)
                    # 40% weight to location proximity
                    skill_component = 0.6 * (0.7 * skill_sim + 0.3 * min(1, skill_overlap/3))
                    location_component = 0.4 * loc_score
                    match_score = skill_component + location_component
            
            # Calculate additional metrics for explanation
            skill_overlap_set = set(volunteer.get('skills', []))
            if 'interest' in volunteer and volunteer['interest']:
                skill_overlap_set.add(volunteer['interest'])
            skill_overlap_set = skill_overlap_set.intersection(set(project.get('skillsReq', [])))
            
            distance = self._calculate_location_distance(
                volunteer.get('location'),
                project.get('location')
            )
            
            # Store recommendation data
            projects_data.append({
                'project': project,
                'project_id': project.get('id', ''),  # Ensure project ID is included
                'score': match_score,
                'explanation': {
                    'matching_skills': list(skill_overlap_set),
                    'distance_km': round(distance, 2),
                    'skill_match_percent': round(features[0] * 100),
                    'location_match_percent': round(features[2] * 100),
                    'is_perfect_match': 1 if (self.model is None and match_score > 1.0) else 0 if self.model is None else None
                }
            })
        
        # Sort by score and return top N
        sorted_projects = sorted(projects_data, key=lambda x: x['score'], reverse=True)
        logger.info(f"Returning {min(top_n, len(sorted_projects))} recommendations")
        return sorted_projects[:top_n]
    
    def save(self, folder_path):
        """Save the model and preprocessing components"""
        os.makedirs(folder_path, exist_ok=True)
        
        # Save model
        if self.model:
            with open(os.path.join(folder_path, 'model.pkl'), 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {folder_path}/model.pkl")
        
        # Save vectorizer and other components
        if self.skill_vectorizer:
            with open(os.path.join(folder_path, 'skill_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.skill_vectorizer, f)
        
        # Save skill categories
        if self.skill_categories:
            with open(os.path.join(folder_path, 'skill_categories.json'), 'w') as f:
                json.dump(self.skill_categories, f)
        
        # Save feature names
        if self.feature_names:
            with open(os.path.join(folder_path, 'feature_names.json'), 'w') as f:
                json.dump(self.feature_names, f)
        
        # Save evaluation results
        if self.evaluation_results:
            with open(os.path.join(folder_path, 'evaluation_results.json'), 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                results_copy = self.evaluation_results.copy()
                # Handle numpy values for JSON serialization
                if 'roc_data' in results_copy and results_copy['roc_data']:
                    results_copy['roc_data']['fpr'] = [float(x) for x in results_copy['roc_data']['fpr']]
                    results_copy['roc_data']['tpr'] = [float(x) for x in results_copy['roc_data']['tpr']]
                
                json.dump(results_copy, f, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
            logger.info(f"Evaluation results saved to {folder_path}/evaluation_results.json")
            
            # Save accuracy graphs
            try:
                img_buf = self.generate_accuracy_graphs()
                if img_buf:
                    with open(os.path.join(folder_path, 'accuracy_graphs.png'), 'wb') as f:
                        f.write(img_buf.getvalue())
                    logger.info(f"Accuracy graphs saved to {folder_path}/accuracy_graphs.png")
            except Exception as e:
                logger.error(f"Error saving accuracy graphs: {e}")
        
        logger.info(f"All model components saved to {folder_path}")
    
    def load(self, folder_path):
        """Load the model and preprocessing components"""
        # Load model
        model_path = os.path.join(folder_path, 'model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {model_path}")
        
        # Load vectorizer
        vectorizer_path = os.path.join(folder_path, 'skill_vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                self.skill_vectorizer = pickle.load(f)
            logger.info(f"Skill vectorizer loaded from {vectorizer_path}")
        
        # Load skill categories
        categories_path = os.path.join(folder_path, 'skill_categories.json')
        if os.path.exists(categories_path):
            with open(categories_path, 'r') as f:
                self.skill_categories = json.load(f)
            logger.info(f"Loaded {len(self.skill_categories)} skill categories")
        
        # Load feature names
        feature_names_path = os.path.join(folder_path, 'feature_names.json')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
        
        # Load evaluation results
        eval_results_path = os.path.join(folder_path, 'evaluation_results.json')
        if os.path.exists(eval_results_path):
            with open(eval_results_path, 'r') as f:
                self.evaluation_results = json.load(f)
            logger.info(f"Evaluation results loaded from {eval_results_path}")


# Initialize the recommender
recommender = VolunteerProjectRecommender()

# API endpoints
@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        volunteers = data.get('volunteers', [])
        projects = data.get('projects', [])
        interactions = data.get('interactions', [])
        test_size = float(data.get('test_size', 0.2))
        
        # Convert interactions to DataFrame if provided
        interactions_df = None
        if interactions:
            interactions_df = pd.DataFrame(interactions)
        
        # Train the model
        eval_results = recommender.fit(volunteers, projects, interactions_df, test_size=test_size)
        
        # Save the model
        recommender.save('./model')
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'model_info': {
                'num_volunteers': len(volunteers),
                'num_projects': len(projects),
                'num_skills': len(recommender.skill_categories) if recommender.skill_categories else 0,
                'timestamp': datetime.now().isoformat()
            },
            'evaluation': eval_results
        })
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@app.route('/api/evaluate', methods=['GET'])
def get_evaluation():
    """Return the latest evaluation results"""
    try:
        if not recommender.evaluation_results:
            # Try to load from file
            eval_path = './model/evaluation_results.json'
            if os.path.exists(eval_path):
                with open(eval_path, 'r') as f:
                    evaluation_results = json.load(f)
            else:
                return jsonify({
                    'success': False,
                    'error': 'No evaluation results available'
                }), 404
        else:
            evaluation_results = recommender.evaluation_results
        
        return jsonify({
            'success': True,
            'evaluation': evaluation_results
        })
    except Exception as e:
        logger.error(f"Evaluation retrieval error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/accuracy/graph', methods=['GET'])
def get_accuracy_graph():
    """Generate and return accuracy graphs as an image"""
    try:
        # Check if we have evaluation results
        if not recommender.evaluation_results:
            # Try to load from file
            eval_path = './model/evaluation_results.json'
            if os.path.exists(eval_path):
                with open(eval_path, 'r') as f:
                    recommender.evaluation_results = json.load(f)
            else:
                return jsonify({
                    'success': False,
                    'error': 'No evaluation results available to plot'
                }), 404
        
        # Check if the graph is already saved
        graph_path = './model/accuracy_graphs.png'
        if os.path.exists(graph_path):
            return send_file(graph_path, mimetype='image/png')
        
        # Generate new graph
        img_buf = recommender.generate_accuracy_graphs()
        if img_buf:
            # Save for future use
            with open(graph_path, 'wb') as f:
                f.write(img_buf.getvalue())
            
            return send_file(img_buf, mimetype='image/png')
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate accuracy graphs'
            }), 500
    except Exception as e:
        logger.error(f"Graph generation error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/accuracy/metrics', methods=['GET'])
def get_accuracy_metrics():
    """Return specific accuracy metrics"""
    try:
        metric_type = request.args.get('type', 'classification')  # Options: classification, regression, cv
        
        if not recommender.evaluation_results:
            # Try to load from file
            eval_path = './model/evaluation_results.json'
            if os.path.exists(eval_path):
                with open(eval_path, 'r') as f:
                    evaluation_results = json.load(f)
            else:
                return jsonify({
                    'success': False,
                    'error': 'No evaluation results available'
                }), 404
        else:
            evaluation_results = recommender.evaluation_results
        
        if metric_type == 'classification' and 'classification_metrics' in evaluation_results:
            return jsonify({
                'success': True,
                'metrics': evaluation_results['classification_metrics']
            })
        elif metric_type == 'regression' and 'regression_metrics' in evaluation_results:
            return jsonify({
                'success': True,
                'metrics': evaluation_results['regression_metrics']
            })
        elif metric_type == 'cv' and 'cross_validation' in evaluation_results:
            return jsonify({
                'success': True,
                'metrics': evaluation_results['cross_validation']
            })
        elif metric_type == 'feature_importance' and 'feature_importance' in evaluation_results:
            return jsonify({
                'success': True,
                'feature_importance': evaluation_results['feature_importance']
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Metric type {metric_type} not available'
            }), 404
    except Exception as e:
        logger.error(f"Metrics retrieval error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/custom_graph', methods=['POST'])
def generate_custom_graph():
    """Generate a custom graph based on the evaluation results"""
    try:
        data = request.json
        metrics = data.get('metrics', ['accuracy', 'precision', 'recall', 'f1'])
        graph_type = data.get('type', 'bar')  # bar, line, radar, etc.
        title = data.get('title', 'Model Performance Metrics')
        
        if not recommender.evaluation_results:
            # Try to load from file
            eval_path = './model/evaluation_results.json'
            if os.path.exists(eval_path):
                with open(eval_path, 'r') as f:
                    recommender.evaluation_results = json.load(f)
            else:
                return jsonify({
                    'success': False,
                    'error': 'No evaluation results available'
                }), 404
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        if graph_type == 'bar':
            # Get train and test values for selected metrics
            train_values = []
            test_values = []
            
            for metric in metrics:
                train_val = recommender.evaluation_results['classification_metrics']['train'].get(metric, 0)
                test_val = recommender.evaluation_results['classification_metrics']['test'].get(metric, 0)
                train_values.append(train_val)
                test_values.append(test_val)
            
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, train_values, width, label='Train')
            plt.bar(x + width/2, test_values, width, label='Test')
            
            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.title(title)
            plt.xticks(x, metrics)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
        elif graph_type == 'line':
            # For line graph, use cross-validation results if available
            if 'cross_validation' in recommender.evaluation_results and recommender.evaluation_results['cross_validation']:
                cv_results = recommender.evaluation_results['cross_validation']
                fold_scores = cv_results.get('fold_scores', {})
                
                # If we have fold scores
                if fold_scores:
                    for metric in metrics:
                        if metric in fold_scores:
                            plt.plot(range(1, len(fold_scores[metric])+1), fold_scores[metric], 
                                   marker='o', label=metric)
                    
                    plt.xlabel('Fold')
                    plt.ylabel('Score')
                    plt.title(f'Cross-Validation Scores: {title}')
                    plt.xticks(range(1, len(next(iter(fold_scores.values()), []))+1))
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                else:
                    plt.text(0.5, 0.5, 'Cross-validation fold scores not available', 
                           ha='center', va='center')
            else:
                plt.text(0.5, 0.5, 'Cross-validation results not available', 
                       ha='center', va='center')
                
        elif graph_type == 'roc':
            if 'roc_data' in recommender.evaluation_results and recommender.evaluation_results['roc_data']:
                fpr = recommender.evaluation_results['roc_data']['fpr']
                tpr = recommender.evaluation_results['roc_data']['tpr']
                auc = recommender.evaluation_results['classification_metrics']['test'].get('auc', 0)
                
                plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc:.3f})')
                plt.plot([0, 1], [0, 1], 'r--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
            else:
                plt.text(0.5, 0.5, 'ROC data not available', ha='center', va='center')
        
        elif graph_type == 'feature_importance':
            if 'feature_importance' in recommender.evaluation_results:
                # Get top features (limit to 15 or fewer)
                n_features = min(15, len(recommender.evaluation_results['feature_importance']))
                top_features = recommender.evaluation_results['feature_importance'][:n_features]
                
                # Extract names and values, and sort for horizontal bar chart
                features = [f['feature'] for f in top_features]
                importance = [f['importance'] for f in top_features]
                
                # Sort for horizontal bar chart
                sorted_indices = np.argsort(importance)
                features = [features[i] for i in sorted_indices]
                importance = [importance[i] for i in sorted_indices]
                
                plt.barh(features, importance)
                plt.xlabel('Importance')
                plt.title('Feature Importance')
                plt.grid(True, linestyle='--', alpha=0.7)
            else:
                plt.text(0.5, 0.5, 'Feature importance data not available', 
                       ha='center', va='center')
        
        # Save the figure to a bytes buffer
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        plt.close()
        
        return send_file(img_buf, mimetype='image/png')
    except Exception as e:
        logger.error(f"Custom graph generation error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.json
        volunteer = data.get('volunteer', {})
        projects = data.get('projects', [])
        top_n = data.get('top_n', 10)
        
        # Try to load the model if it exists
        model_dir = './model'
        if not recommender.skill_vectorizer and os.path.exists(f'{model_dir}/skill_vectorizer.pkl'):
            recommender.load(model_dir)
        
        # If model still not loaded (no saved model), initialize with available data
        if not recommender.skill_vectorizer:
            all_skills = set()
            if 'skills' in volunteer:
                all_skills.update(volunteer.get('skills', []))
            if 'interest' in volunteer and volunteer['interest']:
                all_skills.add(volunteer['interest'])
            
            for project in projects:
                all_skills.update(project.get('skillsReq', []))
            
            recommender.skill_categories = sorted(list(all_skills))
            recommender.skill_vectorizer = TfidfVectorizer(vocabulary=recommender.skill_categories)
            recommender.skill_vectorizer.fit([" ".join(recommender.skill_categories)])
        
        # Get recommendations
        recommendations = recommender.get_recommendations(volunteer, projects, top_n)
        
        # Ensure all values in the JSON are serializable - convert booleans to int if needed
        for rec in recommendations:
            if 'explanation' in rec and 'is_perfect_match' in rec['explanation']:
                # Convert None to null (JSON-friendly) and booleans to integers
                if rec['explanation']['is_perfect_match'] is None:
                    rec['explanation']['is_perfect_match'] = None
                elif isinstance(rec['explanation']['is_perfect_match'], bool):
                    rec['explanation']['is_perfect_match'] = 1 if rec['explanation']['is_perfect_match'] else 0
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'metadata': {
                'total_projects': len(projects),
                'returned_results': len(recommendations),
                'model_status': 'trained' if recommender.model else 'similarity-based',
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'status': 'active',
        'model_loaded': recommender.model is not None,
        'skills_known': len(recommender.skill_categories) if recommender.skill_categories else 0,
        'evaluation_available': bool(recommender.evaluation_results),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Try to load existing model
    model_dir = './model'
    if os.path.exists(f'{model_dir}/model.pkl'):
        try:
            recommender.load(model_dir)
            logger.info("Loaded existing model")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    logger.info("Starting recommendation API server...")
    app.run(host='0.0.0.0', port=3001)