import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
import time
import warnings
import joblib
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import learning_curve
warnings.filterwarnings('ignore')

class DataGenerator:
    def __init__(self):
        self.features = None
        self.feature_configs = None
        self.classes = None
        self.class_configs = None
        
    def generate_synthetic_data(self, n_samples, feature_configs, classes, class_configs=None):
        """Generate synthetic data based on configurations"""
        n_features = len(feature_configs)
        n_classes = len(classes)
        
        X = []
        y = []
        samples_per_class = n_samples // n_classes
        
        for i in range(n_classes):
            class_samples = []
            class_name = classes[i]
            
            for j, (feature_name, config) in enumerate(feature_configs.items()):
                if class_configs and class_name in class_configs:
                    center = class_configs[class_name]['mean'][j]
                    std = class_configs[class_name]['std'][j]
                else:
                    if config['type'] == 'random':
                        center = np.random.randn() * 5
                        std = config['std']
                    else:
                        center = config['center']
                        std = config['std']
                
                feature_samples = np.round(np.random.normal(
                    loc=center,
                    scale=std,
                    size=samples_per_class
                ), decimals=2)
                class_samples.append(feature_samples)
            
            X.append(np.column_stack(class_samples))
            y.extend([classes[i]] * samples_per_class)
        
        X = np.vstack(X)
        return X, np.array(y)

class ModelManager:
    @staticmethod
    def get_classifiers():
        """Return dictionary of classifiers with appropriate preprocessing"""
        return {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=1000),
                'scaler': StandardScaler()
            },
            'RidgeClassifier': {
                'model': RidgeClassifier(),
                'scaler': StandardScaler()
            },
            'RandomForestClassifier': {
                'model': RandomForestClassifier(random_state=42),
                'scaler': StandardScaler()
            },
            'AdaBoostClassifier': {
                'model': AdaBoostClassifier(),
                'scaler': StandardScaler()
            },
            'ExtraTreesClassifier': {
                'model': ExtraTreesClassifier(),
                'scaler': StandardScaler()
            },
            'SVC': {
                'model': SVC(),
                'scaler': StandardScaler()
            },
            'LinearSVC': {
                'model': LinearSVC(max_iter=2000),
                'scaler': StandardScaler()
            },
            'GaussianNB': {
                'model': GaussianNB(),
                'scaler': StandardScaler()
            },
            'KNeighborsClassifier': {
                'model': KNeighborsClassifier(),
                'scaler': StandardScaler()
            },
            'MLPClassifier': {
                'model': MLPClassifier(max_iter=1000),
                'scaler': StandardScaler()
            },
            'MultinomialNB': {
                'model': MultinomialNB(),
                'scaler': MaxAbsScaler()
            }
        }

    @staticmethod
    def ensure_non_negative(X):
        """Ensure data is non-negative by shifting"""
        if isinstance(X, pd.DataFrame):
            min_val = X.values.min()
            if min_val < 0:
                return X + abs(min_val)
            return X
        else:
            min_val = X.min()
            if min_val < 0:
                return X - min_val
            return X

    def save_model(self, model_dict, model_name):
        """Save model and its scaler to files"""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        base_filename = f"{model_name}"
        
        if hasattr(model_dict['model'], 'feature_names_in_'):
            model_dict['scaler'].feature_names_in_ = model_dict['model'].feature_names_in_
        elif hasattr(st.session_state, 'features'):
            model_dict['scaler'].feature_names_in_ = np.array(st.session_state.features)
        
        model_path = os.path.join('models', f"{base_filename}_model.joblib")
        scaler_path = os.path.join('models', f"{base_filename}_scaler.joblib")
        
        joblib.dump(model_dict['model'], model_path)
        joblib.dump(model_dict['scaler'], scaler_path)
        
        return model_path, scaler_path

    def train_and_evaluate_model(self, clf_dict, X_train, X_test, y_train, y_test, model_name):
        """Train and evaluate a single model"""
        start_time = time.time()
        
        try:
            scaler = clf_dict['scaler']
            feature_names = st.session_state.features if hasattr(st.session_state, 'features') else None
            
            if model_name == 'MultinomialNB':
                X_train_positive = self.ensure_non_negative(X_train)
                X_test_positive = self.ensure_non_negative(X_test)
                X_train_scaled = scaler.fit_transform(X_train_positive)
                X_test_scaled = scaler.transform(X_test_positive)
                
                if np.any(X_train_scaled < 0) or np.any(X_test_scaled < 0):
                    raise ValueError("Negative values in scaled data")
            else:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            
            if feature_names is not None:
                if hasattr(clf_dict['model'], 'feature_names_in_'):
                    clf_dict['model'].feature_names_in_ = np.array(feature_names)
                scaler.feature_names_in_ = np.array(feature_names)
            
            clf_dict['model'].fit(X_train_scaled, y_train)
            y_pred = clf_dict['model'].predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            training_time = time.time() - start_time
            
            model_path, scaler_path = self.save_model(clf_dict, model_name)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            return {
                'model_name': model_name,
                'accuracy': accuracy,
                'training_time': training_time,
                'model': clf_dict['model'],
                'predictions': y_pred,
                'status': 'success',
                'scaler': scaler_path,
                'model_path': model_path,
                'confusion_matrix': conf_matrix
            }
        except Exception as e:
            return {
                'model_name': model_name,
                'accuracy': 0,
                'training_time': 0,
                'model': None,
                'predictions': None,
                'status': f'failed: {str(e)}',
                'scaler': None,
                'model_path': None,
                'confusion_matrix': None
            }

class Visualizer:
    @staticmethod
    def plot_learning_curve(estimator, X, y, title, ax):
        """Plot learning curves for a model"""
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        ax.plot(train_sizes, train_mean, label='Training score')
        ax.plot(train_sizes, test_mean, label='Cross-validation score')
        
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        
        ax.set_xlabel('Training Examples')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(True)

    def create_confusion_matrices_plot(self, successful_results, y_test):
        """Create and display confusion matrices for successful models"""
        n_models = len(successful_results)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(15, 5 * n_rows))
        colors = ['white', '#4a90e2']
        n_bins = 100
        # cmap = LinearSegmentedColormap.from_list("custom_blues", colors, N=n_bins)
        
        for idx, result in enumerate(successful_results):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            sns.heatmap(
                result['confusion_matrix'],
                annot=True,
                fmt='d',
                # cmap=cmap,
                cmap='viridis',
                ax=ax,
                xticklabels=sorted(set(y_test)),
                yticklabels=sorted(set(y_test))
            )
            
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f"{result['model_name']}\nAccuracy: {result['accuracy']:.4f}")
        
        plt.tight_layout()
        return fig

    def create_performance_summary_plot(self, successful_df, selected_models):
        """Create performance metrics summary plot"""
        metrics_to_compare = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        summary_df = successful_df[successful_df['Model'].isin(selected_models)].melt(
            id_vars=['Model'],
            value_vars=metrics_to_compare,
            var_name='Metric',
            value_name='Score'
        )
        
        fig_summary = px.bar(
            summary_df,
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title="Model Performance Metrics Comparison",
            text='Score'
        )
        
        fig_summary.update_layout(
            xaxis_tickangle=-45,
            showlegend=True,
            height=600,
            yaxis=dict(
                range=[0, 1],
                title='Score'
            ),
            legend=dict(
                title='Metric',
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        fig_summary.update_traces(
            texttemplate='%{text:.4f}',
            textposition='outside',
            textangle=0
        )
        
        summary_df['Avg_Score'] = summary_df.groupby('Model')['Score'].transform('mean')
        models_order = summary_df.drop_duplicates('Model').sort_values('Avg_Score', ascending=False)['Model']
        fig_summary.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': models_order})
        
        return fig_summary

class StreamlitUI:
    def __init__(self):
        self.data_generator = DataGenerator()
        self.model_manager = ModelManager()
        self.visualizer = Visualizer()
        
        # Add default configurations as class attribute
        self.default_configs = {
            # Features: [length (mm), width (mm), density (g/cm³), pH]
            
            # AMPALAYA: Medium length (150-180mm), thin width (40-50mm)
            # Medium density (95 g/cm³) due to hollow interior, slightly basic pH (6.8-7.0)
            "Ampalaya": {'mean': [165, 45, 95, 6.9], 'std': [15, 5, 10, 0.1]},   
            
            # BANANA: Long length (180-220mm), medium width (30-40mm) 
            # Low density (85 g/cm³), acidic pH (4.5-5.2)
            "Banana": {'mean': [200, 35, 85, 4.8], 'std': [20, 5, 8, 0.3]}, 
            
            # CABBAGE: Round shape - similar length/width (150-200mm x 150-200mm)
            # Very low density (65 g/cm³) due to layered leaves, neutral pH (6.5-7.0)
            "Cabbage": {'mean': [175, 175, 65, 6.8], 'std': [25, 25, 5, 0.2]},
            
            # CARROT: Medium length (140-180mm), narrow width (25-35mm)
            # High density (115 g/cm³) due to dense flesh, slightly acidic pH (6.0-6.5) 
            "Carrot": {'mean': [160, 30, 115, 6.3], 'std': [20, 5, 10, 0.2]},    
            
            # CASSAVA: Long length (200-300mm), thick width (50-80mm)
            # High density (125 g/cm³) due to starchy flesh, slightly acidic pH (6.0-6.5)
            "Cassava": {'mean': [250, 65, 125, 6.2], 'std': [50, 15, 12, 0.2]}
        }
        
        # Default feature names that match the measurements in default_configs
        self.default_features = [
            'length (mm)',
            'width (mm)', 
            'density (g/cm³)',
            'pH'
        ]

        # Add new session state variables for static visualizations
        self.initialize_static_visualizations()

        # Add new session state variable for data source
        if 'data_source' not in st.session_state:
            st.session_state.data_source = 'synthetic'

    def initialize_static_visualizations(self):
        """Initialize session state variables for static visualizations"""
        if 'confusion_matrices_fig' not in st.session_state:
            st.session_state.confusion_matrices_fig = None
        if 'learning_curves_fig' not in st.session_state:
            st.session_state.learning_curves_fig = None

    def initialize_session_state(self):
        """Initialize all session state variables"""
        session_vars = {
            'data_generated': False,
            'df': None,
            'features': None,
            'feature_configs': None,
            'X_train': None,
            'X_test': None,
            'y_train': None,
            'y_test': None,
            'y_pred': None,
            'model_results': None,
            'best_model': None,
            'accuracy': None,
            'feature_importance': None,
            'split_info': None
        }
        
        for var, value in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = value

    def setup_page_config(self):
        """Configure the Streamlit page"""
        st.set_page_config(
            page_title="ML Model Generator & Implementationsssss",
            page_icon="🤖",
            layout="wide",
            menu_items={
                'About': """
## Final project in Modeling and Simulation \n
### Juan Dela Cruz - BSCS 4A"""
            }
        )

    def get_sidebar_inputs(self):
        """Get all inputs from the sidebar"""
        st.sidebar.header("Data Generation Parameters")
        
        # Feature configuration
        st.sidebar.subheader("Feature Configuration")
        
        # Initialize default features if not in session state
        if 'features_input' not in st.session_state:
            st.session_state.features_input = ", ".join(self.default_features)
        
        features_input = st.sidebar.text_input(
            "Enter feature names (comma-separated)",
            key='features_input'
        )
        features = [f.strip() for f in features_input.split(",")]
        
        # Initialize default classes if not in session state 
        if 'classes_input' not in st.session_state:
            st.session_state.classes_input = ", ".join(self.default_configs.keys())
        
        classes_input = st.sidebar.text_input(
            "Enter class names (comma-separated)",
            key='classes_input'
        )
        classes = [c.strip() for c in classes_input.split(",")]
        
        # Generate feature configs
        feature_configs = {}
        for feature in features:
            feature_configs[feature] = {
                'type': 'random',
                'std': 20.0,
                'center': None
            }
        
        return features, feature_configs, classes

    def get_class_configs(self, classes, features):
        """Get class-specific configurations from the sidebar"""
        class_configs = {}
        st.sidebar.subheader("Class-Specific Settings")
        
        for class_name in classes:
            with st.sidebar.expander(f"{class_name} Settings", expanded=False):
                checkbox_key = f"use_specific_{class_name}"
                
                # Initialize checkbox state if not in session state
                if checkbox_key not in st.session_state:
                    st.session_state[checkbox_key] = True
                
                use_specific = st.checkbox(
                    f"Set specific values for {class_name}", 
                    key=checkbox_key
                )
                
                means = []
                stds = []
                
                # Generate unique means for each class if not in default configs
                if class_name not in self.default_configs:
                    # Generate random means between 0-100 that are different from other classes
                    random_means = []
                    for _ in range(len(features)):
                        mean = np.random.uniform(0, 100)
                        # Ensure means are unique across classes
                        while any(abs(mean - c['mean'][_]) < 10 for c in class_configs.values() if 'mean' in c):
                            mean = np.random.uniform(0, 100)
                        random_means.append(mean)
                    default_values = {'mean': random_means, 'std': [20.0] * len(features)}
                else:
                    # Ensure default values match the number of features
                    default_means = self.default_configs[class_name]['mean']
                    default_stds = self.default_configs[class_name]['std']
                    
                    # If we have more features than default values, extend with random values
                    if len(features) > len(default_means):
                        additional_means = [np.random.uniform(0, 100) for _ in range(len(features) - len(default_means))]
                        additional_stds = [20.0 for _ in range(len(features) - len(default_stds))]
                        default_means.extend(additional_means)
                        default_stds.extend(additional_stds)
                    # If we have fewer features than default values, truncate
                    elif len(features) < len(default_means):
                        default_means = default_means[:len(features)]
                        default_stds = default_stds[:len(features)]
                    
                    default_values = {'mean': default_means, 'std': default_stds}
                
                if use_specific:
                    for idx, feature in enumerate(features):
                        mean_key = f"mean_{class_name}_{feature}"
                        std_key = f"std_{class_name}_{feature}"
                        
                        if mean_key not in st.session_state:
                            st.session_state[mean_key] = float(default_values['mean'][idx])
                        if std_key not in st.session_state:
                            st.session_state[std_key] = float(default_values['std'][idx])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            mean = st.number_input(
                                f"Mean for {feature}",
                                key=mean_key
                            )
                            means.append(mean)
                        with col2:
                            std = st.number_input(
                                f"Std Dev for {feature}",
                                min_value=0.1,
                                key=std_key
                            )
                            stds.append(std)
                else:
                    # Use default values if specific values not requested
                    means = default_values['mean']
                    stds = default_values['std']
                
                class_configs[class_name] = {
                    'mean': means,
                    'std': stds
                }
        
        return class_configs

    def get_training_params(self):
        """Get training parameters from the sidebar"""
        st.sidebar.subheader("Sample Size & Train/Test Split Configuration")
        
        # Initialize default values if not in session state
        if 'n_samples' not in st.session_state:
            st.session_state.n_samples = 10000
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            n_samples = st.slider(
                "Number of samples", 
                500, 
                50000,
                step=500,
                key='n_samples'
            )
        
        with col2:
            test_size = st.slider(
                "Test Size",
                min_value=10,
                max_value=50,
                value=30,  # Default value directly in the widget
                step=5,
                key='test_size',
                format="%d%%",
                help="Percentage of data to use for testing"
            )
            st.write(f"Test: {test_size}% / Train: {100 - test_size}%")
        
        return n_samples, test_size

    def generate_and_train(self, n_samples, feature_configs, classes, class_configs, test_size):
        """Generate data and train models"""
        X, y = self.data_generator.generate_synthetic_data(
            n_samples, 
            feature_configs, 
            classes, 
            class_configs
        )
        
        st.session_state.df = pd.DataFrame(X, columns=st.session_state.features)
        st.session_state.df['target'] = y
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size/100, 
            random_state=42
        )
        
        # Store split data
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        
        # Get classifiers and train models
        classifiers = self.model_manager.get_classifiers()
        results = []
        
        with st.spinner('Training models... Please wait.'):
            progress_bar = st.progress(0)
            for idx, (name, clf_dict) in enumerate(classifiers.items()):
                result = self.model_manager.train_and_evaluate_model(
                    clf_dict, 
                    X_train, 
                    X_test, 
                    y_train, 
                    y_test, 
                    name
                )
                results.append(result)
                progress_bar.progress((idx + 1) / len(classifiers))
        
        st.session_state.model_results = results
        st.session_state.data_generated = True
        
        # Find best model
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            best_model = max(successful_results, key=lambda x: x['accuracy'])
            st.session_state.best_model = best_model

        # Store split information
        st.session_state.split_info = {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'test_percentage': test_size
        }
        st.session_state.feature_configs = feature_configs

        # Generate static visualizations after training
        successful_results = [r for r in st.session_state.model_results if r['status'] == 'success']
        if successful_results:
            # Generate and store confusion matrices
            st.session_state.confusion_matrices_fig = self.visualizer.create_confusion_matrices_plot(
                successful_results, 
                st.session_state.y_test
            )
            
            # Generate and store learning curves
            st.session_state.learning_curves_fig = self.generate_learning_curves_figure(successful_results)

    def generate_learning_curves_figure(self, successful_results):
        """Generate learning curves figure"""
        successful_results.sort(key=lambda x: x['accuracy'], reverse=True)
        n_models = len(successful_results)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig_learning = plt.figure(figsize=(15, 5 * n_rows))
        
        for idx, result in enumerate(successful_results):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            model_name = result['model_name']
            model = result['model']
            scaler = joblib.load(result['scaler'])
            
            if model_name == 'MultinomialNB':
                X_scaled = self.model_manager.ensure_non_negative(
                    st.session_state.df.drop('target', axis=1)
                )
                X_scaled = scaler.transform(X_scaled)
            else:
                X_scaled = scaler.transform(st.session_state.df.drop('target', axis=1))
            
            y = st.session_state.df['target']
            
            self.visualizer.plot_learning_curve(
                model,
                X_scaled,
                y,
                f'Learning Curve - {model_name}\nFinal Accuracy: {result["accuracy"]:.4f}',
                ax
            )
        
        plt.tight_layout()
        return fig_learning

    def display_model_comparison(self):
        """Display model comparison section"""
        st.subheader("Model Comparison")
        
        comparison_data = []
        for result in st.session_state.model_results:
            if result['status'] == 'success':
                report_dict = classification_report(
                    st.session_state.y_test,
                    result['predictions'],
                    output_dict=True
                )
                
                macro_avg = report_dict['macro avg']
                
                comparison_data.append({
                    'Model': result['model_name'],
                    'Accuracy': float(f"{result['accuracy']:.4f}"),
                    'Precision': float(f"{macro_avg['precision']:.4f}"),
                    'Recall': float(f"{macro_avg['recall']:.4f}"),
                    'F1-Score': float(f"{macro_avg['f1-score']:.4f}"),
                    'Training Time (s)': float(f"{result['training_time']:.3f}"),
                    'Status': 'Success'
                })
            else:
                comparison_data.append({
                    'Model': result['model_name'],
                    'Accuracy': 0,
                    'Precision': 0,
                    'Recall': 0,
                    'F1-Score': 0,
                    'Training Time (s)': 0,
                    'Status': result['status']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        st.dataframe(comparison_df.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}',
            'Training Time (s)': '{:.3f}'
        }))
        
        return comparison_df

    def display_metric_visualization(self, comparison_df):
        """Display metric visualization section"""
        metric_to_plot = st.selectbox(
            "Select metric to visualize",
            ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time (s)']
        )
        
        successful_df = comparison_df[comparison_df['Status'] == 'Success']
        
        if metric_to_plot == 'Training Time (s)':
            successful_df = successful_df.sort_values(metric_to_plot)
        else:
            successful_df = successful_df.sort_values(metric_to_plot, ascending=False)
            
        fig_comparison = px.bar(
            successful_df,
            x='Model',
            y=metric_to_plot,
            title=f"Model {metric_to_plot} Comparison",
            color=metric_to_plot,
            text=metric_to_plot
        )
        
        fig_comparison.update_layout(
            xaxis_tickangle=-45,
            showlegend=True,
            height=500,
            yaxis=dict(
                range=[0, 1] if metric_to_plot != 'Training Time (s)' else None
            )
        )
        
        fig_comparison.update_traces(
            texttemplate='%{text:.4f}',
            textposition='outside',
            textangle=0
        )
        
        st.plotly_chart(fig_comparison)
        return successful_df

    def display_best_model_performance(self):
        """Display best model performance section"""
        if hasattr(st.session_state, 'best_model'):
            st.subheader("Best Model Performance")
            best_model = st.session_state.best_model
            st.write(f"Best Model: **{best_model['model_name']}**")
            st.write(f"Accuracy: {best_model['accuracy']:.4f}")
            
            st.write("Classification Report (Best Model):")
            report_dict = classification_report(
                st.session_state.y_test, 
                best_model['predictions'], 
                output_dict=True
            )
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format('{:.4f}'))

    def display_dataset_info(self):
        """Display dataset split information"""
        if st.session_state.split_info:
            st.subheader("Dataset Split Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Samples",
                    st.session_state.split_info['total_samples']
                )
            
            with col2:
                st.metric(
                    "Training Samples",
                    f"{st.session_state.split_info['train_samples']} "
                    f"({100 - st.session_state.split_info['test_percentage']}%)"
                )
            
            with col3:
                st.metric(
                    "Testing Samples",
                    f"{st.session_state.split_info['test_samples']} "
                    f"({st.session_state.split_info['test_percentage']}%)"
                )

    def display_feature_configs(self):
        """Display feature configurations"""
        st.subheader("Feature Configurations")
        config_data = []
        for feature, config in st.session_state.feature_configs.items():
            config_data.append({
                'Feature': feature,
                'Type': config['type'],
                'Std Dev': config['std'],
                'Center': config['center'] if config['type'] == 'user-defined' else 'Random'
            })
        st.table(pd.DataFrame(config_data))

    def display_data_samples(self):
        """Display original and scaled data samples"""
        st.subheader("Generated Data Sample")
        
        # Get random samples from each class
        unique_classes = st.session_state.df['target'].unique()
        samples_per_class = 2  # Number of samples to show per class
        
        sampled_data = []
        for class_name in unique_classes:
            class_data = st.session_state.df[st.session_state.df['target'] == class_name]
            sampled_data.append(class_data.sample(n=min(samples_per_class, len(class_data))))
        
        sampled_df = pd.concat(sampled_data).sample(frac=1).reset_index(drop=True)
        
        col1, col2 = st.columns(2)

        with col1:
            st.write("Original Data (Random samples from each class):")
            st.write(sampled_df)

        with col2:
            st.write("Scaled Data (using best model's scaler):")
            if st.session_state.best_model and st.session_state.best_model['status'] == 'success':
                best_model_name = st.session_state.best_model['model_name']
                scaler = joblib.load(st.session_state.best_model['scaler'])
                
                features_df = sampled_df.drop('target', axis=1)
                
                if best_model_name == 'MultinomialNB':
                    features_scaled = self.model_manager.ensure_non_negative(features_df)
                    features_scaled = scaler.transform(features_scaled)
                else:
                    features_scaled = scaler.transform(features_df)
                
                scaled_df = pd.DataFrame(
                    features_scaled, 
                    columns=features_df.columns,
                    index=features_df.index
                )
                scaled_df['target'] = sampled_df['target']
                
                st.write(scaled_df)
            else:
                st.write("No scaled data available (best model not found)")

    def display_confusion_matrices(self):
        """Display confusion matrices section"""
        st.subheader("Confusion Matrices")
        st.write("""
        Confusion matrices show the model's prediction performance across different classes.
        - Each row represents the actual class
        - Each column represents the predicted class
        - Diagonal elements represent correct predictions (True Positives for each class)
        - Off-diagonal elements represent incorrect predictions
        - Numbers show how many samples were classified for each combination
        - Colors range from yellow (high values) to green-blue (low values) using the viridis colormap
        """)
        if st.session_state.confusion_matrices_fig is not None:
            st.pyplot(st.session_state.confusion_matrices_fig)
            plt.close()
        


    def display_performance_summary(self, successful_df):
        """Display performance metrics summary"""
        st.subheader("Performance Metrics Summary")
        
        all_models = successful_df['Model'].unique().tolist()
        default_selection = all_models
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_models = st.multiselect(
                "Select models to compare",
                all_models,
                default=default_selection
            )
        
        if not selected_models:
            st.warning("Please select at least one model to display the comparison.")
            return
        
        fig_summary = self.visualizer.create_performance_summary_plot(
            successful_df, 
            selected_models
        )
        st.plotly_chart(fig_summary, use_container_width=True)

    def display_saved_models(self):
        """Display saved models information"""
        st.subheader("Saved Models")
        saved_models = []
        for result in st.session_state.model_results:
            if result['status'] == 'success' and result['model_path']:
                saved_models.append({
                    'Model': result['model_name'],
                    'Accuracy': result['accuracy'],
                    'Model Path': result['model_path'],
                    'Scaler Path': result['scaler']
                })
        
        if saved_models:
            saved_df = pd.DataFrame(saved_models)
            st.dataframe(saved_df.style.format({
                'Accuracy': '{:.4f}'
            }))
        else:
            st.info("No models were saved. Models are saved automatically when accuracy exceeds 0.5")

    def display_download_section(self):
        """Display dataset download section"""
        st.subheader("Download Dataset")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.df is not None:
                csv = st.session_state.df.to_csv(index=False)
                st.download_button(
                    label="Download Original Dataset (CSV)",
                    data=csv,
                    file_name=f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv',
                    help="Download the original unscaled dataset"
                )
        
        with col2:
            if st.session_state.best_model and st.session_state.best_model['status'] == 'success':
                best_model_name = st.session_state.best_model['model_name']
                scaler = joblib.load(st.session_state.best_model['scaler'])
                
                features_df = st.session_state.df.drop('target', axis=1)
                if best_model_name == 'MultinomialNB':
                    features_scaled = self.model_manager.ensure_non_negative(features_df)
                    features_scaled = scaler.transform(features_scaled)
                else:
                    features_scaled = scaler.transform(features_df)
                
                scaled_df = pd.DataFrame(
                    features_scaled, 
                    columns=features_df.columns,
                    index=features_df.index
                )
                scaled_df['target'] = st.session_state.df['target']
                
                csv_scaled = scaled_df.to_csv(index=False)
                st.download_button(
                    label="Download Scaled Dataset (CSV)",
                    data=csv_scaled,
                    file_name=f"synthetic_data_scaled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv',
                    help="Download the scaled dataset (using best model's scaler)"
                )

    def display_dataset_statistics(self):
        """Display dataset statistics"""
        with st.expander("Dataset Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Original Dataset Statistics:")
                st.write(st.session_state.df.describe())
            
            with col2:
                if st.session_state.best_model and st.session_state.best_model['status'] == 'success':
                    st.write("Scaled Dataset Statistics:")
                    best_model_name = st.session_state.best_model['model_name']
                    scaler = joblib.load(st.session_state.best_model['scaler'])
                    
                    features_df = st.session_state.df.drop('target', axis=1)
                    if best_model_name == 'MultinomialNB':
                        features_scaled = self.model_manager.ensure_non_negative(features_df)
                        features_scaled = scaler.transform(features_scaled)
                    else:
                        features_scaled = scaler.transform(features_df)
                    
                    scaled_df = pd.DataFrame(
                        features_scaled,
                        columns=features_df.columns,
                        index=features_df.index
                    )
                    scaled_df['target'] = st.session_state.df['target']
                    st.write(scaled_df.describe())

    def display_learning_curves(self):
        """Display learning curves section"""
        st.subheader("Learning Curves")
        st.write("""
        Learning curves show how model performance changes with increasing training data.
        - Blue line: Training score
        - Orange line: Cross-validation score
        - Shaded areas represent standard deviation
        """)
        
        if st.session_state.learning_curves_fig is not None:
            st.pyplot(st.session_state.learning_curves_fig)
            plt.close()

    def display_feature_visualization(self):
        """Display 2D and 3D feature visualizations"""
        st.subheader("Feature Visualization")
        plot_type = st.radio("Select plot type", ["2D Plot", "3D Plot"], index=1)
        
        if plot_type == "2D Plot":
            col1, col2 = st.columns(2)
            
            with col1:
                x_feature = st.selectbox(
                    "Select X-axis feature",
                    st.session_state.features,
                    index=0,
                    key='x_2d'
                )
            
            with col2:
                y_features = [f for f in st.session_state.features if f != x_feature]
                y_feature = st.selectbox(
                    "Select Y-axis feature",
                    y_features,
                    index=0,
                    key='y_2d'
                )
            
            fig = px.scatter(
                st.session_state.df, 
                x=x_feature, 
                y=y_feature, 
                color='target',
                title=f"2D Visualization of {x_feature} vs {y_feature}",
                labels={'target': 'Class'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # 3D Plot
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_feature = st.selectbox(
                    "Select X-axis feature",
                    st.session_state.features,
                    index=0,
                    key='x_3d'
                )
            
            with col2:
                y_features = [f for f in st.session_state.features if f != x_feature]
                y_feature = st.selectbox(
                    "Select Y-axis feature",
                    y_features,
                    index=0,
                    key='y_3d'
                )
            
            with col3:
                z_features = [f for f in st.session_state.features if f not in [x_feature, y_feature]]
                z_feature = st.selectbox(
                    "Select Z-axis feature",
                    z_features,
                    index=0,
                    key='z_3d'
                )
            
            fig = px.scatter_3d(
                st.session_state.df,
                x=x_feature,
                y=y_feature,
                z=z_feature,
                color='target',
                title=f"3D Visualization of {x_feature} vs {y_feature} vs {z_feature}",
                labels={'target': 'Class'}
            )
            
            fig.update_layout(
                scene = dict(
                    xaxis_title=x_feature,
                    yaxis_title=y_feature,
                    zaxis_title=z_feature
                ),
                scene_camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def get_data_source(self):
        """Get user's choice of data source"""
        st.sidebar.header("Data Source")
        data_source = st.sidebar.radio(
            "Choose data source",
            ['Generate Synthetic Data', 'Upload Dataset'],
            key='data_source_radio'
        )
        st.session_state.data_source = 'synthetic' if data_source == 'Generate Synthetic Data' else 'upload'
        return st.session_state.data_source

    def upload_dataset(self):
        """Handle dataset upload"""
        st.sidebar.header("Upload Dataset")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with features and target column"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Let user select target column
                target_col = st.sidebar.selectbox(
                    "Select target column",
                    df.columns.tolist()
                )
                
                # Store features and target
                features = [col for col in df.columns if col != target_col]
                X = df[features]
                y = df[target_col]
                
                # Store in session state
                st.session_state.df = df
                st.session_state.features = features
                
                # Train test split
                test_size = st.sidebar.slider(
                    "Test Size",
                    min_value=10,
                    max_value=50,
                    value=30,
                    step=5,
                    format="%d%%",
                    help="Percentage of data to use for testing"
                )
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size/100,
                    random_state=42
                )
                
                # Store split data
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                # Store split information
                st.session_state.split_info = {
                    'total_samples': len(X),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'test_percentage': test_size
                }
                
                return True
            except Exception as e:
                st.sidebar.error(f"Error loading dataset: {str(e)}")
                return False
        return False

    def run(self):
        """Main application logic"""
        self.setup_page_config()
        self.initialize_session_state()
        
        st.title("ML Model Generator")
        
        # Get data source choice
        data_source = self.get_data_source()
        
        if data_source == 'synthetic':
            st.sidebar.header("Synthetic Data Generation")
            # Get inputs from sidebar for synthetic data
            features, feature_configs, classes = self.get_sidebar_inputs()
            class_configs = self.get_class_configs(classes, features)
            n_samples, test_size = self.get_training_params()
            
            # Store features in session state
            st.session_state.features = features
            
            # Generate Data button
            if st.sidebar.button("Generate Data and Train Models"):
                self.generate_and_train(n_samples, feature_configs, classes, class_configs, test_size)
        
        else:  # upload
            # Handle dataset upload
            if self.upload_dataset():
                if st.sidebar.button("Train Models"):
                    # Get classifiers and train models
                    classifiers = self.model_manager.get_classifiers()
                    results = []
                    
                    with st.spinner('Training models... Please wait.'):
                        progress_bar = st.progress(0)
                        for idx, (name, clf_dict) in enumerate(classifiers.items()):
                            result = self.model_manager.train_and_evaluate_model(
                                clf_dict,
                                st.session_state.X_train,
                                st.session_state.X_test,
                                st.session_state.y_train,
                                st.session_state.y_test,
                                name
                            )
                            results.append(result)
                            progress_bar.progress((idx + 1) / len(classifiers))
                    
                    st.session_state.model_results = results
                    st.session_state.data_generated = True
                    
                    # Find best model
                    successful_results = [r for r in results if r['status'] == 'success']
                    if successful_results:
                        best_model = max(successful_results, key=lambda x: x['accuracy'])
                        st.session_state.best_model = best_model
                        
                        # Generate static visualizations
                        st.session_state.confusion_matrices_fig = self.visualizer.create_confusion_matrices_plot(
                            successful_results,
                            st.session_state.y_test
                        )
                        st.session_state.learning_curves_fig = self.generate_learning_curves_figure(successful_results)
        
        # Display results if data has been generated/uploaded and trained
        if st.session_state.data_generated:
            self.display_dataset_info()
            self.display_data_samples()
            self.display_feature_visualization()
            self.display_download_section()
            self.display_dataset_statistics()
            self.display_best_model_performance()
            successful_df = self.display_model_comparison()
            
            if successful_df is not None and not successful_df.empty:
                self.display_performance_summary(successful_df)
                self.display_saved_models()
                self.display_learning_curves()
                self.display_confusion_matrices()
        else:
            if data_source == 'synthetic':
                st.info("Please generate data using the sidebar button to view visualizations and results.")
            else:
                st.info("Please upload a dataset and click 'Train Models' to view visualizations and results.")

def main():
    app = StreamlitUI()
    app.run()

if __name__ == "__main__":
    main()

