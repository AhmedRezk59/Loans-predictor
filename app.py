import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a given file path.

    Args:
        file_path (str): The path to the dataset file.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    if not os.path.exists(os.path.abspath(file_path)):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    file_ext = os.path.splitext(file_path)[1].lower()
    try:
        if file_ext == ".csv":
            return pd.read_csv(file_path)
        elif file_ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        elif file_ext == ".pkl":
            with open(file_path , "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    except Exception as e:
        raise ValueError(f"Error loading dataset from {file_path}: {str(e)}")

def visualize_dataset(df: pd.DataFrame, dataset_name: str):
    viz_folder = f"{dataset_name}_visualizations"
    os.makedirs(viz_folder, exist_ok=True)
    models_folder = os.path.join(viz_folder, "model_results")
    os.makedirs(models_folder, exist_ok=True)
    
    # Visualize basic statistics
    target_col = df.columns[-1]
    plt.figure(figsize=(10, 7))
    if df[target_col].dtype == 'object':
        ax = sns.countplot(x=target_col, data=df)
        plt.title(f"Distribution of {target_col}")
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.2)
    else:
        ax = sns.histplot(df[target_col],kde=True)
        plt.title(f"Distribution of {target_col}")
    
    ### Save the plot
    safe_target_name = target_col.replace(" ", "_").replace("/", "_").replace("\\", "_")
    filename = os.path.join(viz_folder, f"{safe_target_name}_distribution.png")
    plt.tight_layout()
    plt.savefig(filename,dpi=150)
    plt.close()
    print(f"✅ Visualization saved to {filename}")
    
    ### Visualize categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    categorical_cols =[col for col in categorical_cols if not (
        "id" in col.lower() or
        "loan_id" in col.lower() or
        "_id" in col.lower() or
        df[col].nunique() > 20
    )]

    if len(categorical_cols) > 0:
        cat_cols_to_plot = []
        if 'Sub-Group' in categorical_cols:
            cat_cols_to_plot.append('Sub-Group')
        elif 'Group' in categorical_cols:
            cat_cols_to_plot.append('Group')
        elif 'Category' in categorical_cols:
            cat_cols_to_plot.append('Category')
        else:
            cat_cols_sorted = sorted([(col,df[col].nunique()) for col in categorical_cols], key=lambda x:x[1])
            cat_cols_to_plot = [col for col, _ in cat_cols_sorted[:3]]
    
    for col in cat_cols_to_plot:
        plt.figure(figsize=(10,8))
        ax = sns.countplot(x=col , data=df,order=df[col].value_counts().iloc[:20].index)
        plt.title(f"Distribution of {col}")
        ax.tick_params(axis="y", labelsize=10)
        safe_col_name = col.replace(" ", "_").replace("/", "_").replace("\\", "_").replace("(", "").replace(")", "")
        filename = os.path.join(viz_folder, f"{safe_col_name}_distribution.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"✅ Visualization saved to {filename}")
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    numerical_df = df.select_dtypes(include=['number'])
    if not numerical_df.empty:
        numerical_df = numerical_df.loc[:,numerical_df.nunique() > 1]
        if not  numerical_df.empty and numerical_df.shape[1] > 1:
            numerical_df = numerical_df.fillna(numerical_df.mean())
            corr = numerical_df.corr()
            
            if corr.shape[0] > 15:
                print("⚠️ Warning: Too many numerical features to visualize correlation heatmap.")
                corr = corr.iloc[:15, :15]
                mean_abs_corr = corr.abs().mean()
                top_features = mean_abs_corr.nlargest(15).index
                corr = corr.loc[top_features, top_features]   
            
            mask = np.triu(corr)
            
            ### Create a heatmap
            sns.heatmap(
                corr,
                annot=True,
                mask=mask,
                cmap="coolwarm",
                linewidths=0.5,
                fmt=".2f",
                annot_kws={"size": 8 if corr.shape[0] > 10 else 10},
            )
            
            plt.title(f"{dataset_name} - Correlation Heatmap")
            
            filename = os.path.join(viz_folder, f"{dataset_name}_correlation_heatmap.png")
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close()
            print(f"✅ Correlation heatmap saved to '{filename}'")
    
    ### Box plots for numerical features
    plt.figure(figsize=(12, 8))
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns[:5]  # First 5 numerical
    if len(numerical_cols) > 0:
        sns.boxplot(data=df[numerical_cols])
        plt.title(f"{dataset_name} - Box Plots of Numerical Features")
        plt.xticks(rotation=45, ha='right')
        
        # Create a descriptive filename
        numerical_desc = "_".join([col.replace(" ", "").replace("/", "")[:5] for col in numerical_cols[:3]])
        filename = os.path.join(viz_folder, f"{dataset_name}_boxplots_{numerical_desc}.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"✅ Box plots saved to '{filename}'")
    return viz_folder


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the dataset for machine learning.

    Args:
        df (pd.DataFrame): The dataset to preprocess.

    Returns:
        tuple: Processed features and target variable, and the preprocessor.
    """
    print(df.dtypes)
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    
    # Identify target column (assuming it's the last column)
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore',sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if y.dtype == "object":
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, dataset_name, viz_folder):
    """Train and evaluate multiple machine learning models"""
    print("\n🤖 Let's train and evaluate our machine learning models!")
    
    # Create a subfolder for model performance visualizations
    models_folder = os.path.join(viz_folder, "model_results")
    os.makedirs(models_folder, exist_ok=True)
    print(f"✅ Created folder '{models_folder}' for model performance visualizations")
    
    # Preprocess data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(probability=True),
        "Random Forest": RandomForestClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
    }
    
    # Add XGBoost if available
    if 'has_xgboost' in globals() and has_xgboost:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\n🔍 Training {name}...")
        
        # Train model
        model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
        
        # Print results
        print(f"✅ {name} trained successfully!")
        print(f"📈 Accuracy: {accuracy * 100:.2f}%")
        print("\n📊 Confusion Matrix:")
        print(conf_matrix)
        print("\n📋 Classification Report:")
        print(class_report)
        
        # Visualize confusion matrix
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=sorted(set(y_test)),
                       yticklabels=sorted(set(y_test)))
            plt.title(f'{dataset_name} - {name} Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            safe_name = name.replace(' ', '_').replace('/', '_')
            matrix_file = os.path.join(models_folder, f"{dataset_name}_{safe_name}_confusion_matrix.png")
            plt.savefig(matrix_file, dpi=150)
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not create confusion matrix visualization for {name}: {str(e)}")
    
    # Create a bar chart comparing model accuracies
    try:
        model_names = list(results.keys())
        accuracies = [results[model]['accuracy'] * 100 for model in model_names]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(model_names, accuracies, color='skyblue')
        plt.title(f'{dataset_name} - Model Accuracy Comparison')
        plt.xlabel('Models')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45, ha='right')
        
        # Add accuracy values on top of bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f"{accuracies[i]:.2f}%", ha='center', va='bottom',
                   fontsize=9, rotation=0)
            
        plt.ylim(0, 105)  # Leave space at the top for text
        plt.tight_layout()
        plt.savefig(os.path.join(models_folder, f"{dataset_name}_accuracy_comparison.png"), dpi=150)
        plt.close()
        print(f"✅ Model accuracy comparison chart saved to '{models_folder}/{dataset_name}_accuracy_comparison.png'")
    except Exception as e:
        print(f"⚠️ Could not create accuracy comparison chart: {str(e)}")
    
    return results

def process_dataset(file_path:str, dataset_name:str):
    """
    Process a dataset for the loan approval prediction project.

    Args:
        file_path (str): _description_
        dataset_name (str): _description_
    """
    print(f"\n{'='*80}")
    print(f"✨ DATASET: {dataset_name} ✨".center(80))
    print(f"{'='*80}\n")
    df = load_dataset(file_path)
    if df is None:
        print("No data loaded. Please check the dataset file.")
        return None
    
    if len(df) < 1000:
        print(f"⚠️ Warning: This dataset has only {len(df)} records, which is less than the recommended 1000 records.")
        proceed = input("Do you want to continue with this dataset anyway? (y/n): ")
        if proceed.lower() != 'y':
            return None
    print(f"\n📊 This dataset has {df.shape[0]} rows and {df.shape[1]} columns")
    
    try:
        viz_folder = visualize_dataset(df, dataset_name)
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        viz_folder = f"{dataset_name}_visualizations"
        os.makedirs(viz_folder, exist_ok=True)
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, dataset_name, viz_folder)
    return results

def compare_models(all_results):
    """Create a comparison table of model performances across all datasets"""
    print(f"\n{'='*80}")
    print("📊 MODEL PERFORMANCE COMPARISON 📊".center(80))
    print(f"{'='*80}\n")
    
    # Create a folder for comparison results
    comparison_folder = "comparison_results"
    os.makedirs(comparison_folder, exist_ok=True)
    
    # Create comparison DataFrame
    comparison = pd.DataFrame()
    
    # Get accuracies for each dataset and model (as percentages)
    for dataset, results in all_results.items():
        accuracies = {model: result['accuracy'] * 100 for model, result in results.items()}
        comparison[dataset] = pd.Series(accuracies)
    
    # Add average performance column
    comparison['Average'] = comparison.mean(axis=1)
    
    # Sort by average performance
    comparison = comparison.sort_values('Average', ascending=False)
    
    # Format percentages
    comparison = comparison.round(2)
    comparison_display = comparison.copy()
    for col in comparison.columns:
        comparison_display[col] = comparison_display[col].astype(str) + '%'
    
    # Display the table with dividing lines
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 120)  # Set display width
    
    # Calculate column widths
    model_width = 25
    col_widths = {col: max(len(col) + 2, 12) for col in comparison_display.columns}
    
    # Calculate total table width
    total_width = model_width + sum(col_widths.values()) + len(comparison_display.columns) + 3
    
    # Format table with dividing lines
    print("\n" + "-" * total_width)  # Top border
    
    # Header row
    header = f"| {'Model'.ljust(model_width)} |"
    for col in comparison_display.columns:
        header += f" {col.center(col_widths[col])} |"
    print(header)
    print("-" * total_width)  # Header separator line
    
    # Print each row with borders
    for model, row in comparison_display.iterrows():
        model_name = model[:model_width].ljust(model_width)  # Limit and pad model name
        line = f"| {model_name} |"
        for i, (col, value) in enumerate(row.items()):
            line += f" {value.rjust(col_widths[col])} |"
        print(line)
        print("-" * total_width)  # Row separator line
    
    print("\n📌 Best Overall Model: " + comparison.index[0])
    
    # Get timestamp for unique filenames
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    
    # Save comparison to CSV
    csv_file = os.path.join(comparison_folder, f"model_comparison_{timestamp}.csv")
    comparison.to_csv(csv_file)
    print(f"✅ Comparison data saved to '{csv_file}'")
    
    # Create bar chart
    plt.figure(figsize=(14, 10))
    comparison.plot(kind='bar')
    plt.title('Model Performance Comparison Across Datasets (%)')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout()
    comparison_chart = os.path.join(comparison_folder, f"all_models_comparison_{timestamp}.png")
    plt.savefig(comparison_chart, dpi=150)
    plt.close()
    
    print(f"✅ Comparison chart saved to '{comparison_chart}'")
    
    # Create a heatmap for easier visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(comparison, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title('Model Performance Comparison Heatmap (%)')
    plt.tight_layout()
    heatmap_file = os.path.join(comparison_folder, f"performance_heatmap_{timestamp}.png")
    plt.savefig(heatmap_file, dpi=150)
    plt.close()
    
    print(f"✅ Performance heatmap saved to '{heatmap_file}'")
    
    return comparison


if __name__ == "__main__":
    print("\n" + "✨ LOAN APPROVAL PREDICTION PROJECT ✨".center(80))
    print("A beginner-friendly machine learning tutorial".center(80) + "\n")
    dataset_paths = {
        "Original Loan Dataset": "datasets/loan_approval_data.csv",
        "Financial Balance Sheet": "datasets/Balance Sheet .xlsx",
        "German Credit Risk": "datasets/german_credit_data.csv",
        "Loan Approval Dataset 2": "datasets/loan_approval_dataset_2.csv"
    }
    all_results = {}
    for name,path in dataset_paths.items():
        try:
            print(F"\nProcessing dataset: {name} at {path}")
            results = process_dataset(path, name)
            if results:
                all_results[name] = results
        except Exception as e:
            print(F"Error processing {name}: {e}")
        
        if len(all_results) > 1:
            print("\n📊 Comparing model performances across datasets...")
            comparison = compare_models(all_results)
            print("\n📈 Model performance comparison completed!")
        else:
            print("\n⚠️ Not enough datasets processed to compare model performances yet.")
    print("\n✨ Project completed! Check the 'comparison_results' folder for model performance comparisons and visualizations.")