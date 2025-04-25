import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import accuracy_score

# Load your DataFrame, assuming 'target_column' is the column you want to predict
def loo_grid_search_k_for_knn_classifier(df, target_column, k_range=(1, 20)):
    # Split the data into features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Specify which columns are categorical
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create a ColumnTransformer to apply One-Hot Encoding to categorical features
    transformers = [
        ('categorical', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
    ]

    preprocessor = ColumnTransformer(transformers, remainder='passthrough')

    # Create a K-Neighbors Classifier
    knn_classifier = KNeighborsClassifier()

    # Define a parameter grid for grid search with k values
    param_grid = {
        'n_neighbors': range(k_range[0], k_range[1] + 1)
    }

    # Create a GridSearchCV object with Leave-One-Out cross-validation
    loo = LeaveOneOut()
    grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid,
                               scoring='accuracy', cv=loo)

    # Create a Pipeline that first applies one-hot encoding, then K-Neighbors Classifier
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', grid_search)])

    # Fit the grid search to find the best k on the entire dataset
    pipeline.fit(X, y)

    # Get the best k and the best estimator
    best_k = pipeline.named_steps['classifier'].best_params_['n_neighbors']
    best_estimator = pipeline.named_steps['classifier'].best_estimator_

    return best_k, best_estimator

# Example usage
if __name__ == '__main__':
    # Load your DataFrame (replace 'your_dataframe.csv' with your file)
    df = pd.read_csv('data/agaricus-lepiota.csv', skiprows=[1, 2], header=0)

    # Specify the target column
    target_column = 'Class'

    k_min = 1
    k_max = 20
    best_k, best_estimator = loo_grid_search_k_for_knn_classifier(df, target_column, k_range=(k_min, k_max))

    print("Best k for K-Neighbors Classifier:", best_k)
    print("Best Accuracy Score:", accuracy_score(df[target_column], best_estimator.predict(df.drop(columns=[target_column]))))
