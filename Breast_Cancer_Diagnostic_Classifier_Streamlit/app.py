import streamlit as st
import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
class App:
    def __init__(self):
        self.data = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.model = None
        self.best_params = None
    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
    def remove_low_importance_features(self):
        # Create Random Forest classifier
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Split the data into X and Y
        X = self.data.drop(columns=['diagnosis'])
        Y = self.data['diagnosis']

        # Train the Random Forest model
        rf_clf.fit(X, Y)

        # Get feature importances
        feature_importances = rf_clf.feature_importances_

        # Determine features with importance levels lower than the threshold
        threshold = 0.01
        low_importance_features = X.columns[feature_importances < threshold]

        # Remove unnecessary features from the dataset
        self.data.drop(columns=low_importance_features, inplace=True)
    def clean_data(self):
        # Map 'M' values in the 'diagnosis' column to 1 and 'B' values to 0
        self.data['diagnosis'] = self.data['diagnosis'].map({'M': 1, 'B': 0})

        # Remove low importance features
        self.remove_low_importance_features()
    def split_data(self):
        # Split the data into X and Y
        self.X = self.data.drop(columns=['diagnosis'])
        self.Y = self.data['diagnosis']

        # Split the data into X_train, X_test, Y_train, Y_test (80-20 ratio)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2,
                                                                                random_state=42, shuffle=True)
    def build_model(self, model_type):
        if model_type == 'KNN':
            # Create KNN model
            self.model = KNeighborsClassifier()

            # Define parameter grid for GridSearchCV
            param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}

            # Perform GridSearchCV
            grid_search = GridSearchCV(self.model, param_grid, cv=5)
            grid_search.fit(self.X_train, self.Y_train)

            # Store best parameters
            self.best_params = grid_search.best_params_

            # Train the model with the best parameters
            self.model = KNeighborsClassifier(**self.best_params)
            self.model.fit(self.X_train, self.Y_train)
        elif model_type == 'SVM':
            # Create SVM model
            self.model = SVC()

            # Define parameter grid for GridSearchCV
            param_grid = {'C': [1, 10], 'gamma': [0.1, 0.01], 'kernel': ['rbf', 'linear']}

            # Perform GridSearchCV
            grid_search = GridSearchCV(self.model, param_grid, cv=5, n_jobs=-1)
            grid_search.fit(self.X_train, self.Y_train)

            # Store best parameters
            self.best_params = grid_search.best_params_

            # Train the model with the best parameters
            self.model = SVC(**self.best_params)
            self.model.fit(self.X_train, self.Y_train)

        elif model_type == 'Naive Bayes':
            # Create Naive Bayes model
            self.model = GaussianNB()
            self.model.fit(self.X_train, self.Y_train)

    def evaluate_model(self):
        # Evaluate the model
        Y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.Y_test, Y_pred)
        precision = precision_score(self.Y_test, Y_pred)
        recall = recall_score(self.Y_test, Y_pred)
        f1 = f1_score(self.Y_test, Y_pred)
        cm = confusion_matrix(self.Y_test, Y_pred)

        # Show the results
        st.title("Model Evaluation")
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1-Score: {f1}")

        # Visualize the confusion matrix
        st.subheader("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', ax=ax)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        st.pyplot(fig)

    def task_1(self):
        task_1_expander = st.expander("Dataset Analysis Section", expanded=True)
        with task_1_expander:
            st.subheader(f"First 10 Rows: \n Total number of columns: {self.data.shape[1]}")
            st.dataframe(self.data.head(10))
            st.subheader("Columns:")
            columns_list = self.data.columns.tolist()
            for i, column in enumerate(columns_list):
                st.write(f"{i}: {column}")
    def task_2(self):
        task_2_expander = st.expander("Data Preprocessing Section", expanded=True)
        with task_2_expander:
            st.subheader(f"Last 10 Rows: \n Total number of columns: {self.data.shape[1]}")
            st.dataframe(self.data.tail(10))

            # Merge Malignant (M) and Benign (B) data
            malignant_data = self.data[self.data['diagnosis'] == 1]
            benign_data = self.data[self.data['diagnosis'] == 0]
            merged_data = pd.concat([malignant_data, benign_data])

            # Plot correlation matrix using Seaborn
            st.subheader("Correlation Matrix:")
            plt.figure(figsize=(12, 10))
            sns.scatterplot(data=merged_data, x='radius_mean', y='texture_mean', hue='diagnosis',
                            palette=['red', 'green'],
                            hue_order=[1, 0], markers=['o', 's'], style='diagnosis')
            plt.xlabel('radius_mean')
            plt.ylabel('texture_mean')

            # Customize legend manually
            handles, labels = plt.gca().get_legend_handles_labels()
            custom_labels = ['Malignant (Red)', 'Benign (Green)']
            plt.legend(handles, custom_labels, title='Diagnosis', loc='upper right', fontsize='small')

            st.pyplot(plt)

    def show_best_params(self, best_params_dict):
        if best_params_dict:
            st.subheader("Best Parameters:")
            for key, value in best_params_dict.items():
                st.write(f"- {key.replace('_', ' ').title()}: {value}")
    def run(self):
        st.title("Breast Cancer Diagnostic App")
        st.sidebar.title("Data Loading")
        file_path = st.sidebar.file_uploader("Please upload a CSV file", type=["csv"])

        if file_path is not None:
            self.load_data(file_path)
            self.task_1()
            self.clean_data()
            self.split_data()
            self.task_2()

            # Select classifier
            st.title("Model Implementation and Model Analysis \n\n")
            selected_classifier = st.sidebar.selectbox("Select Classifier:", ('KNN', 'SVM', 'Naive Bayes'), index=0)
            st.sidebar.write("Selected model: ", selected_classifier)

            # Provide information about the selected classifier
            st.title(f"-Selected Classifier: {selected_classifier}")
            if selected_classifier == 'KNN':
                st.write("K-Nearest Neighbors (KNN) uses k-nearest neighbors to determine the label of an example.")
            elif selected_classifier == 'SVM':
                st.write(
                    "Support Vector Machine (SVM) is a model used for linear and nonlinear classification and regression problems.")
            elif selected_classifier == 'Naive Bayes':
                st.write("Naive Bayes is an application of Bayes' theorem and is used for classification of examples.")

            self.build_model(selected_classifier)
            self.show_best_params(self.best_params)
            self.evaluate_model()
