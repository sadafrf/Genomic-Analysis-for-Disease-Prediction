import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class GenomicAnalyzer:
    
    def __init__(self, data_file, heart_disease_snps, diabetes_snps, asthma_snps):
        self.data = pd.read_csv(data_file, sep='\t')
        self.heart_disease_snps = heart_disease_snps
        self.diabetes_snps = diabetes_snps
        self.asthma_snps = asthma_snps
        
    def preprocess_data(self):
        # Keep only the SNPs relevant to heart disease, diabetes, and asthma
        snps_of_interest = set(self.heart_disease_snps) | set(self.diabetes_snps) | set(self.asthma_snps)
        self.data = self.data[self.data['rsid'].isin(snps_of_interest)]
        
        # One-hot encode the genotype column
        one_hot = pd.get_dummies(self.data['genotype'])
        self.data = self.data.drop('genotype', axis=1)
        self.data = pd.concat([self.data, one_hot], axis=1)
        
    def analyze_risk(self):
        # Separate the data into heart disease, diabetes, and asthma SNPs
        hd_data = self.data[self.data['rsid'].isin(self.heart_disease_snps)]
        diabetes_data = self.data[self.data['rsid'].isin(self.diabetes_snps)]
        asthma_data = self.data[self.data['rsid'].isin(self.asthma_snps)]
        
        # Calculate the total risk score for each disease
        hd_risk = hd_data.iloc[:, 3:].sum(axis=1)
        diabetes_risk = diabetes_data.iloc[:, 3:].sum(axis=1)
        asthma_risk = asthma_data.iloc[:, 3:].sum(axis=1)
        
        # Calculate the percentage risk for each disease
        hd_risk_pct = (hd_risk / hd_data.shape[0]) * 100
        diabetes_risk_pct = (diabetes_risk / diabetes_data.shape[0]) * 100
        asthma_risk_pct = (asthma_risk / asthma_data.shape[0]) * 100
        
        # Print the risk percentages for each disease
        print(f"Risk of heart disease: {hd_risk_pct:.2f}%")
        print(f"Risk of diabetes: {diabetes_risk_pct:.2f}%")
        print(f"Risk of asthma: {asthma_risk_pct:.2f}%")
        
        # Combine the risks into a single dataframe
        risks = pd.DataFrame({
            'Disease': ['Heart Disease', 'Diabetes', 'Asthma'],
            'Risk Percentage': [hd_risk_pct, diabetes_risk_pct, asthma_risk_pct]
        })
        
        # Plot the risks as a bar chart
        sns.set_style('whitegrid')
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Disease', y='Risk Percentage', data=risks, palette='muted')
        plt.title('Risks of Heart Disease, Diabetes, and Asthma')
        plt.xlabel('Disease')
        plt.ylabel('Risk Percentage')
        plt.ylim(0, 100)
        plt.show()
        
        return risks
    
     def predict_disease(self, test_size=0.3, C=1):
        # Separate the data into heart disease, diabetes, and asthma SNPs
        hd_data = self.data[self.data['rsid'].isin(self.hd_snps)]
        diab_data = self.data[self.data['rsid'].isin(self.diab_snps)]
        asth_data = self.data[self.data['rsid'].isin(self.asth_snps)]

        # Prepare the data for prediction
        hd_X_train, hd_X_test, hd_y_train, hd_y_test = self.prepare_data(hd_data, test_size)
        diab_X_train, diab_X_test, diab_y_train, diab_y_test = self.prepare_data(diab_data, test_size)
        asth_X_train, asth_X_test, asth_y_train, asth_y_test = self.prepare_data(asth_data, test_size)

        # Train the models
        hd_model = self.train_model(hd_X_train, hd_y_train, C=C)
        diab_model = self.train_model(diab_X_train, diab_y_train, C=C)
        asth_model = self.train_model(asth_X_train, asth_y_train, C=C)

        # Make predictions on the test data
        hd_preds = hd_model.predict(hd_X_test)
        diab_preds = diab_model.predict(diab_X_test)
        asth_preds = asth_model.predict(asth_X_test)

        # Compute the accuracy of the models
        hd_acc = accuracy_score(hd_y_test, hd_preds)
        diab_acc = accuracy_score(diab_y_test, diab_preds)
        asth_acc = accuracy_score(asth_y_test, asth_preds)

        # Compute the risk of having each disease
        hd_risk = hd_model.predict_proba(self.X)[:, 1].mean()
        diab_risk = diab_model.predict_proba(self.X)[:, 1].mean()
        asth_risk = asth_model.predict_proba(self.X)[:, 1].mean()

        # Store the results in a dataframe
        results = pd.DataFrame({'Disease': ['Heart Disease', 'Diabetes', 'Asthma'],
                                'Accuracy': [hd_acc, diab_acc, asth_acc],
                                'Risk': [hd_risk, diab_risk, asth_risk]})

        return results

    results = pd.DataFrame({'Disease': ['Heart Disease', 'Diabetes', 'Asthma'],
                                'Accuracy': [hd_acc, diab_acc, asth_acc],
                                'Risk': [hd_risk, diab_risk, asth_risk]})

    # Plot the accuracy and risk for each disease
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(x='Disease', y='Accuracy', data=results, ax=ax[0])
    sns.barplot(x='Disease', y='Risk', data=

