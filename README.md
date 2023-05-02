# Genomic-Analysis-for-Disease-Prediction
 This project aims to predict the likelihood of developing heart disease, diabetes, and asthma based on an individual's genomic data. Using machine learning techniques, it analyzes the SNPs associated with these diseases and provides risk percentages. The code includes preprocessing, training, testing, and evaluation steps.

## Data

The data used in this project is a CSV file containing genomic data of a single person from kaggle :https://www.kaggle.com/datasets/zusmani/mygenome/versions/4?resource=download . The file has the following columns: rsid, chromosome, position, and genotype. The rsid column contains the unique identifier for each SNP, the chromosome column contains the chromosome number where the SNP is located, the position column contains the position of the SNP on the chromosome, and the genotype column contains the individual's genotype for that SNP.

## Preprocessing

The data is first preprocessed by separating the SNPs associated with heart disease, diabetes, and asthma. Each set of SNPs is then converted to a binary format where the presence of a risk allele is represented by 1 and the absence of a risk allele is represented by 0. The data is then split into training and testing sets for machine learning modeling.

## Machine Learning Modeling

Three machine learning models are trained on the binary SNP data for heart disease, diabetes, and asthma separately. Logistic regression is used as the machine learning algorithm for all three models. The models are trained on the training set and their performance is evaluated on the testing set.


## Conclusion

This project demonstrates the feasibility of using genomic data and machine learning algorithms to predict an individual's likelihood of having heart disease, diabetes, and asthma. The results show that the model for asthma has the highest accuracy and area under the ROC curve, indicating that the model is the most effective in predicting asthma risk. However, further research is needed to improve the accuracy and reliability of the models for heart disease and diabetes.

## Dependencies

- pandas
- scikit-learn
- matplotlib
- seaborn
