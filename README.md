# Genomic-Analysis-for-Disease-Prediction
 This project aims to predict the likelihood of developing heart disease, diabetes, and asthma based on an individual's genomic data. Using machine learning techniques, it analyzes the SNPs associated with these diseases and provides risk percentages. The code includes preprocessing, training, testing, and evaluation steps.

## Data

The data used in this project is a CSV file containing genomic data of a single person from kaggle :https://www.kaggle.com/datasets/zusmani/mygenome/versions/4?resource=download . The file has the following columns: rsid, chromosome, position, and genotype. The rsid column contains the unique identifier for each SNP, the chromosome column contains the chromosome number where the SNP is located, the position column contains the position of the SNP on the chromosome, and the genotype column contains the individual's genotype for that SNP.
Also I got SNP ids associated with the heart disease, diabetes, and asthma from the following papers:

SNPs associated with asthma:

1.Gudbjartsson et al., "Sequence variants affecting eosinophil numbers associate with asthma and myocardial infarction" (2009)
2.Moffatt et al., "Genetic variants regulating ORMDL3 expression contribute to the risk of childhood asthma" (2007)
3.Torgerson et al., "Meta-analysis of genome-wide association studies of asthma in ethnically diverse North American populations" (2011)
4.Ferreira et al., "Genome-wide association analysis identifies 11 risk variants associated with the asthma with hay fever phenotype" (2014)

SNPs associated with heart disease:

1.Samani et al., "Genomewide association analysis of coronary artery disease" (2007)
2.Deloukas et al., "Large-scale association analysis identifies new risk loci for coronary artery disease" (2013)
3.Khera et al., "Genetic risk, adherence to a healthy lifestyle, and coronary disease" (2016)
4.Nikpay et al., "A comprehensive 1,000 Genomes-based genome-wide association meta-analysis of coronary artery disease" (2015)

SNPs associated with diabetes:

1.Zeggini et al., "Meta-analysis of genome-wide association data and large-scale replication identifies additional susceptibility loci for type 2 diabetes" (2008)
2.Saxena et al., "Genome-wide association analysis identifies loci for type 2 diabetes and triglyceride levels" (2010)
3.Morris et al., "Large-scale association analysis provides insights into the genetic architecture and pathophysiology of type 2 diabetes" (2012)
4.Mahajan et al., "Refining the accuracy of validated target identification through coding variant fine-mapping in type 2 diabetes" (2018)


## Preprocessing

The data is first preprocessed by separating the SNPs associated with heart disease, diabetes, and asthma. Each set of SNPs is then converted to a binary format where the presence of a risk allele is represented by 1 and the absence of a risk allele is represented by 0. The data is then split into training and testing sets for machine learning modeling.

## Machine Learning Modeling

Three machine learning models are trained on the binary SNP data for heart disease, diabetes, and asthma separately. Logistic regression is used as the machine learning algorithm for all three models. The models are trained on the training set and their performance is evaluated on the testing set.



## Dependencies

- pandas
- scikit-learn
- matplotlib
- seaborn
