# MortgageBackedSecuritiesPrediction
This repository contains a case study on mortgage trading, focusing on predicting loan delinquency and prepayment behavior.

## Architecture
![Screen Recording 2025-01-01 203456 (3)](https://github.com/user-attachments/assets/d893f423-8962-48bb-a0f7-bb2610e54253)


## Dataset Overview
The dataset includes features like `CreditScore`, `FirstPaymentDate`, `EverDelinquent`, and more. The goal is to predict delinquency (`EverDelinquent`) and, if delinquent, assess prepayment behavior.

## Understanding the Dataset
This dataset contains various features related to mortgage loans, borrowers, and properties. The goal of this project is to build a classification model to predict whether a borrower will become delinquent on their loan (using the `EverDelinquent` feature). If the classification result indicates a delinquency (`EverDelinquent = 1`), the next step is to assess prepayment behavior.

<img width="566" alt="image" src="https://github.com/user-attachments/assets/62aad365-7d7a-4cd9-b11d-c81f216331f9" />

### Dataset Features:

- **CreditScore (int64)**: The borrower’s credit score at loan origination.
- **FirstPaymentDate (int64)**: The date of the first loan payment made by the borrower.
- **FirstTimeHomebuyer (object)**: Indicates whether the borrower is a first-time homebuyer (Yes/No).
- **MaturityDate (int64)**: The date when the loan is expected to be fully repaid.
- **MSA (object)**: Metropolitan Statistical Area code, representing the geographical area of the property.
- **MIP (int64)**: Mortgage Insurance Premium to protect the lender against default.
- **Units (int64)**: Number of units in the property (e.g., single-family, duplex, etc.).
- **Occupancy (object)**: Describes how the property is occupied (owner-occupied, second home, or investment).
- **OCLTV (int64)**: Original Combined Loan-to-Value ratio, comparing all secured loans to the property value at origination.
- **DTI (int64)**: Debt-to-Income ratio, representing the borrower's debt payments relative to their income.
- **OrigUPB (int64)**: Original Unpaid Principal Balance at loan origination.
- **LTV (int64)**: Loan-to-Value ratio, representing the loan amount as a percentage of the property value.
- **OrigInterestRate (float64)**: The original interest rate on the mortgage.
- **Channel (object)**: The method through which the loan was originated (e.g., Retail, Broker).
- **PPM (object)**: Indicates if there is a prepayment penalty on the loan.
- **ProductType (object)**: The type of loan product (e.g., fixed-rate, adjustable-rate).
- **PropertyState (object)**: The state where the property is located.
- **PropertyType (object)**: The type of property (e.g., single-family, condominium).
- **PostalCode (object)**: The postal code of the property's location.
- **LoanSeqNum (object)**: A unique identifier for the loan.
- **LoanPurpose (object)**: The purpose of the loan (e.g., purchase, refinance).
- **OrigLoanTerm (int64)**: The term of the loan in months.
- **NumBorrowers (object)**: The number of borrowers involved in the loan.
- **SellerName (object)**: The name of the entity that sold the loan.
- **ServicerName (object)**: The entity responsible for servicing the loan.
- **EverDelinquent (int64)**: The target variable for the classification task, indicating if the borrower was ever delinquent (1 for yes, 0 for no).
- **MonthsDelinquent (int64)**: The number of months the borrower was delinquent.
- **MonthsInRepayment (int64)**: The total number of months the borrower has been repaying the loan.

### Key Features:
- **CreditScore**: Borrower’s credit score.
- **EverDelinquent**: Target variable for delinquency prediction.
- **Prepayment**: Target variable for regression after delinquency.

## Data Preprocessing

Data cleaning is an essential step in the data preprocessing pipeline. This step ensures that the dataset is free from inconsistencies, missing values, and outliers that could negatively impact model performance.

<img width="577" alt="image" src="https://github.com/user-attachments/assets/59ab55d2-ba57-4281-bf83-1af823126ac8" />

1. **Data Cleaning**:
   
   **-->** Dropped irrelevant columns (`ProductType`, `LoanSeqNum`).
   
   **-->** Handled missing values and outliers.
   
   **-->** Removed duplicates and capped outliers.

2. **Exploratory Data Analysis**:

**-->** Temporal data analysis: this figure shows the Average Original Interest Rate Over Time.
   
![Capture d'écran 2024-12-13 123127](https://github.com/user-attachments/assets/319db685-e681-4d06-9685-5ea8fd7d90c6)
   
**-->** Correlation heatmap:
Heatmap of correlations highlights strong positive correlations between LTV and OCLTV and strong negative correlations between OrigInterestRate and CreditScore.

![Capture d'écran 2024-12-13 123318](https://github.com/user-attachments/assets/742d1a94-a71f-46bf-a301-22b9a82ad43b)


3. **Feature Engineering**:
   
   **-->** Created new features like `Credit_range`, `LTV_range`, `EMI`, and `Monthly_Income` To enhance the dataset and provide more informative inputs to machine learning models, several new features were created based on the existing data.

4. **Data Encoding**:
   
   **-->** Applied Label Encoding, One-Hot Encoding, Target Encoding, and Ordinal Encoding for categorical features.

## Feature Selection

<img width="577" alt="image" src="https://github.com/user-attachments/assets/a500408f-c54c-4dc1-96bb-89012a2f276c" />

1. Used **Mutual Information (MI)** to rank features.
2. Filtered features with MI scores below 0.02.

## Modeling

<img width="577" alt="image" src="https://github.com/user-attachments/assets/ea7c10cc-3381-4b44-a037-c8b55d395f0a" />

1. **Classification**:
   
   **-->** **Naive Bayes** was used to predict `EverDelinquent`.

   **-->** Evaluated using accuracy, precision, recall, and F1-score.

3. **Regression**:
   
   **-->** **Linear Regression** was used to predict `prepayment` for delinquent loans.
   
   **-->** Evaluated using MAE, MSE, and R2 score.
     
![Capture d'écran 2024-12-10 181131](https://github.com/user-attachments/assets/5b8d70be-b606-4416-8d2a-d1b6288122a9)

After evaluating the models on accuracy and F1-score, we found that **Naive bayes** outperformed the other models due to its ability to handle imbalanced datasets and its robustness to overfitting. Therefore, Naive bayes was selected as the final model for classification and we selected Linear Regression for prediction task as the final model due to its simplicity and slightly better predictive power on our dataset.

## Pipeline

<img width="630" alt="image" src="https://github.com/user-attachments/assets/8fbb2d2b-aa94-47e1-8293-6a832dc2d1cd" />

- A custom pipeline integrates classification and regression tasks:
  + Predict delinquency using Naive Bayes.
  + Predict prepayment for delinquent loans using Linear Regression.

## Deployment
1. **Flask**:
   + RESTful API for predictions.
   + Check this repository: [link](https://github.com/najmaelboutaheri/MortgagePredictorApp)
     
## Conclusion

This project demonstrates a systematic approach to predicting loan delinquency and prepayment behavior using a combination of data preprocessing, feature engineering, and machine learning techniques. By leveraging Mutual Information for feature selection and integrating both classification and regression tasks, we achieved insightful results that can be applied in real-world mortgage trading scenarios. 

## Future work could include:

+ Enhancing the pipeline with more advanced models or ensemble methods.
+ Incorporating real-time data to improve prediction accuracy.
+ Expanding deployment options for broader accessibility and scalability.

## Contributing
+ Fork the repository.
+ Create a new feature branch.
+ Commit your changes.
+ Push to the branch.
+ Open a Pull Request.

## License
This project is licensed under the MIT License.

### Contact:
- **[Email](najma.elboutaheri@etu.uae.ac.ma)** 
- **[LinkedIn profile](https://www.linkedin.com/in/najma-el-boutaheri-8185a1267/)**

Thank you for exploring this case study. Feedback and contributions are always welcome!
