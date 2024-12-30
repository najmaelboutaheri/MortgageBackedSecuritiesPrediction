# MortgageBackedSecuritiesPrediction
This repository contains a case study on mortgage trading, focusing on predicting loan delinquency and prepayment behavior.

## Architecture
![Architecture](https://github.com/user-attachments/assets/41202276-e456-4214-bb38-51e3b15667b5)

## Dataset Overview
The dataset includes features like `CreditScore`, `FirstPaymentDate`, `EverDelinquent`, and more. The goal is to predict delinquency (`EverDelinquent`) and, if delinquent, assess prepayment behavior.

### Key Features:
- **CreditScore**: Borrower’s credit score.
- **EverDelinquent**: Target variable for delinquency prediction.
- **Prepayment**: Target variable for regression after delinquency.

## Data Preprocessing
1. **Data Cleaning**:
   - Dropped irrelevant columns (`ProductType`, `LoanSeqNum`).
   - Handled missing values and outliers.
   - Removed duplicates and capped outliers.

2. **Feature Engineering**:
   - Created new features like `Credit_range`, `LTV_range`, `EMI`, and `Monthly_Income`.

3. **Data Encoding**:
   - Applied Label Encoding, One-Hot Encoding, Target Encoding, and Ordinal Encoding for categorical features.

## Feature Selection
- Used **Mutual Information (MI)** to rank features.
- Filtered features with MI scores below 0.02.

## Modeling
1. **Classification**:
   - **Random Forest** was used to predict `EverDelinquent`.
   - Evaluated using accuracy, precision, recall, and F1-score.

2. **Regression**:
   - **Linear Regression** was used to predict `prepayment` for delinquent loans.
   - Evaluated using MAE, MSE, and R2 score.

## Pipeline
- A custom pipeline integrates classification and regression tasks:
  1. Predict delinquency using Random Forest.
  2. Predict prepayment for delinquent loans using Linear Regression.

## Deployment
1. **Flask**:
   - RESTful API for predictions.
   - Check this repository : [link](https://github.com/najmaelboutaheri/MortgagePredictorApp)

## Contributing
Fork the repository.
1. Create a new feature branch.
2. Commit your changes.
3. Push to the branch.
4. Open a Pull Request.

## License
This project is licensed under the MIT License.
### Contact:
- **[Email](najma.elboutaheri@etu.uae.ac.ma)** 
- **[Linkedin profile](https://www.linkedin.com/in/najma-el-boutaheri-8185a1267/)** 
