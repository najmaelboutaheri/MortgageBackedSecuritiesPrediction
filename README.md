# MortgageBackedSecuritiesPrediction
This repository contains the code and resources for a comprehensive case study on mortgage trading, designed to help Industrial/Organizational Economists understand the financial system, sharpen data modeling,  and financial analysis skills, and experience the dynamic environment of a mortgage trading desk.
## Understanding the Dataset

This dataset contains various features related to mortgage loans, borrowers, and properties. The goal of this project is to build a classification model to predict whether a borrower will become delinquent on their loan (using the `EverDelinquent` feature). If the classification result indicates a delinquency (`EverDelinquent = 1`), the next step is to assess prepayment behavior.

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

### Initial Goal

The first goal is to build a classification model to predict whether a borrower will become delinquent using the `EverDelinquent` feature. If the model predicts a delinquency (`EverDelinquent = 1`), further steps will be taken to evaluate prepayment behavior.

## Data Preprocessing

### 1. General Information

This section outlines the initial steps taken to set up and explore the dataset before any preprocessing is done.

1. **Import Necessary Libraries**:
   We import essential Python libraries required for data manipulation, analysis, and visualization:
   - **Pandas** for data handling.
   - **NumPy** for numerical operations.
   - **Matplotlib** for plotting graphs.
   - **Seaborn** for statistical data visualization.

2. **Load the Dataset**:
   The dataset is loaded from a CSV file into a Pandas DataFrame. This is the foundation for all subsequent data analysis and preprocessing tasks.

3. **Examine the Dataset**:
   - **View Initial Rows**: Display the first few rows of the dataset to get an overview of the data and its structure.
   - **Check Dataset Shape**: Determine the number of rows and columns to understand the dataset's size.
   - **Review Data Types**: Inspect the data types of each column to identify any type-related issues.
   - **Compute Summary Statistics**: Analyze summary statistics for numerical features to understand their distribution and identify potential anomalies or outliers.

    
Summary statistics for numerical features:
         CreditScore  FirstPaymentDate   MaturityDate            MIP  \
count  291451.000000     291451.000000  291451.000000  291451.000000   
mean      708.936991     199904.760553  202902.367043       9.216685   
std        68.408791         13.821228      16.090508      12.648273   
min         0.000000     199901.000000  202402.000000       0.000000   
25%       675.000000     199903.000000  202902.000000       0.000000   
50%       717.000000     199904.000000  202903.000000       0.000000   
75%       753.000000     199905.000000  202904.000000      25.000000   
max       849.000000     201303.000000  204302.000000      55.000000   

               Units          OCLTV            DTI        OrigUPB  \
count  291451.000000  291451.000000  291451.000000  291451.000000   
mean        1.026022      76.960529      30.033453  124940.387921   
std         0.202986      15.236154      13.203974   53657.440624   
min         0.000000       0.000000       0.000000    8000.000000   
25%         1.000000      70.000000      22.000000   83000.000000   
50%         1.000000      80.000000      31.000000  117000.000000   
75%         1.000000      90.000000      39.000000  160000.000000   
max         4.000000     110.000000      65.000000  497000.000000   

                 LTV  OrigInterestRate   OrigLoanTerm  EverDelinquent  \
count  291451.000000     291451.000000  291451.000000   291451.000000   
mean       76.937986          6.926547     359.835375        0.197848   
std        15.225298          0.343541       2.286939        0.398378   
min         0.000000          4.000000     301.000000        0.000000   
25%        70.000000          6.750000     360.000000        0.000000   
50%        80.000000          6.875000     360.000000        0.000000   
75%        90.000000          7.125000     360.000000        0.000000   
max       103.000000         12.350000     361.000000        1.000000   

       MonthsDelinquent  MonthsInRepayment  
count     291451.000000      291451.000000  
mean           1.837729          60.805291  
std            8.809392          46.142599  
min            0.000000           1.000000  
25%            0.000000          34.000000  
50%            0.000000          48.000000  
75%            0.000000          63.000000  
max          180.000000         212.000000  

These steps ensure that we have a good grasp of the dataset's structure and content, which is crucial for effective data preprocessing and analysis.

### 2. Data Cleaning

Data cleaning is an essential step in the data preprocessing pipeline. This step ensures that the dataset is free from inconsistencies, missing values, and outliers that could negatively impact model performance.

#### a. Dropping Columns with Irrelevant or Constant Values
- **Unique values check**: We first inspect the dataset for columns with a single unique value or irrelevant data.
- **Dropped columns**: 
  - `ProductType`: This column has only one unique value, so it was dropped as it does not provide any meaningful variation for modeling.
  - `LoanSeqNum`: A unique identifier for each loan, which does not add any predictive value, was also dropped.

#### b. Handling Missing Values
- **Replacing placeholder values**: The dataset contained some placeholder values ('X') which were replaced with `NaN` to be handled correctly during the imputation process.
- **Column conversions**: Several columns such as `PostalCode`, `NumBorrowers`, and `MSA` were converted to numeric types, forcing any invalid entries to `NaN` where necessary.
- **Missing value imputation**:
  - **Numerical columns** (`PostalCode`, `MSA`, `NumBorrowers`): Missing values were imputed using the **mean** strategy.
  - **Categorical columns** (`FirstTimeHomebuyer`, `PPM`, `SellerName`): Missing values were imputed using the **most frequent** strategy.

#### c. Identifying and Removing Duplicates
- **Duplicate row detection**: The dataset was checked for duplicate rows. 
- **Duplicates removal**: Any duplicate rows found were dropped to ensure no redundancy in the data.
  
#### d. Handling Zero Values in Important Features
- **DTI (Debt-to-Income Ratio)**: 
  - Rows where the `DTI` value was zero were identified and examined. These zero values were considered invalid and replaced with `NaN`.
  - To fill the missing values in `DTI`, the **median** strategy was used to prevent the influence of outliers.

#### e. Outlier Capping
Outliers in numerical features can distort the distribution of the data and affect model performance. To mitigate this, a **capping** strategy was applied:
- **Outlier capping**: We applied a method to cap the outliers for all numerical columns by setting values below the 1st percentile and above the 99th percentile to the respective boundary values.

### 2. Final Cleaned Dataset Overview
After data cleaning, the dataset was left with no missing values, no duplicate rows, and all outliers were capped to prevent extreme values from skewing the results. The following summarizes the cleaned dataset:
- The cleaned dataset now contains X rows and Y columns.
- All numerical columns have been handled for missing values and outliers, and categorical columns have been imputed appropriately.
  
The cleaned dataset is now ready for further steps such as feature engineering and model building.

Following the data cleaning process, we proceed with various analyses to understand the distribution, relationships, and interactions among features in the dataset.

### 3. Univariate Data Analysis

Univariate analysis involves examining each feature independently to understand its distribution and summary statistics.

#### a. Numerical Features
- **Objective**: Analyze the distribution of each numerical feature to detect patterns, skewness, or outliers.
- **Approach**: 
  - **Histograms**: Plotted to visualize the distribution of values for each numerical feature.
  - **Kernel Density Plots**: Used to estimate the probability density function of the data, providing a smoother alternative to histograms.
  - **Summary Statistics**: Key statistics such as mean, median, standard deviation, and range are computed for each feature to get a better understanding of the central tendency and spread.
  
  Example visualizations:
  - **Histogram of `CreditScore`**: Displays the frequency distribution of credit scores in the dataset.
  - **KDE of `OrigInterestRate`**: Smoother representation of the interest rate distribution, highlighting skewness.

#### b. Categorical Features
- **Objective**: Investigate the frequency distribution of categorical variables to understand their levels and balance.
- **Approach**: 
  - **Bar plots**: Display the count of each category in the dataset for features like `PropertyType`, `Channel`, etc.
  - **Pie charts**: Used for features with fewer categories, providing a visual breakdown of the proportions.

  Example visualizations:
  - **Bar plot of `FirstTimeHomebuyer`**: Shows the distribution between first-time homebuyers and non-first-time buyers.
  - **Pie chart of `Occupancy`**: Proportions of different occupancy types such as owner-occupied, rental, etc.

### 4. Bivariate Data Analysis

Bivariate analysis helps to explore the relationships between pairs of features, providing insights into how two variables interact.

#### a. Numerical vs Numerical
- **Objective**: Examine the relationship between pairs of numerical features to detect any linear or non-linear patterns.
- **Approach**:
  - **Scatter plots**: Visualize the relationship between two numerical features. For instance, the relationship between `OCLTV` and `DTI`.
  - **Correlation coefficients**: Compute Pearson or Spearman correlations to quantify the strength and direction of relationships between pairs of features.
  
  Example:
  - **Scatter plot of `LTV` vs `OrigInterestRate`**: A visual depiction of the relationship between loan-to-value ratios and interest rates.
  - **Correlation matrix**: Displays the correlation coefficients between all numerical features.

#### b. Numerical vs Categorical
- **Objective**: Explore how numerical features vary across different categories.
- **Approach**:
  - **Box plots**: Display the distribution of numerical features grouped by categorical variables (e.g., `OrigInterestRate` distribution across `LoanPurpose` categories).
  - **Violin plots**: Combine box plot and KDE to show both the distribution and summary statistics.
  
  Example:
  - **Box plot of `CreditScore` vs `FirstTimeHomebuyer`**: Visualizes how credit scores differ between first-time and non-first-time homebuyers.
  - **Violin plot of `OrigInterestRate` vs `Occupancy`**: Provides a richer view of how interest rates vary by occupancy type.

### 5. Multivariate Data Analysis

Multivariate analysis investigates the relationships and interactions between more than two features simultaneously. This is useful for detecting deeper patterns and dependencies in the data.

#### a. Correlation Structure
- **Objective**: Visualize the correlation between all numerical features at once to detect highly correlated features that might need attention (e.g., multicollinearity).
- **Approach**:
  - **Heatmap of Correlation Matrix**: A heatmap is used to visualize the pairwise correlations between numerical features, with color intensity representing the strength of correlation.

  Example:
  - **Heatmap of correlations**: Highlights strong positive correlations between `LTV` and `OCLTV` and strong negative correlations between `OrigInterestRate` and `CreditScore`.

#### b. Dimensionality Reduction
- **Objective**: Reduce the dimensionality of the dataset to uncover important combinations of features, simplify the data, and reduce redundancy.
- **Approach**:
  - **Principal Component Analysis (PCA)**: PCA is applied to the numerical features to capture the most important directions of variance in the data, allowing us to reduce the number of dimensions while preserving as much information as possible.

  Example:
  - **PCA components plot**: A 2D plot showing the first two principal components, highlighting clusters or separations in the dataset based on feature combinations.

#### c. Interactions Between Features
- **Objective**: Explore the complex interactions between multiple features simultaneously.
- **Approach**:
  - **3D plots**: Visualize interactions between three numerical variables. For example, exploring how `CreditScore`, `DTI`, and `OrigInterestRate` interact.
  - **Parallel coordinates plot**: A powerful tool to visualize the interactions between several features at once, helping to detect patterns or clusters.

  Example:
  - **3D plot of `CreditScore`, `OrigInterestRate`, and `DTI`**: Helps visualize how interest rates change with credit scores and DTI ratios.
  - **Parallel coordinates plot**: Highlights interactions across multiple numerical features in the dataset, helping to detect potential clusters or interesting patterns.

### Conclusion

By performing a thorough analysis at univariate, bivariate, and multivariate levels, we have gained a deep understanding of the structure of the dataset and the relationships between features. These insights will guide us in the subsequent steps of feature engineering, model building, and evaluation.

## Feature Engineering

### 1. Feature Extraction using Correlation Matrix
Before creating new features, we begin by analyzing the existing relationships between numerical variables using a correlation matrix. This helps us identify highly correlated features and select the most impactful ones for feature engineering.
the LTV and LCTV are the features highly correlated.

1. **Correlation Matrix**:
   - A correlation matrix was calculated to identify strong relationships between numerical features.
   - Features with high correlation values (e.g., greater than 0.9) are considered for removal or transformation to avoid redundancy and multicollinearity in the model.

### 2. Creation of New Features
To enhance the dataset and provide more informative inputs to machine learning models, several new features were created based on the existing data. Below are the steps for feature engineering:

1. **Credit Range (Credit_range)**:
   - The `CreditScore` column is used to categorize credit scores into four categories: 'Poor', 'Fair', 'Good', and 'Excellent'. This helps transform continuous credit scores into a more interpretable categorical feature.
   - Logic:
     - 'Poor' for scores less than 650
     - 'Fair' for scores between 650 and 700
     - 'Good' for scores between 700 and 750
     - 'Excellent' for scores above 750

2. **LTV Range (LTV_range)**:
   - The `LTV` (Loan-to-Value) column is categorized into three ranges: 'Low', 'Medium', and 'High', which allows better grouping of loans based on their risk.
   - Logic:
     - 'Low' for values below 25
     - 'Medium' for values between 25 and 50
     - 'High' for values above 50

3. **Repayment Range (Repay_range)**:
   - Based on the `MonthsInRepayment` column, the number of months a borrower has been repaying their loan is categorized into five different ranges: '0-4 years', '4-8 years', '8-12 years', '12-16 years', and '16+ years'.
   - This helps in understanding the repayment period and its relation to other factors.

4. **First Time Homebuyer (isFirstTime)**:
   - A new binary feature called `isFirstTime` is created directly from the `FirstTimeHomebuyer` column to indicate whether the borrower is purchasing their first home.

5. **Equated Monthly Installment (EMI)**:
   - The Equated Monthly Installment (EMI) is calculated based on the original loan balance (`OrigUPB`), interest rate (`OrigInterestRate`), and loan term (`OrigLoanTerm`). This feature provides insight into the borrower’s monthly payment burden.

6. **Total Payment**:
   - Using the calculated EMI, the total payment over the loan term is computed. This feature is crucial to understanding the total amount a borrower would have paid by the end of the loan term.

7. **Interest Amount**:
   - The total interest amount paid over the loan term is derived by subtracting the original loan balance (`OrigUPB`) from the total payment. This provides insight into the cost of borrowing for each loan.

8. **Monthly Income**:
   - The borrower’s estimated monthly income is calculated using the Debt-to-Income ratio (`DTI`) and EMI. This allows us to understand the borrower’s ability to handle loan repayments based on their income.

9. **Current Principal (cur_principal)**:
   - The current loan principal is computed based on the number of months in repayment, initial loan amount, and interest rate. This feature helps track how much of the loan principal remains to be paid.

10. **Prepayment**:
   - The `prepayment` feature estimates the borrower’s ability to make additional payments toward the loan. This is based on their income and debt-to-income ratio. Prepayments can significantly reduce the loan's interest burden over time.

Each of these features is designed to improve the predictive power of the model by incorporating additional borrower behavior and financial insights.

### 3. Verifying New Features
After creating the new features, the following checks were performed:
   - **Consistency**: Ensuring that the newly engineered features make logical sense and align with the underlying data.
   - **Summary Statistics**: Descriptive statistics are computed to verify the distribution and ranges of the new features.
   - **Initial Visualizations**: Basic plots and summaries (e.g., histograms, box plots) were created to visualize these new features and their potential impact on the analysis.

## Data Encoding

In this project, various encoding techniques were applied to handle categorical features in the dataset. Different encoding methods were used based on the type and cardinality of the categorical variables.

### 1. Label Encoding
**Label Encoding** was applied to binary and ordinal features. Label encoding assigns a unique integer value to each category, maintaining the ordinal relationship if one exists.

- **PPM**: This binary feature (Prepayment Penalty Mortgage) was label-encoded, assigning `0` or `1` to its values.

### 2. One-Hot Encoding
**One-Hot Encoding** was used for nominal categorical variables with fewer unique values. This method creates new binary columns for each category of a feature.

- The following columns were one-hot encoded for both regression (`X_reg`) and classification (`X_class`) tasks:
  - **Occupancy**: Categories representing the occupancy status of the property.
  - **Channel**: Channels through which the loan was obtained.
  - **PropertyType**: The type of property being financed (e.g., single-family home, multi-family home).
  - **LoanPurpose**: The purpose of the loan (e.g., purchase, refinance).

After applying one-hot encoding, each category of these columns is represented by a new column, where a value of `1` indicates the presence of the category, and `0` indicates its absence.

### 3. Target Encoding
For high-cardinality categorical features, **Target Encoding** was employed. This encoding technique replaces each category with the average value of the target variable for that category.

- **PropertyState**: This feature has a large number of unique values (U.S. states), and target encoding was applied to reduce the dimensionality while preserving the relationship between the categories and the target variable (`Y_class` for the classification task).

### 4. Ordinal Encoding
For ordinal features, where the categories have an inherent order, **Ordinal Encoding** was used. In this method, each category is assigned a numerical value based on its order.

- The following features were ordinal-encoded:
  - **Credit_range**: Categories based on credit score (`Poor`, `Fair`, `Good`, `Excellent`).
  - **LTV_range**: Categories based on the loan-to-value ratio (`Low`, `Medium`, `High`).
  - **Repay_range**: Categories based on the number of years in repayment (`0-4 years`, `4-8 years`, `8-12 years`, `12-16 years`, `16+ years`).

These ordinal encodings ensure that the machine learning algorithms can understand the inherent ranking of the categories.

### 5. Verifying Encodings
After applying all encoding techniques, the transformed datasets (`X_reg` for regression and `X_class` for classification) were verified by displaying the first few rows. This step ensures that the encodings were applied correctly and that the categorical variables are now in a format suitable for machine learning models.

## Feature Selection Using Mutual Information (MI)

Mutual Information (MI) is a measure of the dependency between variables. It quantifies how much information the presence or absence of one feature gives about the target variable. In this project, MI was used for both classification and regression tasks to evaluate the importance of features.

### 1. Mutual Information for Classification
For the classification task, the **`mutual_info_classif`** function from the `sklearn` library was used to compute the mutual information between the features and the target variable (`EverDelinquent`). This helps us understand which features are most informative for predicting whether a loan will become delinquent.

### 2. Mutual Information for Regression
Similarly, for the regression task, the **`mutual_info_regression`** function was applied to assess the relationship between the features and the target variable (`prepayment`). This provides insights into which features are most relevant for predicting prepayment amounts.

### 3. Average Mutual Information
To compare feature importance across both classification and regression tasks, the mutual information scores were averaged. This average helps identify features that are important for both tasks. The results were stored in a DataFrame, showing mutual information for each feature.

### 4. Feature Importance Ranking
The features were ranked based on their average mutual information score, and a bar plot was created to visualize the importance of the top features. This provides a clear understanding of which features contribute most to both tasks.

### 5. Visualization
A bar plot was generated to visualize the features based on their average mutual information score:

```python
plt.figure(figsize=(10, 6))
sns.barplot(x='Average_MI', y='Feature', data=mi_df_sorted)
plt.title('Feature Importance Based on Average Mutual Information')
plt.show()
```


### Explanation:

- **Mutual Information for Classification** explains how MI was used to understand the relationship between features and the classification target (`EverDelinquent`).
- **Mutual Information for Regression** covers the application of MI for regression with the target variable (`prepayment`).
- **Average Mutual Information** explains the calculation of the average mutual information score across both tasks to rank features.
- **Feature Importance Ranking** describes the sorting of features by their importance.
- **Visualization** and the corresponding plot show how MI scores were visually represented for better understanding.

This section explains the importance of each feature in a simple yet informative way, enhancing your README file with the MI feature selection process.

### 6. Filtering Features Based on Mutual Information

After calculating the mutual information (MI) for each feature, we filtered out the features with an **Average Mutual Information (Average_MI)** score below a defined threshold of **0.02**. This step ensures that only the most informative features are retained for further modeling, improving the efficiency of the model.

### 7. Feature Selection Process

The filtering process included the following steps:

1. **Setting a Threshold**: Features with an Average_MI score lower than **0.02** were considered less informative and were filtered out from the feature set.

    ```python
    mi_df_filtered = mi_df_sorted[mi_df_sorted['Average_MI'] >= 0.02]
    important_features = mi_df_filtered['Feature'].tolist()
    ```

2. **Removing Specific Features**: After filtering based on MI, some features that were deemed unnecessary or less relevant to the project were manually removed from the final list. This additional step helps eliminate features that might still be redundant or not critical to the analysis, even if their MI scores passed the threshold.

    ```python
    features_to_remove = ['Channel_T', 'OrigLoanTerm', 'Channel_R', 'PropertyType_SF', 'OCLTV']
    important_features = [feature for feature in important_features if feature not in features_to_remove]
    ```

3. **Final List of Important Features**: The remaining features represent the most informative and relevant set for both classification and regression tasks.

    ```python
    print("Important Features (Average_MI >= 0.02):")
    print(important_features)
    ```

This approach helps in dimensionality reduction by retaining only the most meaningful features, which ultimately improves the model's performance and reduces overfitting.

### 8. Conclusion of Feature Selection

By applying mutual information and filtering out less important features, the dataset was streamlined to contain only the most relevant columns. The thresholding and manual removal of certain features contributed to a more efficient and interpretable model, setting the stage for better prediction results.

### 9. Data Scaling and Normalization

Before proceeding with model building, data scaling and normalization were performed to ensure all features are on a similar scale. This is crucial as many machine learning algorithms (especially those based on distances) perform better when the input data is standardized or normalized.

- **Normalization**: Some features were normalized to scale values between 0 and 1, which helps in making the features comparable without altering their distributions.
- **Standardization**: Standardization was applied to features that benefit from being centered around a mean of 0 and a standard deviation of 1. This technique is particularly useful for algorithms like logistic regression, SVMs, and neural networks.

### 10. Data Modeling

We built two sets of models: one for classification (predicting **EverDelinquent**) and one for regression (predicting **prepayment**).

#### 10.1 Classification Models

For the classification task, we used the following algorithms:

1. **Random Forest**: An ensemble learning method that constructs multiple decision trees and aggregates their results for better predictive performance.
2. **Naive Bayes**: A probabilistic classifier based on Bayes' Theorem, effective for categorical data.
3. **Decision Tree**: A tree-based model that splits the dataset into subsets based on feature values, leading to predictions based on majority voting in terminal nodes.

After evaluating the models on accuracy and F1-score, we found that **Random Forest** outperformed the other models due to its ability to handle imbalanced datasets and its robustness to overfitting. Therefore, Random Forest was selected as the final model for classification.

#### 10.2 Regression Models

For the regression task (predicting **prepayment**), we applied the following models:

1. **Linear Regression**: A straightforward approach that models the relationship between the target and features as a linear equation.
2. **Ridge Regression**: An extension of linear regression that adds regularization to prevent overfitting, especially in cases where multicollinearity exists between features.

After comparing the performance using metrics such as RMSE (Root Mean Square Error), we selected **Linear Regression** as the final model due to its simplicity and slightly better predictive power on our dataset.

### 12. Pipeline for Classification and Regression

In this project, we implemented a custom pipeline that handles both classification (predicting **EverDelinquent**) and regression (predicting **prepayment**) tasks sequentially. The pipeline is designed to use the output of the classification model to filter data for the regression model.

#### 12.1 Overview of the Pipeline

The pipeline integrates two stages:

1. **Classification Stage**:
   - A **Random Forest Classifier** is trained to predict whether a loan will experience delinquency (i.e., **EverDelinquent**).
   - This classification model is applied to the entire dataset, and the predictions for delinquency are used to filter the data for the regression task.

2. **Regression Stage**:
   - For loans predicted to be delinquent (i.e., **EverDelinquent == 1**), a **Linear Regression** model is used to predict the **prepayment** values.
   - Only the instances that are predicted as delinquent by the classifier are included in the regression model.

#### 12.2 Data Splitting and Training

- The dataset is split into training and testing sets for both tasks.
- The classification model is trained first, followed by the regression model applied to the filtered data.

#### 12.3 Evaluation

- **Classification Model**:
   - We evaluated the classification model using key metrics such as **accuracy**, **precision**, **recall**, and **F1-score**. The **Random Forest Classifier** provided good performance for this task.
   
- **Regression Model**:
   - The regression model was evaluated using **mean absolute error (MAE)**, **mean squared error (MSE)**, and **R-squared (R2)** score. For the regression task, the **Linear Regression** model gave reliable results.

#### 12.4 Pipeline Serialization

The entire pipeline is serialized using **joblib**, allowing the model to be saved and reloaded for future predictions. The pipeline ensures that both classification and regression tasks are executed seamlessly in a combined flow.

## 13. Deployment

This project includes deployment strategies using **Flask** and **Django** to facilitate machine learning predictions through web interfaces.

### 13.1 Flask Deployment

The Flask application acts as a RESTful API service, which handles machine learning predictions by processing incoming feature data.

#### Flask API Details

- **API Endpoint**: `/predict`
- **HTTP Method**: POST
- **Input**: JSON payload containing feature data
- **Output**: JSON response with classification and regression predictions

**Features**:
- The Flask API is designed to accept feature data in JSON format. The features must match the expected schema used during model training.
- The API uses a pre-trained pipeline which includes both a classification model and a regression model. The classification model predicts categorical outcomes, while the regression model provides continuous predictions.

**Deployment Steps**:
1. **Install Flask**: Ensure Flask is installed using `pip install flask`.
2. **Load Models**: The pipeline and scaler must be pre-loaded from saved files (`combined_pipeline1.pkl` and `scaler2.pkl`).
3. **Run Flask Server**: Start the Flask application using `python app.py`. The server will run locally and listen on port 5000 by default.
4. **Testing**: Use tools like `Postman` or `curl` to test the API endpoint by sending POST requests with feature data.

### 13.2 Django Deployment

The Django application provides a web-based interface for users to interact with the machine learning models. It integrates with the Flask API to obtain predictions and display results.

#### Django Application Details

- **Endpoint**: `/predict/` (via POST request)
- **Form Inputs**: Features such as `monthly_income`, `EMI`, `total_payment`, etc.
- **Output**: JSON response with classification and regression predictions displayed on the web page

**Features**:
- The Django interface allows users to input data through a web form. This data is then sent to the Flask API for processing.
- Results from the Flask API are retrieved and displayed to the user on a dedicated web page.

**Deployment Steps**:
1. **Install Django**: Ensure Django is installed using `pip install django`.
2. **Configure Views**: Set up the `predict_view` in Django to handle form submissions and communicate with the Flask API.
3. **Run Django Server**: Start the Django application using `python manage.py runserver`. The server will run locally on port 8000 by default.
4. **Testing**: Access the web interface through a browser and submit feature data to verify that predictions are correctly received and displayed.

**Integration**:
- **Flask and Django Interaction**: The Django application sends POST requests to the Flask API with feature data. Ensure that the Flask server is running and accessible from the Django application.
- **Error Handling**: Implement error handling in both applications to manage any issues with data submission or predictions.

**Important Notes**:
- **Security**: Ensure that both Flask and Django applications are secured and not exposed to unauthorized access.
- **Performance**: Monitor the performance and responsiveness of both the Flask API and Django web interface.

For detailed setup and configuration, refer to the official documentation for Flask and Django. Make sure to test the entire workflow to ensure smooth integration between the two applications.
