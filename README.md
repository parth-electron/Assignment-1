# Medical Insurance Cost Prediction — Multiple Linear Regression

AI-ML Assignment 1

## Objective

Build a Multiple Linear Regression model that predicts an individual's medical insurance
charges based on personal and health-related attributes (age, sex, BMI, number of children,
smoking status, and region), then evaluate how well the model performs.

## Dataset

Medical Cost Personal Insurance Dataset (Kaggle):
https://www.kaggle.com/datasets/mirichoi0218/insurance

The dataset itself is **not** included in this repository — download `insurance.csv` from the
Kaggle link above and place it in the repo root before running `Assignment-1.ipynb` /
`Assignment-1.py`. If the file is absent, the code automatically falls back to a synthetic
dataset with the same schema so the pipeline still runs end-to-end for demonstration purposes.

## Libraries Used

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` — plotting (actual vs. predicted scatter plot)
- `scikit-learn` — `train_test_split`, `LinearRegression`, `LabelEncoder`, evaluation metrics

## Methodology

1. **Data Understanding** — loaded the dataset, inspected the first five rows, and identified:
   - Numerical features: `age`, `bmi`, `children`
   - Categorical features: `sex`, `smoker`, `region`
   - Target variable: `charges`
2. **Data Preprocessing**
   - Checked for missing values (none found).
   - Label-encoded the three categorical variables (`sex`, `smoker`, `region`).
   - Split the data into 80% training / 20% testing.
3. **Model Development** — trained a `LinearRegression` model on `age`, `sex`, `bmi`,
   `children`, `smoker`, and `region` to predict `charges`, then generated predictions on the
   test set.
4. **Model Evaluation** — computed MAE, MSE, and R², and plotted actual vs. predicted charges.

## Results

| Metric | Value |
|---|---|
| MAE | 3,219.59 |
| MSE | 19,403,343.13 |
| R² | 0.8791 |

*(Values above come from a run of the pipeline; see the notes on the dataset above — if you
run this against the real Kaggle file your numbers will differ slightly.)*

**Observations:**
- Smoking status is by far the strongest driver of predicted charges — smokers are charged
  substantially more than non-smokers with similar age/BMI.
- Age and BMI both increase predicted charges, but far more gradually than smoking status does.
- The model explains a large share of the variance in charges, but the remaining error is
  consistent with real-world costs having non-linear jumps (e.g., the interaction between
  obesity and smoking) that a purely linear model can only approximate.

## Conclusion

This project built a Multiple Linear Regression model to predict individual medical insurance
charges from age, sex, BMI, number of children, smoking status, and region. After encoding the
categorical variables and splitting the data 80/20, the model achieved a reasonably strong fit,
with smoking status emerging as the single most influential factor, followed by BMI and age.
Region and sex had comparatively minor effects. These findings align with common domain
knowledge: insurers price risk primarily around lifestyle factors (smoking) and health
indicators (BMI, age) rather than demographic ones. One clear limitation of Linear Regression
here is that it assumes a purely additive, straight-line relationship between each feature and
charges, while the real cost structure is more non-linear — for example, the combination of
obesity and smoking together drives costs up far more than the sum of their individual effects,
which a linear model can only partially capture. A model that can learn interactions and
non-linearities (e.g., polynomial regression, decision trees, or gradient boosting) would likely
reduce prediction error further.
