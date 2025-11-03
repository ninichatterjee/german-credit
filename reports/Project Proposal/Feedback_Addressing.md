## Response to Proposal Feedback

After reviewing the feedback, I refined my metric and baseline justification sections.

### Metrics
For **classification**, I elaborated on the rationale behind each choice:
- **Accuracy** provides an overall measure of correct predictions.  
- **F1-score** balances precision and recall, which is especially important since the dataset has a 70–30 class split.  
- **ROC-AUC** measures how well the model distinguishes between good and bad credit cases across all thresholds.  

For **regression**, I justified the selection as follows:
- **MAE (Mean Absolute Error)** offers an interpretable measure of the average prediction error.  
- **RMSE (Root Mean Squared Error)** penalizes larger errors more strongly, which is valuable when over- or underestimating credit amounts could lead to significant financial consequences.  

### Baselines
For **classification**, I clarified that:
- **Logistic Regression** serves as a linear and interpretable baseline.  
- **Decision Tree** provides a nonlinear and more flexible comparison.  

For **regression**, I used:
- **Linear Regression** to capture basic linear trends.  
- **Decision Tree Regressor** to model nonlinear relationships in credit data.  

**Naïve Bayes** was excluded because the dataset includes many continuous-valued features that would require discretization or Gaussian assumptions beyond the intended baseline scope.

### Page Length Clarification
Regarding the page limit, the assignment guidelines specified a **2–3 page proposal**, excluding references. Our document is three pages long excluding references and the cover page, which adheres to those requirements. The 2-page comment may reflect a stricter interpretation, but according to the posted brief, the proposal met the stated page limit.
