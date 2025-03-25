### Penguin Species Classifier

Webpage: https://matmatx1.github.io/mlops_assignment_1/

This repository contains a classifier to predict the species of penguins based on some physical characteristics.

### Overview

1. **Database Creation**: 
   - The penguin dataset was downloaded and processed. It was then saved into an SQLite database for easy access and querying.
   
2. **Feature Selection**: 
   - Feature selection was done using **correlation** and **Chi-square tests**, reducing the original set of 5 features to 2: `flipper_length_mm`, `bill_depth_mm`.

3. **Classifier**: 
   - A **Random Forest Classifier** was trained on the selected features. The model achieved an **AUC of 1.0**

4. **API Integration and GitHub Action**:
   - Every monday at 7:30 AM, a GitHub Action fetches new data from the provided API, processes it, and provides a prediction using the trained model.


