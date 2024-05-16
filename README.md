# Spotify Music Preference Classification 🎶

Yay 👍🏻! or Nay 👎🏻! 

A Binary Classification project leveraging powerful algorithms such as Extreme Gradient Boosting, Stacking Classifier (KNN, Decision Tree, Logistic Regression, SVM, Bernoulli Naive Bayes), Gaussian Naive Bayes, and Random Forest to predict user preferences for songs.

## Key Techniques
- **Hyperparameter Tuning**: Using Gridsearch CV.
- **Cross Validation**: Ensures robust model evaluation through 10-fold cross-validation.
- **Data Preprocessing**: Includes handling duplicates, managing outliers via deletion or clipping, and standardizing data alongside label encoding string values to prepare for effective model training.

This project combines advanced algorithms and meticulous data preparation to create a predictive model aimed at enhancing the Spotify user experience by predicting song preferences.

Comparison Results between different ML Models: <br>

## <span style="color:green">Overall Conclusion:</span>
| Algorithm            | Accuracy (%) | Train-Test Diff (%) | Precision (%) | Recall (%) | F1 Score (%) | AUC (%) |
|----------------------|--------------|---------------------|---------------|------------|--------------|----------|
| XGB                  | <span style="color:green">72.80</span>        | <span style="color:green">3.86</span>  | <span style="color:green">72.22</span>         | <span style="color:green">72.96</span>  | <span style="color:green">72.59</span>        | 72.80     |
| Gaussian Naive Bayes | 62.22        | 4.41                | 64.38         | 52.55      | 57.87        | 62.10     |
| Random Forest        | 71.54  | 6.89  | 70.65 | 72.45      | 71.54  | 71.55     |
| Stacking Classifier  | 71.03        | <span style="color:red">7.33</span>                 | 69.95         | 72.45      | 71.18        | <span style="color:green">75.24</span> |

- XGB appears to be the best-performing model when considering accuracy, precision, and F1-score. It also has the lowest difference between train and test accuracy, indicating good generalization.

-  Random Forest is a close second in terms of accuracy, precision, recall, and F1-score. However, it has a higher difference between train and test accuracy, indicating that it may be overfitting.

- Stacking Classifier performs competitively but doesn't outshine the other models in any particular metric except AUC, where it performs the best.

- Gaussian Naive Bayes has the lowest performance across all metrics except for the difference between train and test accuracies, where it performs well, indicating good generalization but possibly a simpler model.


Tran Test Accuracies (To measure overfit):
![image](https://github.com/Bernardbyy/SpotifyMusicClassifier/assets/75737130/220b6450-a1fb-4afa-8b8d-f2447181905d)

Accuracy: 
![image](https://github.com/Bernardbyy/SpotifyMusicClassifier/assets/75737130/6973c318-7a59-42f9-981c-4e9386322b3f)

Precision: 
![image](https://github.com/Bernardbyy/SpotifyMusicClassifier/assets/75737130/c335f07b-d457-4cfb-89c6-1cadd037f812)

Recall:
![image](https://github.com/Bernardbyy/SpotifyMusicClassifier/assets/75737130/dc74265a-4ba3-4586-9d9c-ebcb0a92fc70)

F1-Score: 
![image](https://github.com/Bernardbyy/SpotifyMusicClassifier/assets/75737130/9b4c8274-8ecb-48d7-9131-057652507810)

AUC: 
![image](https://github.com/Bernardbyy/SpotifyMusicClassifier/assets/75737130/9a97a3b6-b3d3-4bb1-8f15-128b7f035dfc)




