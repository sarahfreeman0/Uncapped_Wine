This project provides a method and Python model to predict the relative change in the concentration of volatile compounds in uncapped wine. The model utilizes processed GC-MS data to identify the lines of best fit for compounds' corrected areas over time and uses that model to predict future concentrations. For the full research methodology, please refer to the abridged report.

Key Features:

The model automatically evaluates exponential and polynomial (1-6) regression models, with a threshold R-squared value of 0.980.
It accounts for varying initial concentrations to maintain accuracy over different samples.
The model works off a standardized Excel format that is uploaded by the user. A template containing the experimental data is provided in the repository.
Achieved Performance:

The model was trained on Sauvignon Blanc data and validated against Pinot Grigio test data. The model achieved high predictive accuracy via low normalized root mean square error (NRMSE) values for all compounds, supporting the model's robustness across different wines.

Usage Instructions:

Ensure you have the required libraries installed (see the attached requirements file).
Use the provided template to input your GC-MS corrected peak areas and time points.
Execute the enclosed Python script and upload the completed template.
