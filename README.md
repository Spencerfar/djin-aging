The DJIN (Dynamic joint interpretable network) model of aging, from https://arxiv.org/abs/2105.03410. A diagram of this model is located in Model_Diagram/. This model is trained with the ELSA dataset, which can be accessed here https://www.elsa-project.ac.uk/accessing-elsa-data by registering. A synthetic dataset generated with this model is available at https://zenodo.org/record/4733386.

The DJIN model is located in the Model/ directory, additionally we have included 3 alternate models in Alternate_models/, as well as Elastic net linear comparison models in Comparison_models/. Hyperparameters used to train the models are available in Output/. The parameters for the trained models are available in Parameters/.

After training the model with train.py, predict.py outputs predictions from the test set to Analysis_Data/. Then the plotting code in Plotting_code/ is used to generate plots. Plots are shown in Plots/.

The model requires pytorch, numpy, scipy, and pandas to run. Creating datasets additionally requires pyreadstat, plotting requires matplotlib and seaborn, and lienar comparison models require scikit-learn and lifelines. The code was originally run with pytorch 1.8.1, numpy 1.19.5, scikit-learn 0.24.1, and lifelines 0.25.8.

After acquiring the dataset from https://www.elsa-project.ac.uk/accessing-elsa-data, the create_elsa_data.sh file in clean_elsa/ can be run to create the train, validation, and test data sets. (Requires setting the folder for the dataset in the individual files).

Please cite the paper if you use this model (https://arxiv.org/abs/2105.03410).

