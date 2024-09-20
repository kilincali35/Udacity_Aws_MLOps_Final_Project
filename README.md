# Udacity_Aws_MLOps_Final_Project

Data Prep and Analysis notebook is responsible of preparing the datasets, fetching them and uploading them to S3 bucket. And also making some EDA on the dataset we have

JSON files are responsible of holding the file names required to download from the online repository. 

HPO_benchmark and HPO_revised .py files are responsible of holding main code that will be used in hyperparameter search. You need to make core changes inside these files

train_benchmark and train_revised .py files are responsible of holding the main code for training the algorithms. All training steps, modeling and transformation steps are hold inside these modules.

benchmark_nbk and revised_nbk jupyter files are responsible of running models. Inside these files, you will see a customization like; run "hyperparameter optimization from scratch" or "run training from scratch". You need to set these values either as 1 or 0. 1 means Yes, 0 means 0. If you set to re-train; whole process starts with a brand new hpo or training run. But if you select 0, it will go to your models repository and will choose the latest and most suitable model; or hyperparameters. In the end of these notebooks, there is also inference section, where I couldn't succeed to make them run properly.

There are also inference.py and lambdafunction.py files; that would be required if we were using the inference and real-time prediction options inside the project. 
