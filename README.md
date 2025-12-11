This GitHub repository extracts the sites of distant recurrence for breast cancer patients from the free-text notes (clinical/radiology/pathology notes) using task specific fine-tuned BioLinkBERT model. This model was fine-tuned with recurrence relevant text from notes curated for every 1 month. 

# Inference
Following are the steps to be followed to run the repo:
1. **Create conda environment** using the following command:
>conda create -f environment.yml

Activate the conda environment using the following command:
>conda activate PE

2. **Run BreastRecurrence_Transformer GitHub repository:** To make your notes suitable for using this repository, run the BreastRecurrence_Transformer GitHub repository (given below) first on your notes. This will generate 4 files, namely: Patient_encounters.csv, pred_recur.csv, preprocessed_notes.csv, quarters.csv. Out of which `pred_recur.csv` is the main file that will be used further.

>https://github.com/imonban/BreastRecurrence_Transformer

The above repository curates the text for every 1 month and predicts whether a patient has recurrence during that interval. The main columns of `pred_recur` file are: `PATIENT_ID,START_DATE, END_DATE, text, Prediction`.
The `text` column is used in our repository to extract the sites of distant recurrence. 

3. **Set config.json file**: Now set two fields in the `config.json` file:

- input_file_path (path of pred_recur file)
- output_folder

4. **Model weights:** Send an email to `mbsikha@gmail.com` requesting the link for downloading the model weights. After downloading the model weights, keep them in the `model_weights` folder.

5. **Inference**: For inference, run the following command:
>python inference.py

### Model generalization
Although this model was trained on breast cancer dataset, it also performed well when tested on prostate cancer dataset.
