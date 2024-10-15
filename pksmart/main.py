#!/usr/bin/env python
import os
import sys
import argparse
import warnings
import pickle
import numpy as np
import pandas as pd
from loguru import logger
from tabulate import tabulate
from datetime import datetime
from argparse import RawTextHelpFormatter
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from mordred import Calculator, descriptors
from dimorphite_dl.dimorphite_dl import DimorphiteDL

# Get the directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings("ignore", category=UserWarning)

# Define the threshold and the alert message
threshold = 0.2
now = datetime.now()

banner = """
 ███████████  █████   ████  █████████                                       █████   
░░███░░░░░███░░███   ███░  ███░░░░░███                                     ░░███    
 ░███    ░███ ░███  ███   ░███    ░░░  █████████████    ██████   ████████  ███████  
 ░██████████  ░███████    ░░█████████ ░░███░░███░░███  ░░░░░███ ░░███░░███░░░███░   
 ░███░░░░░░   ░███ ░░███   ███    ░███ ░███ ░███ ░███   ███████  ░███ ░░░   ░███    
 ░███         ░███ ░░███   ███    ░███ ░███ ░███ ░███  ███░░███  ░███       ░███ ███
 █████        █████ ░░████░░█████████  █████░███ █████░░████████ █████      ░░█████ 
░░░░░        ░░░░░   ░░░░  ░░░░░░░░░  ░░░░░ ░░░ ░░░░░  ░░░░░░░░ ░░░░░        ░░░░░                                                 
                                                                                    """

abstract = "Abstract:\nDrug exposure is a key contributor to the safety and efficacy of drugs. It can be defined using human pharmacokinetic (PK) parameters that affect the blood concentration profile of a drug, such as steady-state volume of distribution (VDss), total body clearance (CL), half-life (t½), fraction unbound in plasma (fu) and mean residence time (MRT). In this work, we used molecular structural fingerprints, physicochemical properties, and predicted animal PK data as features to model the human PK parameters VDss, CL, t½, fu and MRT for 1,283 unique compounds. First, we predicted animal PK parameters [VDss, CL, fu] for rats, dogs, and monkeys for 372 unique compounds using molecular structural fingerprints and physicochemical properties. Second, we used Morgan fingerprints, Mordred descriptors and predicted animal PK parameters in a hyperparameter-optimised Random Forest algorithm to predict human PK parameters. When validated using repeated nested cross-validation, human VDss was best predicted with an R2 of 0.55 and a Geometric Mean Fold Error (GMFE) of 2.09; CL with accuracies of R2=0.31 and GMFE=2.43, fu with R2=0.61 and GMFE=2.81, MRT with R2=0.28 and GMFE=2.49, and t½ with R2=0.31 and GMFE=2.46 for models combining Morgan fingerprints, Mordred descriptors and predicted animal PK parameters. We evaluated models with an external test set comprising 315 compounds for VDss (R2=0.33 and GMFE=2.58) and CL (R2=0.45 and GMFE=1.98). We compared our models with proprietary pharmacokinetic (PK) models from AstraZeneca and found that model predictions were similar with Pearson correlations ranging from 0.77-0.78 for human PK parameters of VDss and fu and 0.46-0.71 for animal (dog and rat) PK parameters of VDss, CL and fu. To the best of our knowledge, this is the first work that publicly releases PK models on par with industry-standard models. Early assessment and integration of predicted PK properties are crucial, such as in DMTA cycles, which is possible with models in this study based on the input of only chemical structures. We developed a webhosted application PKSmart (https://broad.io/PKSmart) which users can access using a web browser with all code also downloadable for local use."

cite = """If you use PKSmart in your work, please cite:\nPKSmart: An Open-Source Computational Model to Predict in vivo Pharmacokinetics of Small Molecules
Srijit Seal, Maria-Anna Trapotsi, Vigneshwari Subramanian, Ola Spjuth, Nigel Greene, Andreas Bender
bioRxiv 2024.02.02.578658; doi: https://doi.org/10.1101/2024.02.02.578658\n"""


# use pH 7.4 https://git.durrantlab.pitt.edu/jdurrant/dimorphite_dl/
dimorphite = DimorphiteDL(min_ph=7.4, max_ph=7.4, pka_precision=0)

def standardize(smiles):
    # follows the steps in
    # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    # as described **excellently** (by Greg) in
    # https://www.youtube.com/watch?v=eWTApNX8dJQ
    try:
        mol = Chem.MolFromSmiles(smiles)
        # print(smiles)

        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol)
        # print(Chem.MolToSmiles(clean_mol))

        # if many fragments, get the "parent" (the actual mol we are interested in)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = (
            rdMolStandardize.Uncharger()
        )  # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        # print(uncharged_parent_clean_mol)

        protonated_smiles = dimorphite.protonate(
            Chem.MolToSmiles(uncharged_parent_clean_mol)
        )

        # print("protonated_smiles")

        if len(protonated_smiles) > 0:
            protonated_smile = protonated_smiles[0]

        protonated_mol = Chem.MolFromSmiles(protonated_smile)
        # protonated_mol= AddHs(protonated_mol)
        # protonated_smile = Chem.MolToSmiles(protonated_mol)

        # attempt is made at reionization at this step
        # at 7.4 pH

        te = rdMolStandardize.TautomerEnumerator()  # idem
        taut_uncharged_parent_clean_mol = te.Canonicalize(protonated_mol)

        return Chem.MolToSmiles(taut_uncharged_parent_clean_mol)

    except:

        return "Cannot_do"


def calcdesc(data):
    # create descriptor calculator with all descriptors
    calc = Calculator(descriptors, ignore_3D=True)

    Ser_Mol = data["standardized_smiles"].apply(Chem.MolFromSmiles)
    Mordred_table = calc.pandas(Ser_Mol)
    Mordred_table = Mordred_table.astype("float")

    Morgan_fingerprint = Ser_Mol.apply(GetMorganFingerprintAsBitVect, args=(2, 2048))
    Morganfingerprint_array = np.stack(Morgan_fingerprint)

    Morgan_collection = [f"Mfp{x}" for x in range(Morganfingerprint_array.shape[1])]

    Morganfingerprint_table = pd.DataFrame(
        Morganfingerprint_array, columns=Morgan_collection
    )

    # Combine all features into one DataFrame
    result = pd.concat([data, Mordred_table, Morganfingerprint_table], axis=1)
    
    # Remove duplicate columns if any
    result = result.loc[:, ~result.columns.duplicated()]
    return result


def predict_individual_animal(data, endpoint, animal):
    # Load the feature list for the specific animal model
    with open(os.path.join(SCRIPT_DIR, "data", f"features_mfp_mordred_columns_{animal}_model.txt"), "r") as file:
        features = file.read().splitlines()

    # Load the pre-trained model for the given endpoint
    with open(os.path.join(SCRIPT_DIR, "data", f"log_{endpoint}_model_FINAL.sav"), "rb") as file:
        loaded_rf = pickle.load(file)

    # Select data based on the feature list
    X = data[features]

    # Replace missing descriptors with median
    animal_median = pd.read_csv(
        os.path.join(SCRIPT_DIR, "data", f"Median_mordred_values_{animal}_for_artificial_animal_data_mfp_mrd_model.csv")
    )

    for i in X.columns[X.isna().any()].tolist():
        X[i].fillna(float(animal_median[i]), inplace=True)

    # Load the scaler and apply it to the data
    with open(os.path.join(SCRIPT_DIR, "data", f"scaler_{animal}.pkl"), "rb") as file:
        scaler = pickle.load(file)

    # Scale the features and create a DataFrame with the scaled data
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features)
    
    # Predict the target variable using the loaded model
    y_pred = loaded_rf.predict(X_scaled)
    
    return y_pred


def predict_animal(data):
    animal_endpoints = {
        "dog": ["dog_VDss_L_kg", "dog_CL_mL_min_kg", "dog_fup"],
        "monkey": ["monkey_VDss_L_kg", "monkey_CL_mL_min_kg", "monkey_fup"],
        "rat": ["rat_VDss_L_kg", "rat_CL_mL_min_kg", "rat_fup"]
    }

    # Loop through each animal and its endpoints
    for animal, endpoints in animal_endpoints.items():
        for endpoint in endpoints:
            preds = predict_individual_animal(data, endpoint, animal)
            data[endpoint] = preds

    return data


def predict_VDss(data, features):  
    # Load the pre-trained random forest model
    with open(os.path.join(SCRIPT_DIR, "data", "log_human_VDss_L_kg_withanimaldata_artificial_model_FINAL.sav"), "rb") as model_file:
        loaded_rf = pickle.load(model_file)

    # Extract and scale the feature data
    X = data[features].values

    # Load the scaler and apply it
    with open(os.path.join(SCRIPT_DIR, "data", "artificial_animal_data_mfp_mrd_human_VDss_L_kg_scaler.pkl"), "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    X_scaled = scaler.transform(X)

    # Convert the scaled data back to a DataFrame with the original feature names
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    # Predict using the loaded model
    y_preds = loaded_rf.predict(X_scaled)

    return y_preds


def predict_CL(data, features):

    # Load the pre-trained random forest model
    with open(os.path.join(SCRIPT_DIR, "data", "log_human_CL_mL_min_kg_withanimaldata_artificial_model_FINAL.sav"), "rb") as model_file:
        loaded_rf = pickle.load(model_file)

    # Extract and scale the feature data
    X = data[features].values

    # Load the scaler and apply it
    with open(os.path.join(SCRIPT_DIR, "data", "artificial_animal_data_mfp_mrd_human_CL_mL_min_kg_scaler.pkl"), "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    X_scaled = scaler.transform(X)

    # Convert the scaled data back to a DataFrame with the original feature names
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    # Predict using the loaded model
    y_preds = loaded_rf.predict(X_scaled)
    return y_preds


def predict_fup(data, features):
    # Load the pre-trained random forest model
    with open(os.path.join(SCRIPT_DIR, "data", "log_human_fup_withanimaldata_artificial_model_FINAL.sav"), "rb") as model_file:
        loaded_rf = pickle.load(model_file)

    # Extract and scale the feature data
    X = data[features].values

    # Load the scaler and apply it
    with open(os.path.join(SCRIPT_DIR, "data", "artificial_animal_data_mfp_mrd_human_fup_scaler.pkl"), "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    X_scaled = scaler.transform(X)

    # Convert the scaled data back to a DataFrame with the original feature names
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    # Predict using the loaded model
    y_preds = loaded_rf.predict(X_scaled)

    return y_preds

def predict_MRT(data, features):
    # Load the pre-trained random forest model
    with open(os.path.join(SCRIPT_DIR, "data", "log_human_mrt_withanimaldata_artificial_model_FINAL.sav"), "rb") as model_file:
        loaded_rf = pickle.load(model_file)

    # Extract and scale the feature data
    X = data[features].values

    # Load the scaler and apply it
    with open(os.path.join(SCRIPT_DIR, "data", "artificial_animal_data_mfp_mrd_human_mrt_scaler.pkl"), "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    X_scaled = scaler.transform(X)

    # Convert the scaled data back to a DataFrame with the original feature names
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    # Predict using the loaded model
    y_preds = loaded_rf.predict(X_scaled)

    return y_preds


def predict_thalf(data, features):
    # Load the pre-trained random forest model
    with open(os.path.join(SCRIPT_DIR, "data", "log_human_thalf_withanimaldata_artificial_model_FINAL.sav"), "rb") as model_file:
        loaded_rf = pickle.load(model_file)

    # Extract and scale the feature data
    X = data[features].values

    # Load the scaler and apply it
    with open(os.path.join(SCRIPT_DIR, "data", "artificial_animal_data_mfp_mrd_human_thalf_scaler.pkl"), "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    X_scaled = scaler.transform(X)

    # Convert the scaled data back to a DataFrame with the original feature names
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    # Predict using the loaded model
    y_preds = loaded_rf.predict(X_scaled)

    return y_preds


def get_canonical_smiles(smiles_list, dataset_name):
    """
    Convert SMILES to canonical SMILES format, handling invalid SMILES with error logging.

    Parameters:
    - smiles_list (Series): List of SMILES strings.
    - dataset_name (str): The name of the dataset (for logging purposes).

    Returns:
    - list: A list of canonical SMILES.
    """
    canonical_smiles = []
    for smiles in smiles_list:
        try:
            canonical_smiles.append(Chem.CanonSmiles(smiles))
        except:
            print(f"Invalid SMILES in {dataset_name} dataset: {smiles}")
    return canonical_smiles


def calculate_similarity_test_vs_train(test, train):
    """
    Calculate Tanimoto similarity between test and train datasets based on their SMILES representations.

    Parameters:
    - test (DataFrame): Test dataset containing a 'smiles_r' column with SMILES strings.
    - train (DataFrame): Train dataset containing a 'smiles_r' column with SMILES strings.

    Returns:
    - DataFrame: A DataFrame with 'query' (test compounds), 'target' (train compounds), and 'MFP_Tc' (similarity scores).
    """

    # Convert SMILES to canonical SMILES format
    c_smiles_test = get_canonical_smiles(test['smiles_r'], "test")
    c_smiles_train = get_canonical_smiles(train['smiles_r'], "train")

    # Convert canonical SMILES to RDKit mol objects and generate fingerprints
    ms_test = [Chem.MolFromSmiles(smiles) for smiles in c_smiles_test]
    fps_test = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in ms_test]

    ms_train = [Chem.MolFromSmiles(smiles) for smiles in c_smiles_train]
    fps_train = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in ms_train]

    # Lists for query (test) compounds, target (train) compounds, and similarity scores
    query_list, target_list, similarity_list = [], [], []

    # Compare each test fingerprint against all train fingerprints and calculate Tanimoto similarity
    for i, fp_test in enumerate(fps_test):
        similarities = DataStructs.BulkTanimotoSimilarity(fp_test, fps_train)
        query_smile = c_smiles_test[i]

        # Store query, target, and similarity score for each comparison
        for j, similarity in enumerate(similarities):
            query_list.append(query_smile)
            target_list.append(c_smiles_train[j])
            similarity_list.append(similarity)

    # Create DataFrame from the collected data
    similarity_df = pd.DataFrame({
        'query': query_list,
        'target': target_list,
        'MFP_Tc': similarity_list
    })

    return similarity_df


def avg_kNN_similarity(test_data, train_data_path=os.path.join(SCRIPT_DIR, "data", "Train_data_log_transformed.csv"), n_neighbours=5):
    """
    Parameters:
    - test_data (DataFrame): Test data that will be compared against the training data.
    - train_data_path (str): Path to the CSV file containing the training data (default is '../Train_data_log_transformed.csv').
    - n_neighbours (int): Number of nearest neighbors to consider for similarity (default is 5).
    
    Returns:
    - DataFrame: A DataFrame with the mean similarity scores for each endpoint, rounded to 2 decimal places.
    """

    endpoints = ["human_VDss_L_kg", "human_CL_mL_min_kg", "human_fup", "human_mrt", "human_thalf"]

    # Load the training data
    train_data = pd.read_csv(train_data_path)

    df_master = pd.DataFrame()

    for endpoint in endpoints:
        # Filter the training data for the current endpoint, removing rows with missing values
        df_filtered = train_data.dropna(subset=[endpoint]).reset_index(drop=True)

        # Calculate similarity between test and filtered training data
        df_similarity = calculate_similarity_test_vs_train(test_data, df_filtered)

        # Sort by similarity score (MFP_Tc) in descending order
        df_similarity_sorted = df_similarity.sort_values(['query', 'MFP_Tc'], ascending=[True, False]).reset_index(drop=True)
        
        # Select the top n_neighbours for each unique query (compound)
        df_top_neighbours = df_similarity_sorted.groupby('query').head(n_neighbours)

        # Group by query and calculate the mean of numeric values
        df_aggregated = df_top_neighbours.groupby('query').mean(numeric_only=True)
        
        # Assign the current endpoint to the results
        df_aggregated["endpoint"] = endpoint
        
        # Append the results to the master DataFrame
        df_master = pd.concat([df_master, df_aggregated])

    # Pivot the master DataFrame
    result_df = df_master.pivot_table(index='query', columns='endpoint', values='MFP_Tc').reset_index().round(2)

    return result_df


def check_if_in_training_data(smiles):
    df = pd.read_csv(os.path.join(SCRIPT_DIR, "data", "Train_data_features.csv"))
    if smiles in df['smiles_r'].values:
        logger.critical(f"SMILES string : \"{smiles}\" was found in Training data with following features:")
        logger.critical(tabulate(df[df['smiles_r'] == smiles][['human_VDss_L_kg','human_CL_mL_min_kg','human_fup','human_mrt','human_thalf']], headers='keys', tablefmt='psql', showindex=False))
        

def predict_pk_params(smiles:str):
    logger.info("Starting PK parameter prediction")
    
    # Create an empty DataFrame to hold the SMILES and predictions
    data = pd.DataFrame([smiles], columns=['smiles_r'])
    logger.debug(f"Input data:\n{tabulate(data, headers='keys', tablefmt='psql', showindex=False)}")

    # Standardize and calculate descriptors for the input molecules
    logger.info("Standardizing SMILES and calculating descriptors")
    data['standardized_smiles'] = data['smiles_r'].apply(standardize)
    logger.debug(f"Standardized SMILES:\n{tabulate(data[['smiles_r', 'standardized_smiles']], headers='keys', tablefmt='psql', showindex=False)}")

    for smiles in data["standardized_smiles"].values:
        check_if_in_training_data(smiles)

    data_mordred = calcdesc(data)
    logger.debug(f"Calculated descriptors shape: {data_mordred.shape}")

    ts_data = avg_kNN_similarity(data)
    logger.debug(f"Average kNN similarity:\n{tabulate(ts_data, headers='keys', tablefmt='psql', showindex=False)}")

    # Run predictions for animal models
    logger.info("Predicting animal pharmacokinetic parameters")
    animal_predictions = predict_animal(data_mordred)

    # Filter out only the relevant animal PK columns
    animal_columns = [
        "dog_VDss_L_kg", "dog_CL_mL_min_kg", "dog_fup",
        "monkey_VDss_L_kg", "monkey_CL_mL_min_kg", "monkey_fup",
        "rat_VDss_L_kg", "rat_CL_mL_min_kg", "rat_fup"
    ]

    logger.debug(f"Animal predictions:\n{tabulate(animal_predictions[['smiles_r'] + animal_columns].round(2).head(), headers='keys', tablefmt='psql', showindex=False)}")

    # Run predictions for human models
    logger.info("Predicting human pharmacokinetic parameters")
    human_predictions = pd.DataFrame()

    with open(os.path.join(SCRIPT_DIR, "data", "features_mfp_mordred_animal_artificial_human_modelcolumns.txt")) as f:
        model_features = f.read().splitlines()

    human_predictions['smiles_r'] = data_mordred['smiles_r']

    human_predictions['VDss_L_kg'] = predict_VDss(data_mordred, model_features)
    Vd_Tc = ts_data["human_VDss_L_kg"]

    with open(os.path.join(SCRIPT_DIR, "data", "folderror_human_VDss_L_kg_generator.sav"), 'rb') as f:
        loaded = pickle.load(f)
        human_predictions['Vd_fe'] = loaded.predict(Vd_Tc.values.reshape(-1, 1)).round(2)
    human_predictions['Vd_min'] = human_predictions['VDss_L_kg'] / human_predictions['Vd_fe']
    human_predictions['Vd_max'] = human_predictions['VDss_L_kg'] * human_predictions['Vd_fe']
    alert_message = "Alert: This Molecule May Be Out Of AD for VDss (<0.2 Tc with Training data)"
    human_predictions['comments'] = ts_data['human_VDss_L_kg'].apply(
        lambda x: f"{alert_message}" if x < threshold else ""
    )

    human_predictions['CL_mL_min_kg'] = predict_CL(data_mordred, model_features)
    CL_Tc =  ts_data["human_CL_mL_min_kg"]
    with open(os.path.join(SCRIPT_DIR, "data", "folderror_human_CL_mL_min_kg_generator.sav"), 'rb') as f:    
        loaded = pickle.load(f)
        human_predictions["CL_fe"]= loaded.predict(CL_Tc.values.reshape(-1, 1)).round(2)
    human_predictions['CL_min'] = human_predictions['CL_mL_min_kg'] / human_predictions['CL_fe']
    human_predictions['CL_max'] = human_predictions['CL_mL_min_kg'] * human_predictions['CL_fe']
    alert_message = "Alert: This Molecule May Be Out Of AD for CL (<0.2 Tc with Training data)"
    human_predictions['comments'] = human_predictions['comments'] + ts_data['human_CL_mL_min_kg'].apply(
        lambda x: f"\n{alert_message}" if x < threshold else ""
    )

    human_predictions['fup'] = predict_fup(data_mordred, model_features)
    fup_Tc =  ts_data["human_fup"]
    with open(os.path.join(SCRIPT_DIR, "data", "folderror_human_fup_generator.sav"), 'rb') as f:
        loaded = pickle.load(f)
        human_predictions["fup_fe"]= loaded.predict(fup_Tc.values.reshape(-1, 1)).round(2)
    human_predictions['fup_min'] = human_predictions['fup'] / human_predictions['fup_fe']
    human_predictions['fup_max'] = human_predictions['fup'] * human_predictions['fup_fe']
    alert_message = "Alert: This Molecule May Be Out Of AD for fup (<0.2 Tc with Training data)"

    human_predictions['comments'] = human_predictions['comments'] + ts_data['human_fup'].apply(
        lambda x: f"\n{alert_message}" if x < threshold else ""
    )

    human_predictions['MRT_hr'] = predict_MRT(data_mordred, model_features)
    MRT_Tc =  ts_data["human_mrt"]
    with open(os.path.join(SCRIPT_DIR, "data", "folderror_human_mrt_generator.sav"), 'rb') as f:
        loaded = pickle.load(f)
        human_predictions["MRT_fe"]= loaded.predict(MRT_Tc.values.reshape(-1, 1)).round(2)
    human_predictions['MRT_min'] = human_predictions['MRT_hr'] / human_predictions['MRT_fe']
    human_predictions['MRT_max'] = human_predictions['MRT_hr'] * human_predictions['MRT_fe']
    alert_message = "Alert: This Molecule May Be Out Of AD for MRT (<0.2 Tc with Training data)"

    human_predictions['comments'] = human_predictions['comments'] + ts_data['human_mrt'].apply(
        lambda x: f"\n{alert_message}" if x < threshold else ""
    )

    human_predictions['thalf_hr'] = predict_thalf(data_mordred, model_features)
    thalf_Tc =  ts_data["human_thalf"]
    with open(os.path.join(SCRIPT_DIR, "data", "folderror_human_thalf_generator.sav"), 'rb') as f:
        loaded = pickle.load(f)
        human_predictions["thalf_fe"]= loaded.predict(thalf_Tc.values.reshape(-1, 1)).round(2)
    human_predictions['thalf_min'] = human_predictions['thalf_hr'] / human_predictions['thalf_fe']
    human_predictions['thalf_max'] = human_predictions['thalf_hr'] * human_predictions['thalf_fe']
    alert_message = "Alert: This Molecule May Be Out Of AD for thalf (<0.2 Tc with Training data)"

    human_predictions['comments'] = human_predictions['comments'] + ts_data['human_thalf'].apply(
        lambda x: f"\n{alert_message}" if x < threshold else ""
    )

    human_predictions = human_predictions[[col for col in human_predictions if col != 'comments'] + ['comments']]

    # Display the human predictions in a table
    logger.debug(f"Human predictions:\n{tabulate(human_predictions.round(2).head(), headers='keys', tablefmt='psql', showindex=False)}")

    combined_predictions = pd.merge(human_predictions, animal_predictions[animal_columns + ['smiles_r']], on='smiles_r')

    column_mapping = {
        "VDss": "Volume_of_distribution_(VDss)(L/kg)",
        "Vd_fe": "Volume_of_distribution_(VDss)_folderror",
        "Vd_min": "Volume_of_distribution_(VDss)_lowerbound",
        "Vd_max": "Volume_of_distribution_(VDss)_upperbound",
        "CL": "Clearance_(CL)_(mL/min/kg)",
        "CL_fe": "Clearance_(CL)_folderror",
        "CL_min": "Clearance_(CL)_lowerbound",
        "CL_max": "Clearance_(CL)_upperbound",
        "fup": "Fraction_unbound_in_plasma_(fup)",
        "fup_fe": "Fraction_unbound_in_plasma_(fup)_folderror",
        "fup_min": "Fraction_unbound_in_plasma_(fup)_lowerbound",
        "fup_max": "Fraction_unbound_in_plasma_(fup)_upperbound",
        "MRT": "Mean_Residence_Time_(MRT)(h)",
        "MRT_fe": "Mean_Residence_Time_(MRT)_folderror",
        "MRT_min": "Mean_Residence_Time_(MRT)_lowerbound",
        "MRT_max": "Mean_Residence_Time_(MRT)_upperbound",
        "thalf": "Half_life_(thalf)(h)",
        "thalf_fe": "Half_life_(thalf)_folderror",
        "thalf_min": "Half_life_(thalf)_lowerbound",
        "thalf_max": "Half_life_(thalf)_upperbound"
    }
    combined_predictions.rename(columns=column_mapping, inplace=True)

    return combined_predictions

def main():
    parser = argparse.ArgumentParser(description=banner+'\n\n'+abstract+'\n\n'+cite, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--smiles', '-s', '-smi', '--smi', '-smiles', type=str, help='Input SMILES string to predict properties')
    parser.add_argument('--file', '-f', type=str, help='Path to file containing newline separated SMILES strings')
    parser.add_argument('--log-level', '-l', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Set the logging level')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    if args.file:
        with open(args.file, 'r') as f:
            args.smiles = [line.strip() for line in f]
    elif args.smiles:
        args.smiles = args.smiles.split(',')
    else:
        parser.error("Either --smiles or --file must be provided")

    # Set up logging based on the specified level using loguru
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    logger.debug("Debug mode activated")
    logger.info(f"Log level set to {args.log_level}")

    df_results = pd.DataFrame()
    for smiles in args.smiles:
        combined_predictions = predict_pk_params(smiles)
        df_results = pd.concat([df_results, combined_predictions], ignore_index=True)
    
    logger.info("Saving Results ...")

    df_results.to_csv(f"pksmart_run_{now.strftime('%H-%M-%S-%d-%m-%Y')}.csv", index=False)

    logger.info(f"Results saved as pksmart_run_{now.strftime('%H-%M-%S-%d-%m-%Y')}.csv")


if __name__ == "__main__":
    main()
