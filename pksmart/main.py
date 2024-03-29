#!/usr/bin/env python
import os
import sys
import pickle
import argparse
import warnings
from datetime import datetime
from argparse import RawTextHelpFormatter
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from loguru import logger

warnings.filterwarnings("ignore", category=UserWarning)

now = datetime.now()
# Format the date and time components
formatted_date = now.strftime("%d-%m-%Y")
formatted_time = now.strftime("%H-%M-%S")


banner = """
 ███████████  █████   ████  █████████                                       █████   
░░███░░░░░███░░███   ███░  ███░░░░░███                                     ░░███    
 ░███    ░███ ░███  ███   ░███    ░░░  █████████████    ██████   ████████  ███████  
 ░██████████  ░███████    ░░█████████ ░░███░░███░░███  ░░░░░███ ░░███░░███░░░███░   
 ░███░░░░░░   ░███░░███    ░░░░░░░░███ ░███ ░███ ░███   ███████  ░███ ░░░   ░███    
 ░███         ░███ ░░███   ███    ░███ ░███ ░███ ░███  ███░░███  ░███       ░███ ███
 █████        █████ ░░████░░█████████  █████░███ █████░░████████ █████      ░░█████ 
░░░░░        ░░░░░   ░░░░  ░░░░░░░░░  ░░░░░ ░░░ ░░░░░  ░░░░░░░░ ░░░░░        ░░░░░                                                 
                                                                                    """

abstract = "Abstract:\nDrug exposure is a key contributor to the safety and efficacy of drugs. It can be defined using human pharmacokinetics (PK) parameters that affect the blood concentration profile of a drug, such as steady-state volume of distribution (VDss), total body clearance (CL), half-life (t½), fraction unbound in plasma (fu) and mean residence time (MRT). In this work, we used molecular structural fingerprints, physicochemical properties, and predicted animal PK data as features to model the human PK parameters VDss, CL, t½, fu and MRT for 1,283 unique compounds. First, we predicted animal PK parameters [VDss, CL, fu] for rats, dogs, and monkeys for 372 unique compounds using molecular structural fingerprints and physicochemical properties. Second, we used Morgan fingerprints, Mordred descriptors and predicted animal PK parameters in a hyperparameter-optimised Random Forest algorithm to predict human PK parameters. When validated using repeated nested cross-validation, human VDss was best predicted with an R2 of 0.55 and a Geometric Mean Fold Error (GMFE) of 2.09; CL with accuracies of R2=0.31 and GMFE=2.43, fu with R2=0.61 and GMFE=2.81, MRT with R2=0.28 and GMFE=2.49, and t½ with R2=0.31 and GMFE=2.46 for models combining Morgan fingerprints, Mordred descriptors and predicted animal PK parameters. We evaluated models with an external test set comprising 315 compounds for VDss (R2=0.33 and GMFE=2.58) and CL (R2=0.45 and GMFE=1.98). We compared our models with proprietary pharmacokinetic (PK) models from AstraZeneca and found that model predictions were similar with Pearson correlations ranging from 0.77-0.78 for human PK parameters of VDss and fu and 0.46-0.71 for animal (dog and rat) PK parameters of VDss, CL and fu. To the best of our knowledge, this is the first work that publicly releases PK models on par with industry-standard models. Early assessment and integration of predicted PK properties are crucial, such as in DMTA cycles, which is possible with models in this study based on the input of only chemical structures. We developed a webhosted application PKSmart (https://broad.io/PKSmart) which users can access using a web browser with all code also downloadable for local use."

cite = """If you use PKSmart in your work, please cite:\nPKSmart: An Open-Source Computational Model to Predict in vivo Pharmacokinetics of Small Molecules
Srijit Seal, Maria-Anna Trapotsi, Vigneshwari Subramanian, Ola Spjuth, Nigel Greene, Andreas Bender
bioRxiv 2024.02.02.578658; doi: https://doi.org/10.1101/2024.02.02.578658\n"""

def standardize(smiles):
    # follows the steps in
    # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    # as described **excellently** (by Greg) in
    # https://www.youtube.com/watch?v=eWTApNX8dJQ
    try:
        mol = Chem.MolFromSmiles(smiles)

        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol)

        # if many fragments, get the "parent" (the actual mol we are interested in)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = (
            rdMolStandardize.Uncharger()
        )  # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        # note that no attempt is made at reionization at this step
        # nor at ionization at some pH (rdkit has no pKa caculator)
        # the main aim to to represent all molecules from different sources
        # in a (single) standard way, for use in ML, catalogue, etc.

        te = rdMolStandardize.TautomerEnumerator()  # idem
        taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

        return Chem.MolToSmiles(taut_uncharged_parent_clean_mol)

    except:

        return "Cannot_do"


def calcdesc(data):
    # create descriptor calculator with all descriptors
    calc = Calculator(descriptors, ignore_3D=True)

    # print(len(calc.descriptors))
    Ser_Mol = data["smiles_r"].apply(Chem.MolFromSmiles)
    Mordred_table = calc.pandas(Ser_Mol)
    Mordred_table = Mordred_table.astype("float")
    Mordred_table["smiles_r"] = data["smiles_r"]

    Morgan_fingerprint = Ser_Mol.apply(GetMorganFingerprintAsBitVect, args=(2, 2048))
    Morganfingerprint_array = np.stack(Morgan_fingerprint)

    Morgan_collection = []
    for x in np.arange(
        Morganfingerprint_array.shape[1]
    ):  # np.arange plus rapide que range
        x = "Mfp" + str(x)
        Morgan_collection.append(x)

    Morganfingerprint_table = pd.DataFrame(
        Morganfingerprint_array, columns=Morgan_collection
    )
    Morganfingerprint_table["smiles_r"] = data["smiles_r"]

    data_mfp = pd.merge(data, Morganfingerprint_table)
    data_mfp_Mordred = pd.merge(data_mfp, Mordred_table)

    return data_mfp_Mordred

def calculate_similarity_test_vs_train(test, train):

    df_smiles_test = test["smiles_r"]
    df_smiles_train = train["smiles_r"]

    c_smiles_test = []
    for ds in df_smiles_test:
        try:
            cs = Chem.CanonSmiles(ds)
            c_smiles_test.append(cs)
        except:
            print("test")
            print("Invalid SMILES:", ds)

    c_smiles_train = []
    for ds in df_smiles_train:
        try:
            cs = Chem.CanonSmiles(ds)
            c_smiles_train.append(cs)
        except:
            print("train")
            print("Invalid SMILES:", ds)

    # make a list of mols
    ms_test = [Chem.MolFromSmiles(x) for x in c_smiles_test]

    # make a list of fingerprints (fp)
    fps_test = [
        AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits=2048) for x in ms_test
    ]

    # make a list of mols
    ms_train = [Chem.MolFromSmiles(x) for x in c_smiles_train]

    # make a list of fingerprints (fp)
    fps_train = [
        AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits=2048) for x in ms_train
    ]

    # the list for the dataframe
    qu, ta, sim = [], [], []

    # compare all fp pairwise without duplicates

    for a in range(len(fps_test)):

        s = DataStructs.BulkTanimotoSimilarity(fps_test[a], fps_train)
        for m in range(len(s)):
            qu.append(c_smiles_test[a])
            ta.append(c_smiles_train[m])
            sim.append(s[m])

    # build the dataframe and sort it
    d = {"query": qu, "target": ta, "MFP_Tc": sim}
    df_final_ai = pd.DataFrame(data=d)
    return df_final_ai

def predict_individual_animal(data_features, endpoint, animal):

    # Read columns needed for rat data
    file = open(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/features_mfp_mordred_columns_{animal}_model.txt", "r")
    file_lines = file.read()
    features = file_lines.split("\n")
    features = features[:-1]

    loaded_rf = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/log_{endpoint}_model_FINAL.sav", "rb"))

    X = data_features[features]

    animalmedian = pd.read_csv(
        os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/Median_mordred_values_{animal}_for_artificial_animal_data_mfp_mrd_model.csv"
    )

    for i in X.columns[X.isna().any()].tolist():
        # print(i)
        X[i].fillna(float(animalmedian[i]), inplace=True)
    # Changed to float when replacing 18 Nov 2022

    X = X.values
    y_pred = loaded_rf.predict(X)

    return y_pred

def predict_animal(data):
    data_features = data
    endpoints = {"dog_VDss_L_kg", "dog_CL_mL_min_kg", "dog_fup"}

    for endpoint in endpoints:
        preds = predict_individual_animal(data_features, endpoint, "dog")
        data[endpoint] = preds

    endpoints = {"monkey_VDss_L_kg", "monkey_CL_mL_min_kg", "monkey_fup"}

    for endpoint in endpoints:
        preds = predict_individual_animal(data_features, endpoint, "monkey")
        data[endpoint] = preds

    endpoints = {"rat_VDss_L_kg", "rat_CL_mL_min_kg", "rat_fup"}

    for endpoint in endpoints:
        preds = predict_individual_animal(data_features, endpoint, "rat")
        data[endpoint] = preds

    return data


def determine_TS(test):

    train_data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/Train_data_log_transformed.csv")

    n_neighbours = 5
    df_master = pd.DataFrame()

    for endpoint in [
        "human_VDss_L_kg",
        "human_CL_mL_min_kg",
        "human_fup",
        "human_mrt",
        "human_thalf",
    ]:

        df = train_data
        df = df.dropna(subset=[endpoint]).reset_index(drop=True)
        df_final_ai = calculate_similarity_test_vs_train(test, df)
        df_final_ai = df_final_ai.sort_values("MFP_Tc", ascending=False)
        df_final_ai = df_final_ai.reset_index(drop=True)

        df_final_ai_2 = pd.DataFrame()
        for compound in df_final_ai["query"].unique():

            compounds_wise = pd.DataFrame()
            compounds_wise = (
                df_final_ai[df_final_ai["query"] == compound]
                .sort_values("MFP_Tc", ascending=False)
                .iloc[:n_neighbours, :]
            )
            df_final_ai_2 = pd.concat([df_final_ai_2, compounds_wise])

        df_final_ai_2 = (
            df_final_ai_2.groupby("query")
            .mean(numeric_only=True)
            .sort_values("MFP_Tc")
            .reset_index(drop=True)
        )
        df_final_ai_2["endpoint"] = endpoint

        df_master = pd.concat([df_master, df_final_ai_2]).reset_index(drop=True)

    # return(df_master.round(1))
    return df_master.round(2)


def predict_VDss(data, features):  # log human_VDss_L_kg model

    loaded_rf = pickle.load(
        open(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/log_human_VDss_L_kg_withanimaldata_artificial_model_FINAL.sav", "rb")
    )

    X = data[features].values
    y_pred = loaded_rf.predict(X)

    return np.round(float(10**y_pred), 2)


def predict_CL(data, features):

    loaded_rf = pickle.load(
        open(
            os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/log_human_CL_mL_min_kg_withanimaldata_artificial_model_FINAL.sav", "rb"
        )
    )

    X = data[features].values
    y_pred = loaded_rf.predict(X)

    return np.round(float(10**y_pred), 2)


def predict_fup(data, features):

    loaded_rf = pickle.load(
        open(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/log_human_fup_withanimaldata_artificial_model_FINAL.sav", "rb")
    )

    X = data[features].values
    y_pred = loaded_rf.predict(X)

    return np.round(float(10**y_pred), 2)


def predict_MRT(data, features):

    loaded_rf = pickle.load(
        open(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/log_human_mrt_withanimaldata_artificial_model_FINAL.sav", "rb")
    )

    X = data[features].values
    y_pred = loaded_rf.predict(X)

    return np.round(float(10**y_pred), 2)


def predict_thalf(data, features):

    loaded_rf = pickle.load(
        open(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/log_human_thalf_withanimaldata_artificial_model_FINAL.sav", "rb")
    )

    X = data[features].values
    y_pred = loaded_rf.predict(X)

    return np.round(float(10**y_pred), 2)


def predict_pk_params(smiles):
    print(banner)
    print(cite)
    smiles_r = standardize(smiles)
    if smiles_r == 'Cannot_do':
        logger.critical('Invalid SMILES.')
        raise Exception('Invalid SMILES.')

    comment = ""
    logger.debug(f'Standized SMILES :: {smiles_r}')
    test = {"smiles_r": [smiles_r]}
    test = pd.DataFrame(test)
    logger.debug(f'Calculating Descriptors')
    test_mfp_Mordred = calcdesc(test)
    logger.debug(f'Calculated {test_mfp_Mordred.shape[1]} Descriptors')

    ts_data = determine_TS(test)
    # folderror = folderror_determiner(test)

    # read from file features
    file = open(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/features_mfp_mordred_animal_artificial_human_modelcolumns.txt", "r")
    file_lines = file.read()
    features_mfp_mordred_animal_columns = file_lines.split("\n")
    features_mfp_mordred_animal_columns = features_mfp_mordred_animal_columns[:-1]

    logger.debug(f'Predicting Parameters for Animal')
    test_mfp_Mordred_animal = predict_animal(test_mfp_Mordred)

    dog_VDss_L_kg = np.round(
        float(10 ** test_mfp_Mordred_animal["dog_VDss_L_kg"].values), 2
    )
    dog_CL_mL_min_kg = np.round(
        float(10 ** test_mfp_Mordred_animal["dog_CL_mL_min_kg"].values), 2
    )
    dog_fup = np.round(float(10 ** test_mfp_Mordred_animal["dog_fup"].values), 2)
    monkey_VDss_L_kg = np.round(
        float(10 ** test_mfp_Mordred_animal["monkey_VDss_L_kg"].values), 2
    )
    monkey_CL_mL_min_kg = np.round(
        float(10 ** test_mfp_Mordred_animal["monkey_CL_mL_min_kg"].values), 2
    )
    monkey_fup = np.round(float(10 ** test_mfp_Mordred_animal["monkey_fup"].values), 2)
    rat_VDss_L_kg = np.round(
        float(10 ** test_mfp_Mordred_animal["rat_VDss_L_kg"].values), 2
    )
    rat_CL_mL_min_kg = np.round(
        float(10 ** test_mfp_Mordred_animal["rat_CL_mL_min_kg"].values), 2
    )
    rat_fup = np.round(float(10 ** test_mfp_Mordred_animal["rat_fup"].values), 2)

    # If mordred human columns fail then model fails, later versions should use mediaan values for nan columns
    human_mordred_median = pd.read_csv(
        os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/Median_mordred_values_human_for_artificial_animal_data_mfp_mrd_model.csv"
    )

    # Check if there are any nan in the features needed for mordred human models:
    if any(
        test_mfp_Mordred_animal[features_mfp_mordred_animal_columns]
        .isna()
        .any()
        .to_list()
    ):
        comment = (
            "Alert: Some Mordred Descriptor generation failed, using median values."
        )

        for i in (
            test_mfp_Mordred_animal[features_mfp_mordred_animal_columns]
            .columns[
                test_mfp_Mordred_animal[features_mfp_mordred_animal_columns]
                .isna()
                .any()
            ]
            .tolist()
        ):
            test_mfp_Mordred_animal[i].fillna(human_mordred_median[i], inplace=True)

    if float(ts_data[ts_data["endpoint"] == "human_thalf"]["MFP_Tc"]) <= 0.20:
        comment = (
            "Alert for predicted thalf: May be out of applicability domain, Tanimoto similarity<=0.20, "
            + comment
        )

    if float(ts_data[ts_data["endpoint"] == "human_mrt"]["MFP_Tc"]) <= 0.20:
        comment = (
            "Alert for predicted MRT: May be out of applicability domain, Tanimoto similarity<=0.20, "
            + comment
        )

    if float(ts_data[ts_data["endpoint"] == "human_fup"]["MFP_Tc"]) <= 0.20:
        comment = (
            "Alert for predicted fup: May be out of applicability domain, Tanimoto similarity<=0.20, "
            + comment
        )

    if float(ts_data[ts_data["endpoint"] == "human_CL_mL_min_kg"]["MFP_Tc"]) <= 0.20:
        comment = (
            "Alert for predicted CL: May be out of applicability domain, Tanimoto similarity<=0.20, "
            + comment
        )

    if float(ts_data[ts_data["endpoint"] == "human_VDss_L_kg"]["MFP_Tc"]) <= 0.20:
        comment = (
            "Alert for predicted VDss: May be out of applicability domain, Tanimoto similarity<=0.20, "
            + comment
        )

    logger.debug(f'Predicting Clearance')
    CL = predict_CL(test_mfp_Mordred_animal, features_mfp_mordred_animal_columns)
    # based on mean
    # CL_fe = np.round(float(folderror[folderror["endpoint"]=="human_CL_mL_min_kg"]["folderror"]), 2)

    # based on polynomial fitting
    # CL_Tc =  float(ts_data[ts_data["endpoint"]=="human_CL_mL_min_kg"]["MFP_Tc"])
    # CL_fe = np.round(13.18 - 31.90*CL_Tc + 21.69*CL_Tc**2, 2)

    # based on kernel regressor
    CL_Tc = float(ts_data[ts_data["endpoint"] == "human_CL_mL_min_kg"]["MFP_Tc"])
    loaded = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/folderror_human_CL_mL_min_kg_generator.sav", "rb"))
    CL_fe = np.round(float(loaded.predict([[CL_Tc]])), 2)

    logger.debug(f'Predicting VDss')
    Vd = predict_VDss(test_mfp_Mordred_animal, features_mfp_mordred_animal_columns)
    # Vd_fe = np.round(float(folderror[folderror["endpoint"]=="human_VDss_L_kg"]["folderror"]), 2)
    # Vd_Tc =  float(ts_data[ts_data["endpoint"]=="human_VDss_L_kg"]["MFP_Tc"])
    # Vd_fe = np.round(4.38 - 5.66*Vd_Tc + 3.28*Vd_Tc**2, 2)

    # based on kernel regressor
    Vd_Tc = float(ts_data[ts_data["endpoint"] == "human_VDss_L_kg"]["MFP_Tc"])
    loaded = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/folderror_human_VDss_L_kg_generator.sav", "rb"))
    Vd_fe = np.round(float(loaded.predict([[Vd_Tc]])), 2)

    logger.debug(f'Predicting Fraction Unbound in Plasma')
    fup = predict_fup(test_mfp_Mordred_animal, features_mfp_mordred_animal_columns)
    # fup_fe = np.round(float(folderror[folderror["endpoint"]=="human_fup"]["folderror"]), 2)
    # fup_Tc =  float(ts_data[ts_data["endpoint"]=="human_fup"]["MFP_Tc"])
    # fup_fe = np.round(10.44 - 26.72*fup_Tc + 20.26*fup_Tc**2, 2)

    # based on kernel regressor
    fup_Tc = float(ts_data[ts_data["endpoint"] == "human_fup"]["MFP_Tc"])
    loaded = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/folderror_human_fup_generator.sav", "rb"))
    fup_fe = np.round(float(loaded.predict([[fup_Tc]])), 2)

    logger.debug(f'Predicting Mean Residance Time')
    MRT = predict_MRT(test_mfp_Mordred_animal, features_mfp_mordred_animal_columns)
    # MRT_fe = np.round(float(folderror[folderror["endpoint"]=="human_mrt"]["folderror"]), 2)
    # MRT_Tc =  float(ts_data[ts_data["endpoint"]=="human_mrt"]["MFP_Tc"])
    # MRT_fe = np.round(10.67 - 23.65*MRT_Tc + 15.56*MRT_Tc**2, 2)

    # based on kernel regressor
    MRT_Tc = float(ts_data[ts_data["endpoint"] == "human_mrt"]["MFP_Tc"])
    loaded = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/folderror_human_mrt_generator.sav", "rb"))
    MRT_fe = np.round(float(loaded.predict([[MRT_Tc]])), 2)

    logger.debug(f'Predicting t1/2')
    thalf = predict_thalf(test_mfp_Mordred_animal, features_mfp_mordred_animal_columns)
    # thalf_fe = np.round(float(folderror[folderror["endpoint"]=="human_thalf"]["folderror"]), 2)
    # thalf_Tc =  float(ts_data[ts_data["endpoint"]=="human_thalf"]["MFP_Tc"])
    # thalf_fe = np.round(7.67 - 13.99*thalf_Tc + 7.82*thalf_Tc**2, 2)

    # based on kernel regressor
    thalf_Tc = float(ts_data[ts_data["endpoint"] == "human_thalf"]["MFP_Tc"])
    loaded = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/folderror_human_thalf_generator.sav", "rb"))
    thalf_fe = np.round(float(loaded.predict([[thalf_Tc]])), 2)

    # Preparations for interactive plot
    file = open(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/features_mordred_columns_human.txt", "r")
    file_lines = file.read()
    features_mordred_columns = file_lines.split("\n")
    features_mordred_columns = features_mordred_columns[:-1]

    test_features = test_mfp_Mordred[features_mordred_columns]

    train_data_features = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + f"/PK_Model_v8/Train_data_features.csv")

    data_features = pd.concat([train_data_features, test_features])
    logger.debug(f'Scaling Data')
    train_data_features_Std = StandardScaler().fit_transform(
        data_features[features_mordred_columns]
    )
    logger.debug(f'Running PCA')
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(train_data_features_Std)
    pcv_1 = np.round(pca.explained_variance_ratio_[0], 4) * 100
    pcv_2 = np.round(pca.explained_variance_ratio_[1], 4) * 100

    logger.debug(f'Done')

    out = {
        "smiles_r": smiles_r,
        "pca_res": pca_res.tolist(),
        "dog_VDss_L_kg": dog_VDss_L_kg,
        "dog_CL_mL_min_kg": dog_CL_mL_min_kg,
        "dog_fup": dog_fup,
        "monkey_VDss_L_kg": monkey_VDss_L_kg,
        "monkey_CL_mL_min_kg": monkey_CL_mL_min_kg,
        "monkey_fup": monkey_fup,
        "rat_VDss_L_kg": rat_VDss_L_kg,
        "rat_CL_mL_min_kg": rat_CL_mL_min_kg,
        "rat_fup": rat_fup,
        # Human parameters
        "CL": CL,
        "CL_fe": CL_fe,
        "CL_Tc": CL_Tc,
        "Vd": Vd,
        "Vd_fe": Vd_fe,
        "Vd_Tc": Vd_Tc,
        "MRT": MRT,
        "MRT_fe": MRT_fe,
        "MRT_Tc": MRT_Tc,
        "thalf": thalf,
        "thalf_fe": thalf_fe,
        "thalf_Tc": thalf_Tc,
        "fup": fup,
        "fup_fe": fup_fe,
        "fup_Tc": fup_Tc,
        "pcv_1": pcv_1,
        "pcv_2": pcv_2,
        # comments
        "comment": comment
    }
    return out

def main():
    parser = argparse.ArgumentParser(description=banner+'\n\n'+abstract+'\n\n'+cite, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--smiles', '-s', '-smi', '--smi', '-smiles', help='Input SMILES string to predict properties')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    results = predict_pk_params(smiles=args.smiles)
    smiles_r = results.get("smiles_r")
    CL_fe = results.get("CL_fe")
    CL = results.get("CL")
    Vd_fe = results.get("Vd_fe")
    Vd = results.get("Vd")
    MRT_fe = results.get("MRT_fe")
    MRT = results.get("MRT")
    thalf_fe = results.get("thalf_fe")
    thalf = results.get("thalf")
    fup_fe = results.get("fup_fe")
    fup = results.get("fup")

    pcv_1 = results.get("pcv_1")
    pcv_2 = results.get("pcv_2")

    CL_range =[np.round(CL/CL_fe,2), np.round(CL*CL_fe, 2)]
    Vd_range =[np.round(Vd/Vd_fe,2) , np.round(Vd*Vd_fe, 2)]
    fup_range =[np.round(fup/fup_fe,2) , np.round(fup*fup_fe, 2)]
    MRT_range =[np.round(MRT/MRT_fe,2), np.round(MRT*MRT_fe, 2)]
    thalf_range =[np.round(thalf/thalf_fe,2), np.round(thalf*thalf_fe, 2)]

    preds_dict = {

            'smiles_r':[smiles_r],

            'Clearance_(CL)':[CL],
            'Clearance_(CL)_units': ["mL/min/kg"],
            'Clearance_(CL)_folderror': [CL_fe],
            'Clearance_(CL)_upperbound': np.max(CL_range),
            'Clearance_(CL)_lowerbound': np.min(CL_range),

            'Volume_of_distribution_(VDss)':[Vd],
            'Volume_of_distribution_(VDss)_units': ["L/kg"],
            'Volume_of_distribution_(VDss)_folderror': [Vd_fe],
            'Volume_of_distribution_(VDss)_upperbound': np.max(Vd_range),
            'Volume_of_distribution_(VDss)_lowerbound': np.min(Vd_range),

            'Fraction_unbound_in_plasma_(fup)':[fup],
            'Fraction_unbound_in_plasma_(fup)_units': ["dimensionless"],
            'Fraction_unbound_in_plasma_(fup)_folderror': [fup_fe],
            'Fraction_unbound_in_plasma_(fup)_upperbound': np.max(fup_range),
            'Fraction_unbound_in_plasma_(fup)_lowerbound': np.min(fup_range),

            'Mean_Residence_Time_(MRT)':[MRT],
            'Mean_Residence_Time_(MRT)_units': ["h"],
            'Mean_Residence_Time_(MRT)_folderror': [MRT_fe],
            'Mean_Residence_Time_(MRT)_upperbound': np.max(MRT_range),
            'Mean_Residence_Time_(MRT)_lowerbound': np.min(MRT_range),

            'Half_life_(thalf)':[thalf],
            'Half_life_(thalf)_units': ["h"],
            'Half_life_(thalf)_folderror': [thalf_fe],
            'Half_life_(thalf)_upperbound': np.max(thalf_range),
            'Half_life_(thalf)_lowerbound': np.min(thalf_range),

            }
    filename = f"PKSmart_Result_{formatted_time}_{formatted_date}.csv"
    pd.DataFrame(preds_dict).to_csv(filename, index=False)
    logger.critical(f'Results are saved at {filename}')

if __name__ == "__main__":
    main()
