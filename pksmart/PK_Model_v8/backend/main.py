from urllib.parse import unquote, quote
from fastapi import FastAPI
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import pickle
from mordred import Calculator, descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from rdkit import Chem

from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from bokeh.io import output_notebook

output_notebook()


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
    # save as csv


def determine_TS(test):

    train_data = pd.read_csv("../Train_data_log_transformed.csv")

    n_neighbours = 5
    list_of_lists = []
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


def predict_individual_animal(data_features, endpoint, animal):  # predict animal data

    # Read columns needed for rat data
    file = open(f"../features_mfp_mordred_columns_{animal}_model.txt", "r")
    file_lines = file.read()
    features = file_lines.split("\n")
    features = features[:-1]

    loaded_rf = pickle.load(open(f"../log_{endpoint}_model_FINAL.sav", "rb"))

    X = data_features[features]

    animalmedian = pd.read_csv(
        f"../Median_mordred_values_{animal}_for_artificial_animal_data_mfp_mrd_model.csv"
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


def predict_VDss(data, features):  # log human_VDss_L_kg model

    loaded_rf = pickle.load(
        open("../log_human_VDss_L_kg_withanimaldata_artificial_model_FINAL.sav", "rb")
    )

    X = data[features].values
    y_pred = loaded_rf.predict(X)

    return np.round(float(10**y_pred), 2)


def predict_CL(data, features):

    loaded_rf = pickle.load(
        open(
            "../log_human_CL_mL_min_kg_withanimaldata_artificial_model_FINAL.sav", "rb"
        )
    )

    X = data[features].values
    y_pred = loaded_rf.predict(X)

    return np.round(float(10**y_pred), 2)


def predict_fup(data, features):

    loaded_rf = pickle.load(
        open("../log_human_fup_withanimaldata_artificial_model_FINAL.sav", "rb")
    )

    X = data[features].values
    y_pred = loaded_rf.predict(X)

    return np.round(float(10**y_pred), 2)


def predict_MRT(data, features):

    loaded_rf = pickle.load(
        open("../log_human_mrt_withanimaldata_artificial_model_FINAL.sav", "rb")
    )

    X = data[features].values
    y_pred = loaded_rf.predict(X)

    return np.round(float(10**y_pred), 2)


def predict_thalf(data, features):

    loaded_rf = pickle.load(
        open("../log_human_thalf_withanimaldata_artificial_model_FINAL.sav", "rb")
    )

    X = data[features].values
    y_pred = loaded_rf.predict(X)

    return np.round(float(10**y_pred), 2)


def mol2svg(mol):
    d2d = rdMolDraw2D.MolDraw2DSVG(200, 100)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()


app = FastAPI()


@app.get("/")
def read_root():

    print("Enter SMILES")
    smiles = "C#CCCCC(=O)c1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1"
    smiles = quote(smiles)
    return {"smiles": smiles}


@app.get("/smiles/{smiles}")
def read_item(smiles: str):

    comment = ""
    smiles = unquote(smiles)

    smiles_r = standardize(smiles)
    test = {"smiles_r": [smiles_r]}
    test = pd.DataFrame(test)
    test_mfp_Mordred = calcdesc(test)

    ts_data = determine_TS(test)
    # folderror = folderror_determiner(test)

    # read from file features
    file = open("../features_mfp_mordred_animal_artificial_human_modelcolumns.txt", "r")
    file_lines = file.read()
    features_mfp_mordred_animal_columns = file_lines.split("\n")
    features_mfp_mordred_animal_columns = features_mfp_mordred_animal_columns[:-1]

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
        f"../Median_mordred_values_human_for_artificial_animal_data_mfp_mrd_model.csv"
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

    CL = predict_CL(test_mfp_Mordred_animal, features_mfp_mordred_animal_columns)
    # based on mean
    # CL_fe = np.round(float(folderror[folderror["endpoint"]=="human_CL_mL_min_kg"]["folderror"]), 2)

    # based on polynomial fitting
    # CL_Tc =  float(ts_data[ts_data["endpoint"]=="human_CL_mL_min_kg"]["MFP_Tc"])
    # CL_fe = np.round(13.18 - 31.90*CL_Tc + 21.69*CL_Tc**2, 2)

    # based on kernel regressor
    CL_Tc = float(ts_data[ts_data["endpoint"] == "human_CL_mL_min_kg"]["MFP_Tc"])
    loaded = pickle.load(open(f"../folderror_human_CL_mL_min_kg_generator.sav", "rb"))
    CL_fe = np.round(float(loaded.predict([[CL_Tc]])), 2)

    Vd = predict_VDss(test_mfp_Mordred_animal, features_mfp_mordred_animal_columns)
    # Vd_fe = np.round(float(folderror[folderror["endpoint"]=="human_VDss_L_kg"]["folderror"]), 2)
    # Vd_Tc =  float(ts_data[ts_data["endpoint"]=="human_VDss_L_kg"]["MFP_Tc"])
    # Vd_fe = np.round(4.38 - 5.66*Vd_Tc + 3.28*Vd_Tc**2, 2)

    # based on kernel regressor
    Vd_Tc = float(ts_data[ts_data["endpoint"] == "human_VDss_L_kg"]["MFP_Tc"])
    loaded = pickle.load(open(f"../folderror_human_VDss_L_kg_generator.sav", "rb"))
    Vd_fe = np.round(float(loaded.predict([[Vd_Tc]])), 2)

    fup = predict_fup(test_mfp_Mordred_animal, features_mfp_mordred_animal_columns)
    # fup_fe = np.round(float(folderror[folderror["endpoint"]=="human_fup"]["folderror"]), 2)
    # fup_Tc =  float(ts_data[ts_data["endpoint"]=="human_fup"]["MFP_Tc"])
    # fup_fe = np.round(10.44 - 26.72*fup_Tc + 20.26*fup_Tc**2, 2)

    # based on kernel regressor
    fup_Tc = float(ts_data[ts_data["endpoint"] == "human_fup"]["MFP_Tc"])
    loaded = pickle.load(open(f"../folderror_human_fup_generator.sav", "rb"))
    fup_fe = np.round(float(loaded.predict([[fup_Tc]])), 2)

    MRT = predict_MRT(test_mfp_Mordred_animal, features_mfp_mordred_animal_columns)
    # MRT_fe = np.round(float(folderror[folderror["endpoint"]=="human_mrt"]["folderror"]), 2)
    # MRT_Tc =  float(ts_data[ts_data["endpoint"]=="human_mrt"]["MFP_Tc"])
    # MRT_fe = np.round(10.67 - 23.65*MRT_Tc + 15.56*MRT_Tc**2, 2)

    # based on kernel regressor
    MRT_Tc = float(ts_data[ts_data["endpoint"] == "human_mrt"]["MFP_Tc"])
    loaded = pickle.load(open(f"../folderror_human_mrt_generator.sav", "rb"))
    MRT_fe = np.round(float(loaded.predict([[MRT_Tc]])), 2)

    thalf = predict_thalf(test_mfp_Mordred_animal, features_mfp_mordred_animal_columns)
    # thalf_fe = np.round(float(folderror[folderror["endpoint"]=="human_thalf"]["folderror"]), 2)
    # thalf_Tc =  float(ts_data[ts_data["endpoint"]=="human_thalf"]["MFP_Tc"])
    # thalf_fe = np.round(7.67 - 13.99*thalf_Tc + 7.82*thalf_Tc**2, 2)

    # based on kernel regressor
    thalf_Tc = float(ts_data[ts_data["endpoint"] == "human_thalf"]["MFP_Tc"])
    loaded = pickle.load(open(f"../folderror_human_thalf_generator.sav", "rb"))
    thalf_fe = np.round(float(loaded.predict([[thalf_Tc]])), 2)

    # Preparations for interactive plot
    file = open("../features_mordred_columns_human.txt", "r")
    file_lines = file.read()
    features_mordred_columns = file_lines.split("\n")
    features_mordred_columns = features_mordred_columns[:-1]

    test_features = test_mfp_Mordred[features_mordred_columns]

    train_data_features = pd.read_csv("../Train_data_features.csv")

    data_features = pd.concat([train_data_features, test_features])
    train_data_features_Std = StandardScaler().fit_transform(
        data_features[features_mordred_columns]
    )
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(train_data_features_Std)
    pcv_1 = np.round(pca.explained_variance_ratio_[0], 4) * 100
    pcv_2 = np.round(pca.explained_variance_ratio_[1], 4) * 100

    return {
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
        "comment": comment,
    }
