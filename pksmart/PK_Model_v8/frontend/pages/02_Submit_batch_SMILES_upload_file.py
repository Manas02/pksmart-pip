
from asyncore import write
from urllib.parse import unquote, quote
import requests
import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
import pickle
from mordred import Calculator, descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.feature_selection import VarianceThreshold
from itertools import compress
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit import RDPaths
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import MolDraw2DSVG

from bokeh.plotting import ColumnDataSource, figure, output_file, show
from sklearn.decomposition import PCA
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from bokeh.io import output_notebook
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table
from sqlalchemy import Column, BIGINT, Index
from sqlalchemy.orm import mapper
from sqlalchemy.ext.declarative import declarative_base

def mol2svg(mol):
    d2d = rdMolDraw2D.MolDraw2DSVG(200, 100)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def main():
    
    st.set_page_config(
    page_title="DrugWise PK Predictor",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
    )
    
    st.image("logo_front.png", width=500)
    #st.title("DrugWise PK Predictor")
    st.write(
    """
    [![Follow](https://img.shields.io/twitter/follow/srijitseal?style=social)](https://www.twitter.com/srijitseal)
    """
)
    st.download_button('Download Sample Input file', "CCCO\nCCOC\nCCOCC")
    uploaded_file = st.file_uploader("Enter SMILES of the query compounds, one in each line in a .txt. file")
    
    if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file, header=None)
        #st.write(dataframe)
        smiles_list= list(dataframe.values.flatten()) 

    else:
        st.write("No file uploaded!")

    #predict
    results = ''
    CL =''
    CL_fe =''
    Vd =''
    Vd_fe =''
    MRT =''
    MRT_fe =''
    thalf =''
    thalf_fe =''
    fup =''
    fup_fe =''

    dog_VDss_L_kg =''
    dog_CL_mL_min_kg =''
    dog_fup =''
    monkey_VDss_L_kg=''
    monkey_CL_mL_min_kg=''
    monkey_fup =''
    rat_VDss_L_kg =''
    rat_CL_mL_min_kg =''
    rat_fup =''
    comment=''

    smiles_r=''
    my_bar = st.progress(0)
    percent_complete = 0
    
    if st.button('Predict PK parameters'):

        with st.spinner('Predicting PK parameters...'):

            preds_dict_download_batch=pd.DataFrame()

            if(smiles_list==['']):
                st.write("No input provided")
                st.success("Fail!")
                st.stop()
            
            for smiles in smiles_list:

                try:

                    smiles_quoted =  quote(smiles)
                    results = requests.get(f"http://127.0.0.1:8000/smiles/{smiles_quoted}")
                    results = results.json()

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

                    dog_VDss_L_kg =results.get("dog_VDss_L_kg")
                    dog_CL_mL_min_kg =results.get("dog_CL_mL_min_kg")
                    dog_fup =results.get("dog_fup")
                    monkey_VDss_L_kg=results.get("monkey_VDss_L_kg")
                    monkey_CL_mL_min_kg=results.get("monkey_CL_mL_min_kg")
                    monkey_fup =results.get("monkey_fup")
                    rat_VDss_L_kg =results.get("rat_VDss_L_kg")
                    rat_CL_mL_min_kg =results.get("rat_CL_mL_min_kg")
                    rat_fup =results.get("rat_fup")

                    comment= results.get("comment")

                    CL_range =[np.round(CL/CL_fe,2), np.round(CL*CL_fe, 2)]
                    Vd_range =[np.round(Vd/Vd_fe,2) , np.round(Vd*Vd_fe, 2)]
                    #fup_range =[np.round(fup/fup_fe,2) , np.round(fup*fup_fe, 2)]
                    MRT_range =[np.round(MRT/MRT_fe,2), np.round(MRT*MRT_fe, 2)]
                    thalf_range =[np.round(thalf/thalf_fe,2), np.round(thalf*thalf_fe, 2)]

                    if(np.round(fup*fup_fe, 2)<=1):
                        fup_range =[np.round(fup/fup_fe,2) , np.round(fup*fup_fe, 2)]
                    
                    else:
                        fup_range =[np.round(fup/fup_fe,2), 1.00]

                    preds_dict_download = {

                        'smiles':[smiles],
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

                        "Dog_Volume_of_distribution_(VDss)": [dog_VDss_L_kg],
                        'Dog_Volume_of_distribution_(VDss)_units': ["L/kg"],

                        "Dog_Clearance_(CL)": [dog_CL_mL_min_kg],
                        'Dog_Clearance_(CL)_units': ["mL/min/kg"],

                        "Dog_Fraction_unbound_in_plasma_(fup)": [dog_fup],
                        'Dog_Fraction_unbound_in_plasma_(fup)_units': ["dimensionless"],

                        "Monkey_Volume_of_distribution_(VDss)": [monkey_VDss_L_kg],
                        'Monkey_Volume_of_distribution_(VDss)_units': ["L/kg"],

                        "Monkey_Clearance_(CL)": [monkey_CL_mL_min_kg],
                        'Monkey_Clearance_(CL)_units': ["mL/min/kg"],

                        "Monkey_Fraction_unbound_in_plasma (fup)": [monkey_fup],
                        'Monkey_Fraction_unbound_in_plasma_(fup)_units': ["dimensionless"],

                        "Rat_Volume_of_distribution_(VDss)": [rat_VDss_L_kg],
                        'Rat_Volume_of_distribution_(VDss)_units': ["L/kg"],

                        "Rat_Clearance_(CL)": [rat_CL_mL_min_kg],
                        'Rat_Clearance_(CL)_units': ["mL/min/kg"],

                        "Rat_Fraction_unbound_in_plasma_(fup)": [rat_fup],
                        'Rat_Fraction_unbound_in_plasma_(fup)_units': ["dimensionless"],

                        "comment":  comment
                        }
                        
                    preds_dict_download= pd.DataFrame(preds_dict_download)
                    preds_dict_download_batch = pd.concat([preds_dict_download_batch, preds_dict_download], axis=0)

                    st.write(f"{smiles} Successful!")
                    
                    percent_complete = percent_complete + 1
                    progress = percent_complete/len(smiles_list)
                    my_bar.progress(progress)

                except:
                    
                    if (smiles==' '):
                        st.write(f"Empty row : Unsuccessful!")

                    elif (smiles==''):
                        st.write(f"Empty row : Unsuccessful!")
                    
                    else:
                        st.write(f"{smiles} Unsuccessful! Check SMILES")
                    
                    preds_dict_download = {

                        'smiles':[smiles],
                        'smiles_r':["Unsuccessful! Check SMILES"],

                        'Clearance_(CL)':[0],
                        'Clearance_(CL)_units': ["NA"],
                        'Clearance_(CL)_folderror': [0],
                        'Clearance_(CL)_upperbound': [0],
                        'Clearance_(CL)_lowerbound': [0],

                        'Volume_of_distribution_(VDss)':[0],
                        'Volume_of_distribution_(VDss)_units': ["NA"],
                        'Volume_of_distribution_(VDss)_folderror': [0],
                        'Volume_of_distribution_(VDss)_upperbound': [0],
                        'Volume_of_distribution_(VDss)_lowerbound': [0],

                        'Fraction_unbound_in_plasma_(fup)':[0],
                        'Fraction_unbound_in_plasma_(fup)_units': ["NA"],
                        'Fraction_unbound_in_plasma_(fup)_folderror': [0],
                        'Fraction_unbound_in_plasma_(fup)_upperbound': [0],
                        'Fraction_unbound_in_plasma_(fup)_lowerbound': [0],

                        'Mean_Residence_Time_(MRT)':[0],
                        'Mean_Residence_Time_(MRT)_units': ["NA"],
                        'Mean_Residence_Time_(MRT)_folderror': [0],
                        'Mean_Residence_Time_(MRT)_upperbound': [0],
                        'Mean_Residence_Time_(MRT)_lowerbound': [0],

                        'Half_life_(thalf)':[0],
                        'Half_life_(thalf)_units': ["NA"],
                        'Half_life_(thalf)_folderror': [0],
                        'Half_life_(thalf)_upperbound': [0],
                        'Half_life_(thalf)_lowerbound': [0],

                        "Dog_Volume_of_distribution_(VDss)": [0],
                        'Dog_Volume_of_distribution_(VDss)_units': ["NA"],

                        "Dog_Clearance_(CL)": [0],
                        'Dog_Clearance_(CL)_units': ["NA"],

                        "Dog_Fraction_unbound_in_plasma_(fup)": [0],
                        'Dog_Fraction_unbound_in_plasma_(fup)_units': ["NA"],

                        "Monkey_Volume_of_distribution_(VDss)": [0],
                        'Monkey_Volume_of_distribution_(VDss)_units': ["NA"],

                        "Monkey_Clearance_(CL)": [0],
                        'Monkey_Clearance_(CL)_units': ["NA"],

                        "Monkey_Fraction_unbound_in_plasma (fup)": [0],
                        'Monkey_Fraction_unbound_in_plasma_(fup)_units': ["NA"],

                        "Rat_Volume_of_distribution_(VDss)": [0],
                        'Rat_Volume_of_distribution_(VDss)_units': ["NA"],

                        "Rat_Clearance_(CL)": [0],
                        'Rat_Clearance_(CL)_units': ["NA"],

                        "Rat_Fraction_unbound_in_plasma_(fup)": [0],
                        'Rat_Fraction_unbound_in_plasma_(fup)_units': ["NA"],

                        "comment":  ["Failed: Check SMILES"]
                        }
                    
                    preds_dict_download= pd.DataFrame(preds_dict_download)
                    preds_dict_download_batch = pd.concat([preds_dict_download_batch, preds_dict_download], axis=0)
                    
                    percent_complete = percent_complete + 1
                    progress = percent_complete/len(smiles_list)
                    my_bar.progress(progress)

            preds_dict_download_batch = preds_dict_download_batch.reset_index(drop=True)
            
            preds_dict_download_batch_2 = convert_df(preds_dict_download_batch)
    
            #Dowload Predictions

            st.download_button(
            label="Download predictions as CSV",
            data=preds_dict_download_batch_2,
            file_name='Batch_Human_PK_prediction.csv',
            mime='text/csv',
            )
            
            st.table(preds_dict_download_batch)
            
            
            

        st.success("Complete")

if __name__ == '__main__': 
    main()   

