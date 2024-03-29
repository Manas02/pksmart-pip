
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
    page_title="Human PK Predictor",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
    )
    
    st.image("logo.png", width=500)
    #st.title("Human PK Predictor")
    st.write(
    """
    [![Follow](https://img.shields.io/twitter/follow/srijitseal?style=social)](https://www.twitter.com/srijitseal)
    """
)
    
    smiles=st.text_input("Enter SMILES of the query compound:")
    smiles =  quote(smiles) 
    
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

    pca_res= []
    pcv_1 =''
    pcv_2 =''

    smiles_r=''
    molsvgs_train = ''
    data_train=dict()

    preds_dict={}
    preds= pd.DataFrame(columns=['Endpoint Predicted', 'Predicted value'
                                    ,'Predicted Fold error', 'Predicted Range'])
    
    if st.button('Predict Human PK parameters'):

        with st.spinner('Predicting PK parameters...'):

            results = requests.get(f"http://127.0.0.1:8000/smiles/{smiles}")
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

            pcv_1 = results.get("pcv_1")
            pcv_2 = results.get("pcv_2")

            CL_range =[np.round(CL/CL_fe,2), np.round(CL*CL_fe, 2)]
            Vd_range =[np.round(Vd/Vd_fe,2) , np.round(Vd*Vd_fe, 2)]
            fup_range =[np.round(fup/fup_fe,2) , np.round(fup*fup_fe, 2)]
            MRT_range =[np.round(MRT/MRT_fe,2), np.round(MRT*MRT_fe, 2)]
            thalf_range =[np.round(thalf/thalf_fe,2), np.round(thalf*thalf_fe, 2)]



            molecule = Chem.MolFromSmiles(smiles_r)
            
            st.image(Draw.MolToImage(molecule), width=200)
            st.write(f"Standardised Query compound: {smiles_r}")

            preds_dict = {'Endpoint Predicted':['Clearance (CL)', 
            'Volume of distribution (VDss)', 
            "Fraction unbound in plasma (fup)", 
            "Mean Residence Time (MRT)", 
            "Half-life (thalf)"],

            'Predicted value':[f'{CL} mL/min/kg', f'{Vd} L/kg', f'{fup}', f'{MRT} h', f'{thalf} h'],
            'Predicted Fold error':[CL_fe, Vd_fe, fup_fe, MRT_fe, thalf_fe],
            'Predicted Range':[ f'{np.round(CL/CL_fe,2)} to {np.round(CL*CL_fe, 2)} mL/min/kg', 
            f'{np.round(Vd/Vd_fe,2)} to {np.round(Vd*Vd_fe, 2)} L/kg', 
            f'{np.round(fup/fup_fe,2)} to {np.round(fup*fup_fe, 2)}', 
            f'{np.round(MRT/MRT_fe,2)} to {np.round(MRT*MRT_fe, 2)} h', 
            f'{np.round(thalf/thalf_fe,2)} to {np.round(thalf*thalf_fe, 2)} h', ]
            }

            preds_dict_download = {

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
    
            preds = pd.DataFrame(preds_dict)
            st.table(preds)
            
            preds_dict_download = pd.DataFrame(preds_dict_download)
            preds_dict_download = convert_df(preds_dict_download)

            #interactive plot
            
            molsvgs_train = pickle.load(open("../molsvgs_train.sav", 'rb'))
            pca_res = results.get("pca_res")
            pca_res = np.array(pca_res)

            train_data_features= pd.read_csv("../Train_data_features.csv") 
            train_data_features["Data"] = "Train"

            human_VDss_L_kg=10**train_data_features[:]['human_VDss_L_kg']
            human_CL_mL_min_kg=10**train_data_features[:]['human_CL_mL_min_kg']
            human_fup=10**train_data_features[:]['human_fup']
            human_mrt=10**train_data_features[:]['human_mrt']
            human_thalf=10**train_data_features[:]['human_thalf']

            file = open("../features_mfp_mordred_columns_human.txt", "r")
            file_lines = file.read()
            features_mfp_mordred_columns = file_lines.split("\n")
            features_mfp_mordred_columns = features_mfp_mordred_columns[:-1]
        
            mols_test = Chem.MolFromSmiles(smiles_r)
            molsvgs_test = [mol2svg(mols_test)]

            data_train = dict(
                x= pca_res[:-1][:,0],
                y=pca_res[:-1][:,1],
                img = molsvgs_train,
                human_VDss_L_kg=human_VDss_L_kg,
                human_CL_mL_min_kg=human_CL_mL_min_kg,
                human_fup=human_fup,
                human_mrt=human_mrt,
                human_thalf=human_thalf
                )

            data_test = dict(
            
                x= pca_res[-1:][:,0],
                y=pca_res[-1:][:,1],
                img = molsvgs_test[-1:],
                human_CL_mL_min_kg=[np.round(CL,2)],
            )


            source_train = ColumnDataSource(data_train)
            source_test= ColumnDataSource(data_test)

            TOOLTIPS = """
            <div>
            human_VDss_L_kg: @human_VDss_L_kg<br>
            human_CL_mL_min_kg: @human_CL_mL_min_kg<br>
            human_fup: @human_fup<br>
            human_mrt: @human_mrt<br>
            human_thalf: @human_thalf<br>
            @img{safe}
            </div>
            """

            st.write("Projecting query compound in the structural-physicochemical space of the training data:")

            p = figure(plot_width=600, plot_height=600, tooltips=TOOLTIPS,
                    title=f"Principal Component Analysis using selected Mordred descriptors ({np.round(pcv_1+pcv_2, 2)}% variance explained)",
                    x_axis_label=f"Principal Component 1 ({np.round(pcv_1, 2)}% explained variance)",
                    y_axis_label=f"Principal Component 2 ({np.round(pcv_2, 2)}% explained variance)")
            
            p.circle('x', 'y', size=10, source=source_train, color="red", legend_label="Training data")
            p.circle('x', 'y', size=10, source=source_test, color="blue", legend_label="Query Compound")
            
            p.legend.location = "top_left"

            # change border and background of legend
            p.legend.border_line_width = 3
            p.legend.border_line_color = "black"


            st.bokeh_chart(p, use_container_width=True)

            st.write("This model is currently under development...")

            #st.write("Predicted CL", np.round(CL,2), "ml/min/kg")
            #st.write("Expected Fold Error:", np.round(CL_fe,2))
            #st.write("Expected range of prediction:", np.round(CL/CL_fe,2), " to ", np.round(CL*CL_fe, 2), "ml/min/kg")

            #st.write("Predicted Vdss", np.round(Vd,2), "L/kg")
            #st.write("Expected Fold Error:", np.round(CL_fe,2))
            #st.write("Expected range of prediction:", np.round(Vd/Vd_fe,2), " to ", np.round(Vd*Vd_fe, 2), "L/kg")

            #st.write("Predicted fup", np.round(fup,2))
            #st.write("Expected Fold Error:", np.round(fup_fe,2))
            #st.write("Expected range of prediction:", np.round(fup/fup_fe,2), " to ", np.round(fup*fup_fe, 2), "")

            #st.write("Predicted MRT", np.round(MRT,2))
            #st.write("Expected Fold Error:", np.round(MRT_fe,2))
            #st.write("Expected range of prediction:", np.round(MRT/MRT,2), " to ", np.round(MRT*MRT_fe, 2), "")
            
            #st.write("Predicted thalf", np.round(thalf,2))
            #st.write("Expected Fold Error:", np.round(thalf_fe,2))
            #st.write("Expected range of prediction:", np.round(thalf/thalf_fe,2), " to ", np.round(thalf*thalf_fe, 2), "")


            #Dowload Predictions

            st.download_button(
            label="Download predictions as CSV",
            data=preds_dict_download,
            file_name='Human_PK_prediction.csv',
            mime='text/csv',
            )

        st.success("Success!")

if __name__ == '__main__': 
    main()   

