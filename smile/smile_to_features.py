from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd

def SMILE_to_desc_dict(SMILE, descriptors_set="original"):
    mol = Chem.MolFromSmiles(SMILE)
    if descriptors_set == "original":
        desc = {
            "SMILES": SMILE,
            "Molecular Weight": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "Hydrogen Bond Donors": Descriptors.NumHDonors(mol),
            "Hydrogen Bond Acceptors": Descriptors.NumHAcceptors(mol),
            "Topological Polar Surface Area": Descriptors.TPSA(mol),
            "Molar Refractivity": Descriptors.MolMR(mol),
            "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
            "Aromatic Rings": Descriptors.NumAromaticRings(mol),
            "Heavy Atoms": Descriptors.HeavyAtomCount(mol)
        }
    elif descriptors_set == "ZINC":
        desc = {
            "SMILES": SMILE,
            "Molecular Weight": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "Rings": Descriptors.RingCount(mol),
            "Hetero Atoms": Descriptors.NumHeteroatoms(mol),
            "Fraction sp3": Descriptors.FractionCSP3(mol),
            "Hydrogen Bond Donors": Descriptors.NumHDonors(mol),
            "Hydrogen Bond Acceptors": Descriptors.NumHAcceptors(mol),
            "Topological Polar Surface Area": Descriptors.TPSA(mol),
            "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
            "Heavy Atoms": Descriptors.HeavyAtomCount(mol)
        }
    elif descriptors_set == "ZINC expanded":
        desc = {
            "SMILES": SMILE,
            "Molecular Weight": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "Rings": Descriptors.RingCount(mol),
            "Hetero Atoms": Descriptors.NumHeteroatoms(mol),
            "Fraction sp3": Descriptors.FractionCSP3(mol),
            "Hydrogen Bond Donors": Descriptors.NumHDonors(mol),
            "Hydrogen Bond Acceptors": Descriptors.NumHAcceptors(mol),
            "Topological Polar Surface Area": Descriptors.TPSA(mol),
            "Molar Refractivity": Descriptors.MolMR(mol),
            "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
            "Aromatic Rings": Descriptors.NumAromaticRings(mol),
            "Heavy Atoms": Descriptors.HeavyAtomCount(mol)
        }
    else:
        print("Must choose either original, ZINC or ZINC expanded for the descriptor set.")
    return desc

# print(SMILE_to_desc_dict("O=C(O)CCCc1ccc(N(CCCl)CCCl)cc1", "ZINC"))


def SMILE_series_to_features_df(SMILE_series, descriptors_set="original", drop_duplicates=True):
    features_df = pd.DataFrame(list(SMILE_series.apply(SMILE_to_desc_dict, descriptors_set=descriptors_set)))
    if drop_duplicates:
        features_df = features_df.drop_duplicates(subset=features_df.columns[1:])
    return features_df