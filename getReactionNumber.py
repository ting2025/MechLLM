import torch
from molscribe import MolScribe
from huggingface_hub import hf_hub_download
import json
import cv2
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from PIL import Image 
import io
import re
from IPython.display import display, Image as IPythonImage
import sys
import torch
from rxn import RxnScribe

def getNumber(IMAGE_PATH):
    ##### Graph recognition
    device = torch.device('cpu')
    ckpt_path = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth')
    model = MolScribe(ckpt_path, device=device)
    
    # Predict structure
    prediction = model.predict_image_file(IMAGE_PATH, return_atoms_bonds=True)
    smiles = prediction['smiles']
    molfile = prediction['molfile']
    
    # Read the image using OpenCV
    cv_image = cv2.imread(IMAGE_PATH)

    # Draw prediction numbers onto the image; this returns an annotated NumPy array (number_image) 
    # and the number of detected elements (number_count)
    number_image, number_count = model.draw_prediction_number(prediction, cv_image)

    im = Image.fromarray(number_image)


    ##### Text recognition
    rxn_model_path = '/home/ctleungaf/ChemEagle/RxnScribe_main/ckpt/pix2seq_reaction_full.ckpt'
    rxnmodel = RxnScribe(rxn_model_path, device)
    rxnpredictions = rxnmodel.predict_image_file(IMAGE_PATH, molscribe=True, ocr=True)

    # Convert original image to PIL
    pil_image = Image.open(IMAGE_PATH).convert("RGB")
    text_molecules, new_index = rxnmodel.process_text_based_molecules(rxnpredictions,
        pil_image, 
        starting_index=number_count + 1
    )

    # Convert the annotated OpenCV (NumPy) image to a PIL image
    pil_number_image = Image.fromarray(number_image).convert("RGB")

    # This likely returns a NumPy array
    result_image_array = rxnmodel.draw_text_molecules(pil_number_image, text_molecules)

    # Convert result from NumPy array to PIL Image so we can call .show() / .save() etc.
    result_image_pil = Image.fromarray(result_image_array)


    return result_image_pil
