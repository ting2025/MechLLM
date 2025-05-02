import os
import argparse
from typing import List
import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .pix2seq import build_pix2seq_model
from .tokenizer import get_tokenizer
from .dataset import make_transforms
from .data import postprocess_reactions, ReactionImageData

from molscribe import MolScribe
from huggingface_hub import hf_hub_download
import easyocr
import re
import cv2


class RxnScribe:

    def __init__(self, model_path, device=None):
        """
        RxnScribe Interface
        :param model_path: path of the model checkpoint.
        :param device: torch device, defaults to be CPU.
        """
        args = self._get_args()
        args.format = 'reaction'
        states = torch.load(model_path, map_location=torch.device('cpu'))
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.tokenizer = get_tokenizer(args)
        self.model = self.get_model(args, self.tokenizer, self.device, states['state_dict'])
        self.transform = make_transforms('test', augment=False, debug=False)
        self.molscribe = self.get_molscribe()
        self.ocr_model = self.get_ocr_model()

    def _get_args(self):
        parser = argparse.ArgumentParser()
        # * Backbone
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        # * Transformer
        parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=1024, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--pre_norm', action='store_true')
        # Data
        parser.add_argument('--format', type=str, default='reaction')
        parser.add_argument('--input_size', type=int, default=1333)
        parser.add_argument('--use_hf_transformer', action='store_true', 
                        help="Flag to use HuggingFace transformer")

        args = parser.parse_args([])
        args.pix2seq = True
        args.pix2seq_ckpt = None
        args.pred_eos = True
        return args

    def get_model(self, args, tokenizer, device, model_states):
        def remove_prefix(state_dict):
            return {k.replace('model.', ''): v for k, v in state_dict.items()}

        model = build_pix2seq_model(args, tokenizer[args.format])
        model.load_state_dict(remove_prefix(model_states), strict=False)
        model.to(device)
        model.eval()
        return model

    def get_molscribe(self):
        ckpt_path = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
        molscribe = MolScribe(ckpt_path, device=self.device)
        return molscribe

    def get_ocr_model(self):
        reader = easyocr.Reader(['en'], gpu=(self.device.type == 'cuda'))
        return reader

    def predict_images(self, input_images: List, batch_size=16, molscribe=False, ocr=False):
        # images: a list of PIL images
        device = self.device
        tokenizer = self.tokenizer['reaction']
        predictions = []
        for idx in range(0, len(input_images), batch_size):
            batch_images = input_images[idx:idx+batch_size]
            images, refs = zip(*[self.transform(image) for image in batch_images])
            images = torch.stack(images, dim=0).to(device)
            with torch.no_grad():
                pred_seqs, pred_scores = self.model(images, max_len=tokenizer.max_len)
            for i, (seqs, scores) in enumerate(zip(pred_seqs, pred_scores)):
                reactions = tokenizer.sequence_to_data(seqs.tolist(), scores.tolist(), scale=refs[i]['scale'])
                reactions = postprocess_reactions(
                    reactions,
                    image=input_images[i],
                    molscribe=self.molscribe if molscribe else None,
                    ocr=self.ocr_model if ocr else None
                )
                predictions.append(reactions)
        return predictions

    def predict_image(self, image, **kwargs):
        predictions = self.predict_images([image], **kwargs)
        return predictions[0]

    def predict_image_files(self, image_files: List, **kwargs):
        input_images = []
        for path in image_files:
            image = PIL.Image.open(path).convert("RGB")
            input_images.append(image)
        return self.predict_images(input_images, **kwargs)

    def predict_image_file(self, image_file: str, **kwargs):
        predictions = self.predict_image_files([image_file], **kwargs)
        return predictions[0]

    def draw_predictions(self, predictions, image=None, image_file=None):
        results = []
        assert image or image_file
        data = ReactionImageData(predictions=predictions, image=image, image_file=image_file)
        h, w = np.array([data.height, data.width]) * 10 / max(data.height, data.width)
        for r in data.pred_reactions:
            fig, ax = plt.subplots(figsize=(w, h))
            fig.tight_layout()
            canvas = FigureCanvasAgg(fig)
            ax.imshow(data.image)
            ax.axis('off')
            r.draw(ax)
            canvas.draw()
            buf = canvas.buffer_rgba()
            results.append(np.asarray(buf))
            plt.close(fig)
        return results
    
    def draw_predictions_cropped(self, predictions, image=None, image_file=None):
        results = []
        assert image or image_file
        data = ReactionImageData(predictions=predictions, image=image, image_file=image_file)
        h, w = np.array([data.height, data.width]) * 10 / max(data.height, data.width)
        for r in data.pred_reactions:
            fig, ax = plt.subplots(figsize=(w, h))
            fig.tight_layout()
            canvas = FigureCanvasAgg(fig)
            ax.imshow(data.image)
            ax.axis('off')
            r.draw(ax)
            canvas.draw()
            buf = canvas.buffer_rgba()
            results.append(np.asarray(buf))
            plt.close(fig)
        return results

    def draw_predictions_combined(self, predictions, image=None, image_file=None):
        assert image or image_file
        data = ReactionImageData(predictions=predictions, image=image, image_file=image_file)
        h, w = np.array([data.height, data.width]) * 10 / max(data.height, data.width)
        n = len(data.pred_reactions)
        fig, axes = plt.subplots(n, 1, figsize=(w, h * n))
        if n == 1:
            axes = [axes]
        fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.99))
        canvas = FigureCanvasAgg(fig)
        for i, r in enumerate(data.pred_reactions):
            ax = axes[i]
            ax.imshow(data.image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'reaction # {i}', fontdict={'fontweight': 'bold', 'fontsize': 14})
            r.draw(ax)
        canvas.draw()
        buf = canvas.buffer_rgba()
        result_image = np.asarray(buf)
        plt.close(fig)
        return result_image

    def is_molecular_formula(self, text: str) -> bool:
        """
        Simple check to decide whether a given text string appears to be a molecular formula.
        For this example, we require that the text has no spaces and contains at least one uppercase letter.
        """
        if " " in text:
            return False
        if re.search(r'[A-Z]', text):
            return True
        return False

    
    def process_text_based_molecules(self, predictions, image, starting_index: int):
        """
        Process a list of predictions (reactions) and extract text-based molecules.
        
        This method handles predictions as a list, where each element in the list is a reaction
        prediction (a dictionary).
        
        Args:
            predictions (list): List of reaction prediction dictionaries.
            image (PIL.Image): The image object.
            starting_index (int): The starting number for numbering atoms in formulas.

        Returns:
            text_molecules (list): A list of processed text-based molecules.
            new_index (int): The next available numbering index after processing all molecules.
        """
        text_molecules = []
        width, height = image.size
        for prediction_dict in predictions:
            # Process each reaction prediction dictionary
            all_conditions = []
            if 'conditions' in prediction_dict:
                all_conditions.extend(prediction_dict['conditions'])
            # Similarly, you can check for 'reactants' and 'products' if needed
            # Add reactants and products handling if applicable
            # if 'reactants' in prediction_dict:
            #    all_conditions.extend(prediction_dict['reactants'])
            # if 'products' in prediction_dict:
            #    all_conditions.extend(prediction_dict['products'])

            # Process the conditions (and optionally other sections)
            for cond in all_conditions:
                if cond['category'] == '[Txt]':  # Only process text-based objects
                    formula_str = " ".join(cond['text']).strip()  # Combine text parts into a single string
                    if self.is_molecular_formula(formula_str):
                        elements = re.findall(r'[A-Z][a-z]?', formula_str)

                        # The bounding box from the model is (x_min, y_min, x_max, y_max)
                        (x_min, y_min, x_max, y_max) = cond['bbox']
                        corners = [
                            [x_min * width, y_min * height],
                            [x_max * width, y_min * height],
                            [x_max * width, y_max * height],
                            [x_min * width, y_max * height]
                        ]
                        text_molecules.append({
                            'formula': formula_str,
                            'elements': elements,
                            'bbox': corners  # Store the corners of the bounding box
                        })

            # Assign sequential numbering starting from starting_index
            for mol in text_molecules:
                num_atoms = len(mol['elements'])
                #print(num_atoms)
                mol['numbering'] = list(range(starting_index, starting_index + num_atoms))
                #starting_index += num_atoms

        return text_molecules, starting_index

    def draw_text_molecules(self, image, text_molecules):
        """
        Draws number labels for the text-based molecules on the image.
        
        Args:
            image (PIL.Image): The original image to draw on.
            text_molecules (list): List of text-based molecules with their properties (formula, elements, etc.)
        
        Returns:
            result_image (np.array): The image with drawn labels for each molecule.
        """
        # Use a copy of the original image to ensure the original is not modified
        image_copy = image.copy()
        fig, ax = plt.subplots(figsize=(image_copy.width / 100, image_copy.height / 100))  # Maintain original size

        # Convert PIL Image to array (for display with matplotlib)
        image_arr = np.array(image_copy)

        # Display the image
        ax.imshow(image_arr)

        for mol in text_molecules:
            bbox = np.array(mol['bbox'])
            x_min, y_min = np.min(bbox, axis=0)
            x_max, y_max = np.max(bbox, axis=0)

            # Create as many x positions as there are elements
            x_coords = np.linspace(x_min, x_max, num=len(mol['elements']))
            y_coord = (y_min + y_max) / 2  # vertical center

            for element, num, x in zip(mol['elements'], mol['numbering'], x_coords):
                #print(num)
                ax.annotate(
                    str(num),
                    xy=(x, y_coord),
                    ha='center',
                    va='center',
                    color='red',
                    fontsize=12,
                    fontweight='bold'
                )

        ax.axis('off')  # Turn off axis labels
        fig.tight_layout()

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        result_image = np.asarray(buf)
        plt.close(fig)

        return result_image