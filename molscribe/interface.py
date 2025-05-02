import argparse
from typing import List

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import random
import math
import json
from PIL import Image

from .dataset import get_transforms
from .model import Encoder, Decoder
from .chemistry import convert_graph_to_smiles
from .tokenizer import get_tokenizer


BOND_TYPES = ["", "single", "double", "triple", "aromatic", "solid wedge", "dashed wedge"]


def safe_load(module, module_states):
    def remove_prefix(state_dict):
        return {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = module.load_state_dict(remove_prefix(module_states), strict=False)
    return


class MolScribe:

    def __init__(self, model_path, device=None):
        """
        MolScribe Interface
        :param model_path: path of the model checkpoint.
        :param device: torch device, defaults to be CPU.
        """
        model_states = torch.load(model_path, map_location=torch.device('cpu'))
        args = self._get_args(model_states['args'])
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.tokenizer = get_tokenizer(args)
        self.encoder, self.decoder = self._get_model(args, self.tokenizer, self.device, model_states)
        self.transform = get_transforms(args.input_size, augment=False)

    def _get_args(self, args_states=None):
        parser = argparse.ArgumentParser()
        # Model
        parser.add_argument('--encoder', type=str, default='swin_base')
        parser.add_argument('--decoder', type=str, default='transformer')
        parser.add_argument('--trunc_encoder', action='store_true')  # use the hidden states before downsample
        parser.add_argument('--no_pretrained', action='store_true')
        parser.add_argument('--use_checkpoint', action='store_true', default=True)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--embed_dim', type=int, default=256)
        parser.add_argument('--enc_pos_emb', action='store_true')
        group = parser.add_argument_group("transformer_options")
        group.add_argument("--dec_num_layers", help="No. of layers in transformer decoder", type=int, default=6)
        group.add_argument("--dec_hidden_size", help="Decoder hidden size", type=int, default=256)
        group.add_argument("--dec_attn_heads", help="Decoder no. of attention heads", type=int, default=8)
        group.add_argument("--dec_num_queries", type=int, default=128)
        group.add_argument("--hidden_dropout", help="Hidden dropout", type=float, default=0.1)
        group.add_argument("--attn_dropout", help="Attention dropout", type=float, default=0.1)
        group.add_argument("--max_relative_positions", help="Max relative positions", type=int, default=0)
        parser.add_argument('--continuous_coords', action='store_true')
        parser.add_argument('--compute_confidence', action='store_true')
        # Data
        parser.add_argument('--input_size', type=int, default=384)
        parser.add_argument('--vocab_file', type=str, default=None)
        parser.add_argument('--coord_bins', type=int, default=64)
        parser.add_argument('--sep_xy', action='store_true', default=True)

        args = parser.parse_args([])
        if args_states:
            for key, value in args_states.items():
                args.__dict__[key] = value
        return args

    def _get_model(self, args, tokenizer, device, states):
        encoder = Encoder(args, pretrained=False)
        args.encoder_dim = encoder.n_features
        decoder = Decoder(args, tokenizer)

        safe_load(encoder, states['encoder'])
        safe_load(decoder, states['decoder'])
        # print(f"Model loaded from {load_path}")

        encoder.to(device)
        decoder.to(device)
        encoder.eval()
        decoder.eval()
        return encoder, decoder

    def predict_images(self, input_images: List, return_atoms_bonds=False, return_confidence=False, batch_size=16):
        device = self.device
        predictions = []
        self.decoder.compute_confidence = return_confidence

        for idx in range(0, len(input_images), batch_size):
            batch_images = input_images[idx:idx+batch_size]
            images = [self.transform(image=image, keypoints=[])['image'] for image in batch_images]
            images = torch.stack(images, dim=0).to(device)
            with torch.no_grad():
                features, hiddens = self.encoder(images)
                batch_predictions = self.decoder.decode(features, hiddens)
            predictions += batch_predictions

        smiles = [pred['chartok_coords']['smiles'] for pred in predictions]
        node_coords = [pred['chartok_coords']['coords'] for pred in predictions]
        node_symbols = [pred['chartok_coords']['symbols'] for pred in predictions]
        edges = [pred['edges'] for pred in predictions]

        smiles_list, molblock_list, r_success = convert_graph_to_smiles(
            node_coords, node_symbols, edges, images=input_images)

        outputs = []
        for smiles, molblock, pred in zip(smiles_list, molblock_list, predictions):
            pred_dict = {"smiles": smiles, "molfile": molblock}
            if return_confidence:
                pred_dict["confidence"] = pred["overall_score"]
            if return_atoms_bonds:
                coords = pred['chartok_coords']['coords']
                symbols = pred['chartok_coords']['symbols']
                # get atoms info
                atom_list = []
                for i, (symbol, coord) in enumerate(zip(symbols, coords)):
                    atom_dict = {"atom_symbol": symbol, "x": coord[0], "y": coord[1]}
                    if return_confidence:
                        atom_dict["confidence"] = pred['chartok_coords']['atom_scores'][i]
                    atom_list.append(atom_dict)
                pred_dict["atoms"] = atom_list
                # get bonds info
                bond_list = []
                num_atoms = len(symbols)
                for i in range(num_atoms-1):
                    for j in range(i+1, num_atoms):
                        bond_type_int = pred['edges'][i][j]
                        if bond_type_int != 0:
                            bond_type_str = BOND_TYPES[bond_type_int]
                            bond_dict = {"bond_type": bond_type_str, "endpoint_atoms": (i, j)}
                            if return_confidence:
                                bond_dict["confidence"] = pred["edge_scores"][i][j]
                            bond_list.append(bond_dict)
                pred_dict["bonds"] = bond_list
            outputs.append(pred_dict)
        return outputs

    def predict_image(self, image, return_atoms_bonds=False, return_confidence=False):
        return self.predict_images([
            image], return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)[0]

    def predict_image_files(self, image_files: List, return_atoms_bonds=False, return_confidence=False):
        input_images = []
        for path in image_files:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_images.append(image)
        return self.predict_images(
            input_images, return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)

    def predict_image_file(self, image_file: str, return_atoms_bonds=False, return_confidence=False):
        return self.predict_image_files(
            [image_file], return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)[0]

    def draw_prediction(self, prediction, image, notebook=False, label_atoms_by_number=False):
        if "atoms" not in prediction or "bonds" not in prediction:
            raise ValueError("atoms and bonds information are not provided.")
        
        h, w, _ = image.shape
        h, w = np.array([h, w]) * 400 / max(h, w)
        image = cv2.resize(image, (int(w), int(h)))
        
        def plot_atoms_and_bonds(ax, label_mode="symbol"):
            ax.axis('off')
            ax.set_xlim(-0.05 * w, w * 1.05)
            ax.set_ylim(1.05 * h, -0.05 * h)
            plt.imshow(image, alpha=0.)
            
            x = [a['x'] * w for a in prediction['atoms']]
            y = [a['y'] * h for a in prediction['atoms']]
            markersize = min(w, h) / 3
            plt.scatter(x, y, marker='o', s=markersize, color='lightskyblue', zorder=10)
            
            # Track positions to handle overlaps
            label_positions = set()
            for i, atom in enumerate(prediction['atoms']):
                if label_mode == "symbol":
                    label = atom['atom_symbol'].lstrip('[').rstrip(']')
                elif label_mode == "number":
                    label = str(i + 1)  # Numbering atoms starting from 1
                else:
                    raise ValueError("Invalid label_mode. Choose 'symbol' or 'number'.")
                
                # Handle potential overlaps by shifting labels slightly
                label_x, label_y = x[i], y[i]
                # while (round(label_x, 1), round(label_y, 1)) in label_positions:
                #     label_x += 5  # Offset x slightly to the right
                #     label_y += 5  # Offset y slightly downward
                
                # label_positions.add((round(label_x, 1), round(label_y, 1)))
                if (round(label_x, 1), round(label_y, 1)) in label_positions:
                    continue
                label_positions.add((round(label_x, 1), round(label_y, 1)))
                plt.annotate(label, xy=(label_x, label_y), ha='center', va='center', color='black', zorder=100)
            
            for bond in prediction['bonds']:
                u, v = bond['endpoint_atoms']
                x1, y1, x2, y2 = x[u], y[u], x[v], y[v]
                bond_type = bond['bond_type']
                if bond_type == 'single':
                    color = 'tab:green'
                    ax.plot([x1, x2], [y1, y2], color, linewidth=4)
                elif bond_type == 'aromatic':
                    color = 'tab:purple'
                    ax.plot([x1, x2], [y1, y2], color, linewidth=4)
                elif bond_type == 'double':
                    color = 'tab:green'
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=7)
                    ax.plot([x1, x2], [y1, y2], color='w', linewidth=1.5, zorder=2.1)
                elif bond_type == 'triple':
                    color = 'tab:green'
                    x1s, x2s = 0.8 * x1 + 0.2 * x2, 0.2 * x1 + 0.8 * x2
                    y1s, y2s = 0.8 * y1 + 0.2 * y2, 0.2 * y1 + 0.8 * y2
                    ax.plot([x1s, x2s], [y1s, y2s], color=color, linewidth=9)
                    ax.plot([x1, x2], [y1, y2], color='w', linewidth=5, zorder=2.05)
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, zorder=2.1)
                else:
                    length = 10
                    width = 10
                    color = 'tab:green'
                    if bond_type == 'solid wedge':
                        ax.annotate('', xy=(x1, y1), xytext=(x2, y2),
                                    arrowprops=dict(color=color, width=3, headwidth=width, headlength=length), zorder=2)
                    else:
                        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                    arrowprops=dict(color=color, width=3, headwidth=width, headlength=length), zorder=2)

        # Plot with atom symbols
        fig1, ax1 = plt.subplots(1, 1)
        plot_atoms_and_bonds(ax1, label_mode="symbol")
        fig1.tight_layout()
        if not notebook:
            canvas1 = FigureCanvasAgg(fig1)
            canvas1.draw()
            buf1 = canvas1.buffer_rgba()
            result_image_symbol = np.asarray(buf1)
            plt.close(fig1)
            return result_image_symbol

    def draw_curved_arrow(self, ax, start, end, direction=1, n_points=100):
        start = np.array(start)
        end = np.array(end)
        midpoint = (start + end) / 2
        vec = end - start
        d = np.linalg.norm(vec)
        radius = d / 2

        start_angle = np.arctan2(start[1] - midpoint[1], start[0] - midpoint[0])
        angles = np.linspace(start_angle, start_angle + np.pi, n_points)
        arc_x = midpoint[0] + radius * np.cos(angles)
        arc_y = midpoint[1] + radius * np.sin(angles)
        
        ax.plot(arc_x, arc_y, color='black', linewidth=2)
        ax.annotate(
            '', 
            xy=(arc_x[-1], arc_y[-1]), 
            xytext=(arc_x[-2], arc_y[-2]),
            arrowprops=dict(arrowstyle="->", color='black', lw=2)
        )
        
        arrow_info = {
            "start": start.tolist(),
            "end": end.tolist(),
            "midpoint": midpoint.tolist(),
            "radius": float(radius),
            "start_angle": float(start_angle),
            "direction": direction
        }
        return arrow_info
    def get_number_image_path(self, original_path):
        """
        Given an original image path like 'a/b/image.png',
        returns the number image path 'a/number/b/image.png'
        by inserting 'number' after the first folder.
        """
        parts = original_path.split(os.sep)
        new_parts = [parts[0], "number"] + parts[1:]
        number_path = os.sep.join(new_parts)
        return number_path

    def get_synthetic_image_path(self, original_path):
        """
        Given an original image path like 'a/b/image.png',
        returns the synthetic image path 'a/synthetic/b/image.png'
        by inserting 'synthetic' after the first folder.
        """
        parts = original_path.split(os.sep)
        new_parts = [parts[0], "synthetic"] + parts[1:]
        synthetic_path = os.sep.join(new_parts)
        return synthetic_path

    def syn_draw_prediction_number(self, prediction, image_path, label_atoms_by_number=True):
        """
        Draws atoms, bonds, and random curved arrows (electron pairs) onto an image,
        then saves two synthetic images to modified paths. One image (the "number image") 
        is drawn on a white background and saved in a folder with "number" inserted into the
        original path. The second image (the "arrow image") has curved arrows (drawn on the 
        original image) and is saved to a path with "synthetic" inserted. In addition, 
        arrow information for images in the same subfolder is aggregated into a single JSON file.
        
        For example, for an image located at "a/b/image.png":
        - The number image is saved at: "a/number/b/image.png"
        - The arrow image is saved at: "a/synthetic/b/image.png"
        - The aggregated arrow JSON for folder b is saved at: "a/b_json/arrow_info.json"
        
        The aggregated arrow JSON will have a structure similar to:
        
        {
        "image 1": {
            "Step_1": { "Start": 10, "End": 9 },
            "Step_2": { "Start": 17, "End": 16 },
            "Step_3": { "Start": 27, "End": 26 }
        },
        "image 2": { ... },
        "image 3": { ... }
        }
        
        Parameters:
            prediction: dict with 'atoms' and 'bonds'
            image_path: str, path to the original image (e.g., "a/b/image.png")
            label_atoms_by_number: bool, whether to label atoms by number

        Returns:
            number_image: numpy array of the white background number image
            arrow_image: numpy array of the arrow image (curved arrows on original image)
            label: total number of atoms
            atoms_json: dict of atom coordinates
            arrow_pairs_json: dict of arrow pair details for the current image
        """
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        x = [a['x'] * w for a in prediction['atoms']]
        y = [a['y'] * h for a in prediction['atoms']]
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        if "atoms" not in prediction or "bonds" not in prediction:
            raise ValueError("atoms and bonds information are not provided.")
        
        # Compute atom coordinates.
        atoms_json = {}
        for i, (xi, yi) in enumerate(zip(x, y)):
            atoms_json[i + 1] = {"xcord": xi, "ycord": yi}
        
        dpi = 100
        figsize = (w / dpi, h / dpi)
        
        # ----------------------------------------------------------------
        # 1. Create the number image (white background with atoms, bonds, numbers)
        # ----------------------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
        fig1.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        def plot_atoms_and_bonds(ax, label_mode="symbol"):
            ax.axis('off')
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)  # Top-left origin.
            # White background.
            white_bg = 255 * np.ones((h, w, 3), dtype=np.uint8)
            ax.imshow(white_bg, interpolation='nearest')
            
            markersize = min(w, h) / 3
            ax.scatter(x, y, marker='o', s=markersize, color='lightskyblue', zorder=10)
            
            label_positions = set()
            for i, atom in enumerate(prediction['atoms']):
                if label_mode == "symbol":
                    label = atom['atom_symbol'].lstrip('[').rstrip(']')
                elif label_mode == "number":
                    label = str(i + 1)
                else:
                    raise ValueError("Invalid label_mode. Choose 'symbol' or 'number'.")
                label_x, label_y = x[i], y[i]
                if (round(label_x, 1), round(label_y, 1)) in label_positions:
                    continue
                label_positions.add((round(label_x, 1), round(label_y, 1)))
                ax.annotate(label, xy=(label_x, label_y), ha='center', va='center', color='black', zorder=100)
            
            # Draw bonds.
            for bond in prediction['bonds']:
                u, v = bond['endpoint_atoms']
                x1, y1, x2, y2 = x[u], y[u], x[v], y[v]
                bond_type = bond['bond_type']
                if bond_type == 'single':
                    ax.plot([x1, x2], [y1, y2], color='tab:green', linewidth=4)
                elif bond_type == 'aromatic':
                    ax.plot([x1, x2], [y1, y2], color='tab:purple', linewidth=4)
                elif bond_type == 'double':
                    ax.plot([x1, x2], [y1, y2], color='tab:green', linewidth=7)
                    ax.plot([x1, x2], [y1, y2], color='w', linewidth=1.5, zorder=2.1)
                elif bond_type == 'triple':
                    x1s, x2s = 0.8 * x1 + 0.2 * x2, 0.2 * x1 + 0.8 * x2
                    y1s, y2s = 0.8 * y1 + 0.2 * y2, 0.2 * y1 + 0.8 * y2
                    ax.plot([x1s, x2s], [y1s, y2s], color='tab:green', linewidth=9)
                    ax.plot([x1, x2], [y1, y2], color='w', linewidth=5, zorder=2.05)
                    ax.plot([x1, x2], [y1, y2], color='tab:green', linewidth=2, zorder=2.1)
                else:
                    length = 10
                    width = 10
                    if bond_type == 'solid wedge':
                        ax.annotate('', xy=(x1, y1), xytext=(x2, y2),
                                    arrowprops=dict(color='tab:green', width=3, headwidth=width, headlength=length),
                                    zorder=2)
                    else:
                        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                    arrowprops=dict(color='tab:green', width=3, headwidth=width, headlength=length),
                                    zorder=2)
        
        plot_atoms_and_bonds(ax1, label_mode="number")
        fig1.tight_layout(pad=0)
        canvas1 = FigureCanvasAgg(fig1)
        canvas1.draw()
        buf1 = canvas1.buffer_rgba()
        number_image = np.asarray(buf1)
        plt.close(fig1)
        
        # Save number image to folder "a/number/b/image.png"
        number_image_path = self.get_number_image_path(image_path)
        os.makedirs(os.path.dirname(number_image_path), exist_ok=True)
        plt.imsave(number_image_path, number_image)
        print("Saved number image to:", number_image_path)
        
        # ----------------------------------------------------------------
        # 2. Create the arrow image (curved arrows drawn on the original image)
        # ----------------------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=figsize, dpi=dpi)
        fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax2.imshow(image, interpolation='nearest')
        ax2.axis('off')
        
        arrow_pairs = []
        # Select arrow pairs based on the rule: a random base atom paired with one within ±2.
        if len(atoms_json) >= 2:
            num_pairs = random.randint(1, 5)
            atom_keys = sorted(list(atoms_json.keys()))
            for _ in range(num_pairs):
                base = random.choice(atom_keys)
                candidates = [k for k in atom_keys if k != base and abs(k - base) <= 2]
                if not candidates:
                    continue
                partner = random.choice(candidates)
                direction = random.choice([1, -1])
                start_coord = (atoms_json[base]['xcord'], atoms_json[base]['ycord'])
                end_coord   = (atoms_json[partner]['xcord'], atoms_json[partner]['ycord'])
                # Draw curved arrow using the class method.
                self.draw_curved_arrow(ax2, start_coord, end_coord, direction=direction)
                arrow_pairs.append((base, partner))
        
        # Sort the arrow pairs by the base atom and reassign step numbers.
        arrow_pairs = sorted(arrow_pairs, key=lambda pair: pair[0])
        arrow_pairs_json = {}
        for step, (base, partner) in enumerate(arrow_pairs, start=1):
            arrow_pairs_json[f"Step_{step}"] = {"Start": base, "End": partner}
        
        fig2.tight_layout(pad=0)
        canvas2 = FigureCanvasAgg(fig2)
        canvas2.draw()
        buf2 = canvas2.buffer_rgba()
        arrow_image = np.asarray(buf2)
        plt.close(fig2)
        
        synthetic_image_path = self.get_synthetic_image_path(image_path)
        os.makedirs(os.path.dirname(synthetic_image_path), exist_ok=True)
        plt.imsave(synthetic_image_path, arrow_image)
        print("Saved arrow image to:", synthetic_image_path)
        
        # ----------------------------------------------------------------
        # 3. Aggregate arrow JSON per subfolder into one big JSON file.
        # For an image at "a/b/image.png", save aggregated JSON at "a/b_json/arrow_info.json".
        # ----------------------------------------------------------------
        # Determine the subfolder containing the image.
        subfolder = os.path.dirname(image_path)  # e.g., "a/b"
        # Create an aggregated JSON folder in the parent directory of the subfolder.
        aggregated_dir = os.path.join(os.path.dirname(subfolder), f"{os.path.basename(subfolder)}_json")
        os.makedirs(aggregated_dir, exist_ok=True)
        aggregated_json_path = os.path.join(aggregated_dir, "arrow_info.json")
        
        # Load existing aggregated data if available.
        if os.path.exists(aggregated_json_path):
            with open(aggregated_json_path, "r") as f:
                agg_data = json.load(f)
        else:
            agg_data = {}
        
        # Update the aggregated data with the current image's arrow pairs.
        agg_data[image_name] = arrow_pairs_json
        with open(aggregated_json_path, "w") as f:
            json.dump(agg_data, f, indent=4)
        
        print("Aggregated arrow JSON updated at:", aggregated_json_path)
        
        label = len(prediction['atoms'])
        return number_image, arrow_image, label, atoms_json, arrow_pairs_json


    def draw_prediction_number(self, prediction, image, label_atoms_by_number=True):
        if "atoms" not in prediction or "bonds" not in prediction:
            raise ValueError("atoms and bonds information are not provided.")
        
        # Use original image dimensions.
        h, w, _ = image.shape
        
        # Create a white background image of the same dimensions.
        white_background = 255 * np.ones((h, w, 3), dtype=np.uint8)
        
        # Decide whether to label atoms by number or by symbol.
        label_mode = "number" if label_atoms_by_number else "symbol"
        
        def plot_atoms_and_bonds(ax, label_mode="symbol"):
            ax.axis('off')
            # Set the axes limits to exactly match the image dimensions.
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)  # Invert y-axis so the origin is at the top-left.
            # Display the white background.
            ax.imshow(white_background, interpolation='nearest')
            
            # Compute atom positions based on normalized coordinates.
            x = [a['x'] * w for a in prediction['atoms']]
            y = [a['y'] * h for a in prediction['atoms']]
            markersize = min(w, h) / 3
            ax.scatter(x, y, marker='o', s=markersize, color='lightskyblue', zorder=10)
            
            label_positions = set()
            for i, atom in enumerate(prediction['atoms']):
                if label_mode == "symbol":
                    label = atom['atom_symbol'].lstrip('[').rstrip(']')
                elif label_mode == "number":
                    label = str(i + 1)
                else:
                    raise ValueError("Invalid label_mode. Choose 'symbol' or 'number'.")
                
                label_x, label_y = x[i], y[i]
                if (round(label_x, 1), round(label_y, 1)) in label_positions:
                    continue
                label_positions.add((round(label_x, 1), round(label_y, 1)))
                ax.annotate(label, xy=(label_x, label_y), ha='center', va='center', color='black', zorder=100)
            
            # Draw bonds.
            for bond in prediction['bonds']:
                u, v = bond['endpoint_atoms']
                x1, y1, x2, y2 = x[u], y[u], x[v], y[v]
                bond_type = bond['bond_type']
                if bond_type == 'single':
                    color = 'tab:green'
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=4)
                elif bond_type == 'aromatic':
                    color = 'tab:purple'
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=4)
                elif bond_type == 'double':
                    color = 'tab:green'
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=7)
                    ax.plot([x1, x2], [y1, y2], color='w', linewidth=1.5, zorder=2.1)
                elif bond_type == 'triple':
                    color = 'tab:green'
                    x1s, x2s = 0.8 * x1 + 0.2 * x2, 0.2 * x1 + 0.8 * x2
                    y1s, y2s = 0.8 * y1 + 0.2 * y2, 0.2 * y1 + 0.8 * y2
                    ax.plot([x1s, x2s], [y1s, y2s], color=color, linewidth=9)
                    ax.plot([x1, x2], [y1, y2], color='w', linewidth=5, zorder=2.05)
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, zorder=2.1)
                else:
                    length = 10
                    width = 10
                    color = 'tab:green'
                    if bond_type == 'solid wedge':
                        ax.annotate('', xy=(x1, y1), xytext=(x2, y2),
                                    arrowprops=dict(color=color, width=3, headwidth=width, headlength=length), zorder=2)
                    else:
                        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                    arrowprops=dict(color=color, width=3, headwidth=width, headlength=length), zorder=2)
        
        # Create a figure with dimensions matching the original image.
        dpi = 100
        figsize = (w / dpi, h / dpi)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        # Remove margins so that the canvas exactly matches the image dimensions.
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        plot_atoms_and_bonds(ax, label_mode=label_mode)
        fig.tight_layout(pad=0)
        
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        output_image = np.asarray(buf)
        
        plt.close(fig)
        label = len(prediction['atoms'])
        return output_image, label

    def _draw_curved_arrow(self, ax, start, end):
        """
        Draws a curved arrow between start and end.
        The arrow is composed of a half-circle arc (using points along a semicircle)
        and an arrow head at the end.
        :param ax: matplotlib axis on which to draw.
        :param start: tuple (x, y) for the start point.
        :param end: tuple (x, y) for the end point.
        """
        x1, y1 = start
        x2, y2 = end
        # Compute center as midpoint (using the chord as the diameter)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        distance = math.hypot(dx, dy)
        radius = distance / 2
        # Angle for start point relative to center
        theta_start = math.atan2(y1 - cy, x1 - cx)
        num_points = 100
        # Generate angles spanning half a circle (pi radians)
        angles = np.linspace(theta_start, theta_start + math.pi, num_points)
        arc_x = cx + radius * np.cos(angles)
        arc_y = cy + radius * np.sin(angles)
        # Plot the arc
        ax.plot(arc_x, arc_y, linestyle='-', color='black', linewidth=2)
        # Draw arrow head using the last two points of the arc
        head_start = (arc_x[-2], arc_y[-2])
        head_end = (arc_x[-1], arc_y[-1])
        ax.annotate('', xy=head_end, xytext=head_start,
                    arrowprops=dict(arrowstyle="->", color='black', lw=2))
