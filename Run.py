import numpy as np
from PIL import Image
import os
import sys

# Assuming 'leffa' and 'leffa_utils' are custom modules.
# Ensure they are in Python's path or installed.
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

class LeffaPredictor(object):
    def __init__(self, ckpt_dir="./ckpts"):
        self.ckpt_dir = os.path.abspath(ckpt_dir) # Store absolute path
        print(f"Initializing LeffaPredictor with checkpoint directory: {self.ckpt_dir}")

        if not os.path.exists(self.ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.ckpt_dir}")

        self.mask_predictor = AutoMasker(
            densepose_path=os.path.join(self.ckpt_dir, "densepose"),
            schp_path=os.path.join(self.ckpt_dir, "schp"),
        )

        # Corrected: config_path should use ckpt_dir
        densepose_config_path = os.path.join(self.ckpt_dir, "densepose", "densepose_rcnn_R_50_FPN_s1x.yaml")
        if not os.path.exists(densepose_config_path):
            # Fallback if the user still has a hardcoded path idea or different structure.
            # This is less ideal but provides a transition.
            # The D:\ path from your script was an example of what to avoid generally.
            # For this correction, we rely on ckpt_dir being structured correctly.
            print(f"Warning: Default densepose_rcnn_R_50_FPN_s1x.yaml not found in {os.path.join(self.ckpt_dir, 'densepose')}.")
            # If you absolutely need to support an old hardcoded path as a last resort:
            # old_hardcoded_path = r"D:\Cloth Segmentation\Leffa\ckpts\densepose\densepose_rcnn_R_50_FPN_s1x.yaml"
            # if os.path.exists(old_hardcoded_path):
            #     print(f"Using fallback hardcoded path for densepose config: {old_hardcoded_path}")
            #     densepose_config_path = old_hardcoded_path
            # else:
            raise FileNotFoundError(f"DensePose config file not found at {densepose_config_path} or other expected locations.")

        self.densepose_predictor = DensePosePredictor(
            config_path=densepose_config_path,
            weights_path=os.path.join(self.ckpt_dir, "densepose", "model_final_162be9.pkl"),
        )

        self.parsing = Parsing(
            atr_path=os.path.join(self.ckpt_dir, "humanparsing", "parsing_atr.onnx"),
            lip_path=os.path.join(self.ckpt_dir, "humanparsing", "parsing_lip.onnx"),
        )

        self.openpose = OpenPose(
            body_model_path=os.path.join(self.ckpt_dir, "openpose", "body_pose_model.pth"),
        )

        sd_inpainting_path = os.path.join(self.ckpt_dir, "stable-diffusion-inpainting")
        if not os.path.isdir(sd_inpainting_path):
             raise FileNotFoundError(f"Stable Diffusion inpainting directory not found: {sd_inpainting_path}")

        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path=sd_inpainting_path,
            pretrained_model=os.path.join(self.ckpt_dir, "virtual_tryon.pth"),
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)

        print("LeffaPredictor models initialized.")

    def _leffa_predict_internal(
        self,
        src_image_pil,
        ref_image_pil,
        ref_acceleration=False,
        step=50,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body", # This will be passed from leffa_predict_vt
        vt_repaint=False,
        preprocess_garment_flag=False
    ):
        # ... (rest of the code before this section is unchanged)

        src_image = resize_and_center(src_image_pil, 768, 1024)
        temp_garment_file_path = None

        if preprocess_garment_flag:
            if ref_image_pil.format == 'PNG':
                temp_garment_file_path = "temp_garment_for_processing.png"
                ref_image_pil.save(temp_garment_file_path)
                ref_image = preprocess_garment_image(temp_garment_file_path)
                print(f"Preprocessed garment image from {temp_garment_file_path}")
            else:
                print("Warning: Garment preprocessing is enabled but the input image for garment is not a PNG. Proceeding without preprocessing the garment.")
                ref_image = resize_and_center(ref_image_pil, 768, 1024)
        else:
            ref_image = resize_and_center(ref_image_pil, 768, 1024)

        # *** ADD THIS BLOCK ***
        # Ensure ref_image is always RGB (3 channels) before proceeding to model
        if ref_image.mode != 'RGB':
            ref_image = ref_image.convert('RGB')
            print("Converted ref_image to RGB (3 channels).")
        # ********************

        src_image_array = np.array(src_image)
        src_image_rgb = src_image.convert("RGB") # This line is already good for src_image
        model_parse, _ = self.parsing(src_image_rgb.resize((384, 512)))
        keypoints = self.openpose(src_image_rgb.resize((384, 512)))


        if vt_model_type == "viton_hd":
            mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
        elif vt_model_type == "dress_code":
            mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)
        else:
            raise ValueError(f"Unknown vt_model_type: {vt_model_type}")
        mask = mask.resize((768, 1024))

        if vt_model_type == "viton_hd":
            src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
            densepose = Image.fromarray(src_image_seg_array)
        elif vt_model_type == "dress_code":
            src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
            src_image_seg_array = src_image_iuv_array[:, :, 0:1]
            src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
            densepose = Image.fromarray(src_image_seg_array)
        
        transform = LeffaTransform()
        data = {
            "src_image": [src_image], "ref_image": [ref_image],
            "mask": [mask], "densepose": [densepose],
        }
        data = transform(data)

        inference_engine = self.vt_inference_hd if vt_model_type == "viton_hd" else self.vt_inference_dc
        
        output = inference_engine(
            data, ref_acceleration=ref_acceleration, num_inference_steps=step,
            guidance_scale=scale, seed=seed, repaint=vt_repaint,
        )
        gen_image_pil = output["generated_image"][0]

        if temp_garment_file_path and os.path.exists(temp_garment_file_path):
            try:
                os.remove(temp_garment_file_path)
                print(f"Removed temporary file: {temp_garment_file_path}")
            except Exception as e:
                print(f"Error removing temporary file {temp_garment_file_path}: {e}")
                
        return np.array(gen_image_pil), np.array(mask), np.array(densepose) # Return arrays as before

    def leffa_predict_vt(
        self,
        src_image_pil,
        ref_image_pil,
        vt_garment_type, # Made this a required argument to be passed from config
        ref_acceleration=False,
        step=50,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_repaint=False,
        preprocess_garment_flag=False
    ):
        if not isinstance(src_image_pil, Image.Image):
            raise TypeError("src_image_pil must be a PIL.Image.Image object.")
        if not isinstance(ref_image_pil, Image.Image):
            raise TypeError("ref_image_pil must be a PIL.Image.Image object.")

        return self._leffa_predict_internal(
            src_image_pil=src_image_pil,
            ref_image_pil=ref_image_pil,
            ref_acceleration=ref_acceleration,
            step=step,
            scale=scale,
            seed=seed,
            vt_model_type=vt_model_type,
            vt_garment_type=vt_garment_type, # Pass the configured garment type
            vt_repaint=vt_repaint,
            preprocess_garment_flag=preprocess_garment_flag,
        )

if __name__ == "__main__":
    # --- Configuration for Local Windows Run ---
    # Ensure this path is correct and points to your main checkpoints folder.
    # It should contain subfolders like 'densepose', 'humanparsing', etc.
    CHECKPOINT_DIR_LOCAL = "./ckpts" 

    # Paths to your input images
    # Replace these with the actual paths to your images.
    # Using raw strings (r"") for Windows paths is a good practice if they contain backslashes.
    PERSON_IMAGE_PATH = r"Model.png" 
    GARMENT_IMAGE_PATH = r"Cloth.png"  

    # Manually set the garment type here
    # Options: "upper_body", "lower_body", "dresses"
    VT_GARMENT_TYPE_INPUT = "upper_body"

    # Output file name
    OUTPUT_IMAGE_NAME = "output.png"

    # --- Virtual Try-on Parameters ---
    MODEL_TYPE = "viton_hd"  # "viton_hd" or "dress_code"
    PREPROCESS_GARMENT = True   # True if garment is PNG and needs preprocessing
    STEPS = 30
    GUIDANCE_SCALE = 2.5
    SEED_VALUE = 42
    # ------------------------------------

    print(f"Initializing LeffaPredictor with local checkpoint directory: {os.path.abspath(CHECKPOINT_DIR_LOCAL)}")
    try:
        predictor = LeffaPredictor(ckpt_dir=CHECKPOINT_DIR_LOCAL)
        print("LeffaPredictor initialized successfully.")

        # Verify image paths
        if not os.path.exists(PERSON_IMAGE_PATH):
            raise FileNotFoundError(f"Person image not found: {PERSON_IMAGE_PATH}")
        if not os.path.exists(GARMENT_IMAGE_PATH):
            raise FileNotFoundError(f"Garment image not found: {GARMENT_IMAGE_PATH}")

        print(f"Loading person image from: {PERSON_IMAGE_PATH}")
        person_pil_image = Image.open(PERSON_IMAGE_PATH)
        print(f"Loading garment image from: {GARMENT_IMAGE_PATH}")
        garment_pil_image = Image.open(GARMENT_IMAGE_PATH)

        print(f"\nPerforming virtual try-on with the following settings:")
        print(f"  Garment Type: {VT_GARMENT_TYPE_INPUT}")
        print(f"  Model Type: {MODEL_TYPE}")
        print(f"  Steps: {STEPS}, Scale: {GUIDANCE_SCALE}, Seed: {SEED_VALUE}")
        print(f"  Preprocess Garment: {PREPROCESS_GARMENT}")

        generated_image_array, mask_array, densepose_array = predictor.leffa_predict_vt(
            src_image_pil=person_pil_image,
            ref_image_pil=garment_pil_image,
            vt_garment_type=VT_GARMENT_TYPE_INPUT, # Pass the configured garment type
            vt_model_type=MODEL_TYPE,
            preprocess_garment_flag=PREPROCESS_GARMENT,
            step=STEPS,
            scale=GUIDANCE_SCALE,
            seed=SEED_VALUE
        )
        print("Prediction complete.")

        result_image = Image.fromarray(generated_image_array)
        result_image.save(OUTPUT_IMAGE_NAME)
        print(f"Generated image saved as '{OUTPUT_IMAGE_NAME}'")

        # Optionally save mask and densepose PIL Images
        # mask_pil = Image.fromarray(mask_array)
        # mask_pil.save("generated_mask.png")
        # densepose_pil = Image.fromarray(densepose_array) # Assuming densepose from _leffa_predict_internal is PIL-convertible
        # densepose_pil.save("generated_densepose.png")

    except FileNotFoundError as e:
        print(f"🚨 Error: A required file or directory was not found: {e}")
        print("👉 Please ensure that CHECKPOINT_DIR_LOCAL is correct and contains all necessary model files (densepose, humanparsing, etc.).")
        print("👉 Also, verify that PERSON_IMAGE_PATH and GARMENT_IMAGE_PATH point to valid images in the same directory as the script or provide full paths.")
    except ImportError as e:
        print(f"🚨 Error: An import failed. This often means a required Python package is missing or not in PYTHONPATH: {e}")
        print("👉 Make sure you have installed all packages (numpy, Pillow, torch, onnxruntime) in your active Python environment.")
        print("👉 Ensure 'leffa', 'leffa_utils', 'preprocess' folders are in the same directory as this script or correctly installed.")
    except RuntimeError as e:
        print(f"🚨 RuntimeError: {e}")
        print("👉 This could be related to CUDA setup, GPU memory, or model incompatibility. If using GPU, check your PyTorch and CUDA versions.")
    except ValueError as e: # Catch specific ValueErrors like invalid garment type
        print(f"🚨 Configuration or Value Error: {e}")
    except Exception as e:
        print(f"🚨 An unexpected error occurred: {type(e).__name__} - {e}")
