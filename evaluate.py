#!/usr/bin/env python3
"""
GGBench unified evaluation script.

Configure paths, API, and module switches at the top of this file.
No command-line arguments are required when running.
The script appends various evaluation results to the model output JSON and writes new JSON/JSONL files.
"""

from __future__ import annotations

import base64
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from collections import OrderedDict

import numpy as np
import torch
from openai import OpenAI
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import lpips

from eval_prompts import (
    IMAGE_CONSISTENCY_PROMPT,
    MID_PROCESS_PROMPT,
    TEXT_STEP_PROMPT_TEMPLATE,
)


# === User Configuration (Modify as needed) ================================================================
# Modify the following constants to configure paths, judge model, and evaluation modules.
# Paths are resolved relative to the script's directory, not the current working directory
_SCRIPT_DIR = Path(__file__).parent.resolve()

DATASET_PATH = _SCRIPT_DIR / "GGBench_dataset.json"   # Dataset path
MODEL_OUTPUT_PATH = _SCRIPT_DIR / "YOUR_MODEL_OUTPUT_PATH.json"   # Model output file path to evaluate
DATASET_ROOT = _SCRIPT_DIR   # Dataset root path
PRED_ROOT = _SCRIPT_DIR   # Prediction results root path
OUTPUT_JSON = _SCRIPT_DIR / "eval_output" / "result.json"   # Output JSON file path
OUTPUT_JSONL = _SCRIPT_DIR / "eval_output" / "result.jsonl"   # Output JSONL file path

DATASET_TOTAL_COUNT = 1411   # Total number of samples in the dataset (used for score normalization)

JUDGE_MODEL = "gpt-4o"   # Judge model
JUDGE_URL = "YOUR_JUDGE_URL"   # Judge model URL
JUDGE_API_KEY = "YOUR_JUDGE_API_KEY"  # Judge model API key
MAX_WORKERS = 4   # Number of concurrent workers

ENABLE_IMAGE_JUDGE = True   # Enable final image judge
ENABLE_TEXT_JUDGE = True   # Enable text judge
ENABLE_MID_PROCESS_JUDGE = True   # Enable mid-process (long image) judge
ENABLE_LPIPS = True   # Enable LPIPS evaluation
ENABLE_PSNR = True   # Enable PSNR evaluation
ENABLE_SSIM = True   # Enable SSIM evaluation

LOG_FILE: Optional[Path] = _SCRIPT_DIR / "eval_output" / "evaluate.log"
# ======================================================================


def _log_setup(log_file: Optional[Path]) -> None:
    """Configure logging output, supporting both terminal and file output."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def _resolve_path(base: Optional[Path], value: Optional[str]) -> Optional[Path]:
    """Helper function to resolve relative paths for unified resource location management.
    
    If a path starts with / but the file doesn't exist, it will try to re-resolve based on base.
    """
    if not value:
        return None
    path = Path(value)
    
    # If path is absolute
    if path.is_absolute():
        # Check if file exists
        if path.exists():
            return path
        # If file doesn't exist and base is provided, try to re-resolve based on base
        # Remove leading / to make it a relative path
        if base and value.startswith('/'):
            relative_path = value.lstrip('/')
            resolved = (base / relative_path).resolve()
            if resolved.exists():
                return resolved
        # If neither exists, return original path (let caller handle it)
        return path
    
    # Relative path: resolve based on base
    if base:
        return (base / path).resolve()
    return path.resolve()


def _image_to_data_url(image_path: Path) -> str:
    """Convert local image to data URL for direct reading by judge model."""
    suffix = image_path.suffix.lower().lstrip(".")
    if suffix not in {"png", "jpg", "jpeg"}:
        raise ValueError(
            f"Unsupported image format for base64 conversion: {image_path.suffix}"
        )
    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/{suffix};base64,{encoded}"


def load_dataset(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load dataset and create index by id for easy completion of problem information."""
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: Dict[str, Dict[str, Any]] = {}
    for item in data:
        item_id = str(item.get("id"))
        if not item_id:
            logging.warning("Dataset item lacks 'id': %s", item)
            continue
        mapping[item_id] = item
    return mapping


def load_model_results(result_path: Path) -> List[Dict[str, Any]]:
    """Load the result list output by the model to be evaluated."""
    with result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Model result JSON must be a list of objects.")
    return data


def ensure_fields(
    items: List[Dict[str, Any]],
    dataset_map: Dict[str, Dict[str, Any]],
    dataset_root: Path,
    problem_field: str = "question_image",
    reference_image_field: str = "res_image",
) -> None:
    """Combine dataset metadata to complete fields like problem text/reference images for model output."""
    for item in items:
        item_id = str(item.get("id"))
        if item_id not in dataset_map:
            logging.warning("Skip item id=%s: not found in dataset.", item_id)
            item["_skip"] = True
            continue
        dataset_entry = dataset_map[item_id]
        item["_dataset"] = dataset_entry

        # Complete common fields
        if problem_field not in item:
            item[problem_field] = dataset_entry.get(problem_field)
        if "text_answer" not in item:
            item["text_answer"] = dataset_entry.get("text_answer")
        if "question_image" not in item:
            item["question_image"] = dataset_entry.get("question_image")
        if "question" not in item:
            item["question"] = dataset_entry.get("question")

        # Try multiple possible field names
        ref_image_rel = (dataset_entry.get(reference_image_field) or 
                        dataset_entry.get("res_image"))
        if ref_image_rel:
            item["_reference_image"] = _resolve_path(dataset_root, ref_image_rel)
            if item["_reference_image"] and not item["_reference_image"].exists():
                logging.debug("Reference image path resolved but not exists: id=%s, path=%s", 
                             item_id, item["_reference_image"])
        else:
            item["_reference_image"] = None
            logging.debug("Reference image field missing: id=%s, tried fields: %s, res_image", 
                         item_id, reference_image_field)


class JudgeClient:
    """Encapsulate judge model calls, unified management of prompts and interface format."""
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def image_consistency(self, prompt: str, ref: Path, pred: Path) -> str:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(ref)}},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(pred)}},
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
        )
        return completion.choices[0].message.content.strip()

    def text_chain(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            temperature=0.0,
        )
        return completion.choices[0].message.content.strip()

    def mid_process(self, problem_text: str, long_image: Path) -> str:
        content = [
            {"type": "text", "text": f"Problem Text:\n{problem_text}"},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(long_image)}},
            {"type": "text", "text": MID_PROCESS_PROMPT},
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
        )
        return completion.choices[0].message.content.strip()


def evaluate_image_consistency(
    items: Iterable[Dict[str, Any]],
    judge: JudgeClient,
    pred_root: Optional[Path],
    max_workers: int,
) -> None:
    """Final image judge: Given reference image and model image, evaluate geometric consistency."""
    def worker(item: Dict[str, Any]) -> None:
        if item.get("_skip"):
            return
        reference = item.get("_reference_image")
        if not reference:
            logging.warning("id=%s missing reference image (path is None).", item.get("id"))
            item["VLM_eval_image_result"] = "1"
            return
        if not reference.exists():
            logging.warning("id=%s missing reference image (file not found): %s", item.get("id"), reference)
            item["VLM_eval_image_result"] = "1"
            return
        pred_path = _resolve_path(pred_root, item.get("image_4") or item.get("output_image") or item.get("output_image_path"))
        if not pred_path:
            logging.warning("id=%s missing predicted image (path is None).", item.get("id"))
            item["VLM_eval_image_result"] = "1"
            return
        if not pred_path.exists():
            logging.warning("id=%s missing predicted image (file not found): %s", item.get("id"), pred_path)
            item["VLM_eval_image_result"] = "1"
            return
        try:
            score = judge.image_consistency(IMAGE_CONSISTENCY_PROMPT, reference, pred_path)
        except Exception as exc:
            logging.error("Image consistency failed id=%s: %s", item.get("id"), exc)
            score = "1"
        item["VLM_eval_image_result"] = score

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Image consistency"):
            future.result()


def evaluate_text_chain(
    items: Iterable[Dict[str, Any]],
    judge: JudgeClient,
    max_workers: int,
) -> None:
    """Text judge: Compare problem text/reference answer with model answer and return score 1~5."""
    def worker(item: Dict[str, Any]) -> None:
        if item.get("_skip"):
            return
        problem = item.get("question") or item["_dataset"].get("question")
        ref_answer = item.get("text_answer") or item["_dataset"].get("text_answer")
        model_answer = item.get("output") or item.get("model_text")
        if not all([problem, ref_answer, model_answer]):
            logging.warning("Text eval missing fields id=%s", item.get("id"))
            item["eval_text_result"] = "1"
            return
        prompt = TEXT_STEP_PROMPT_TEMPLATE.format(
            problem=problem,
            reference_answer=ref_answer,
            model_answer=model_answer,
        )
        try:
            score = judge.text_chain(prompt)
        except Exception as exc:
            logging.error("Text evaluation failed id=%s: %s", item.get("id"), exc)
            score = "1"
        item["eval_text_result"] = score.strip()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Text evaluation"):
            future.result()


def _load_image_for_metric(path: Path, size: int = 256) -> Image.Image:
    """Uniformly load and resize image as input for subsequent metric calculations."""
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def evaluate_lpips(
    items: Iterable[Dict[str, Any]],
    pred_root: Optional[Path],
) -> None:
    metric = lpips.LPIPS(net="alex")
    for item in tqdm(items, desc="LPIPS"):
        if item.get("_skip"):
            continue
        reference: Optional[Path] = item.get("_reference_image")
        pred = _resolve_path(pred_root, item.get("image_4") or item.get("output_image") or item.get("output_image_path"))
        if not reference or not reference.exists() or not pred or not pred.exists():
            item["lpips_eval_image_result"] = "nan"
            continue
        try:
            img1 = _load_image_for_metric(reference)
            img2 = _load_image_for_metric(pred)
            tensor1 = torch.tensor(np.asarray(img1).transpose(2, 0, 1))[None, ...] / 255.0
            tensor2 = torch.tensor(np.asarray(img2).transpose(2, 0, 1))[None, ...] / 255.0
            score = metric.forward(tensor1, tensor2).item()
            item["lpips_eval_image_result"] = f"{score:.6f}"
        except Exception as exc:
            logging.error("LPIPS failed id=%s: %s", item.get("id"), exc)
            item["lpips_eval_image_result"] = "nan"


def evaluate_psnr(
    items: Iterable[Dict[str, Any]],
    pred_root: Optional[Path],
) -> None:
    for item in tqdm(items, desc="PSNR"):
        if item.get("_skip"):
            continue
        reference = item.get("_reference_image")
        pred = _resolve_path(pred_root, item.get("image_4") or item.get("output_image") or item.get("output_image_path"))
        if not reference or not reference.exists() or not pred or not pred.exists():
            item["psnr_eval_image_result"] = "nan"
            continue
        try:
            img1 = np.asarray(_load_image_for_metric(reference), dtype=np.float32) / 255.0
            img2 = np.asarray(_load_image_for_metric(pred), dtype=np.float32) / 255.0
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                score = float("inf")
            else:
                score = 10 * np.log10(1.0 / mse)
            item["psnr_eval_image_result"] = f"{score:.6f}"
        except Exception as exc:
            logging.error("PSNR failed id=%s: %s", item.get("id"), exc)
            item["psnr_eval_image_result"] = "nan"


def evaluate_ssim(
    items: Iterable[Dict[str, Any]],
    pred_root: Optional[Path],
) -> None:
    for item in tqdm(items, desc="SSIM"):
        if item.get("_skip"):
            continue
        reference = item.get("_reference_image")
        pred = _resolve_path(pred_root, item.get("image_4") or item.get("output_image") or item.get("output_image_path"))
        if not reference or not reference.exists() or not pred or not pred.exists():
            item["ssim_eval_image_result"] = "0"
            continue
        try:
            img1 = np.asarray(_load_image_for_metric(reference).convert("L"), dtype=np.float32)
            img2 = np.asarray(_load_image_for_metric(pred).convert("L"), dtype=np.float32)
            score, _ = ssim(img1, img2, full=True)
            if math.isnan(score) or math.isinf(score):
                logging.warning("SSIM returned %s for id=%s; falling back to 0.0", score, item.get("id"))
                score = 0
            item["ssim_eval_image_result"] = f"{score:.6f}"
        except Exception as exc:
            logging.error("SSIM failed id=%s: %s", item.get("id"), exc)
            item["ssim_eval_image_result"] = "0"


def parse_mid_process_result(text: str) -> Dict[str, Any]:
    result = {
        "Step Accuracy": 1,
        "Process Consistency": 1,
        "Problem-Solution Accuracy": 1,
        "Rationale": "",
        "_raw_judge_output": text,
    }
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("step accuracy"):
            try:
                result["Step Accuracy"] = int(stripped.split(":", 1)[1])
            except Exception:
                pass
        elif stripped.lower().startswith("process consistency"):
            try:
                result["Process Consistency"] = int(stripped.split(":", 1)[1])
            except Exception:
                pass
        elif stripped.lower().startswith("problem-solution accuracy"):
            try:
                result["Problem-Solution Accuracy"] = int(stripped.split(":", 1)[1])
            except Exception:
                pass
        elif stripped.lower().startswith("rationale"):
            result["Rationale"] = stripped.split(":", 1)[1].strip()
    if not result["Rationale"]:
        result["Rationale"] = "Parsing failure; fallback scores applied."
    return result


def evaluate_mid_process(
    items: Iterable[Dict[str, Any]],
    judge: JudgeClient,
    pred_root: Optional[Path],
    max_workers: int,
) -> None:
    """Mid-process judge: Evaluate three-dimensional scores based on problem text and generated long image."""
    def worker(item: Dict[str, Any]) -> None:
        if item.get("_skip"):
            return
        problem_text = item.get("question") or item["_dataset"].get("question")
        long_image_path = _resolve_path(pred_root, item.get("long_image_path") or item.get("mid_process_image"))
        if not problem_text or not long_image_path or not long_image_path.exists():
            logging.warning("Mid-process evaluation missing data id=%s", item.get("id"))
            item["Step Accuracy"] = 1
            item["Process Consistency"] = 1
            item["Problem-Solution Accuracy"] = 1
            item["Rationale"] = "Missing problem text or long image."
            return
        try:
            raw = judge.mid_process(problem_text, long_image_path)
            parsed = parse_mid_process_result(raw)
        except Exception as exc:
            logging.error("Mid-process judge failed id=%s: %s", item.get("id"), exc)
            parsed = {
                "Step Accuracy": 1,
                "Process Consistency": 1,
                "Problem-Solution Accuracy": 1,
                "Rationale": f"Exception: {exc}",
            }
        item.update(parsed)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Mid-process judge"):
            future.result()


def _safe_to_numeric(value: Any, default: float = 0.0) -> float:
    """Safely convert value to numeric, returning default value on conversion failure.
    
    Specifically handles "nan" strings and float('nan') cases, returning default value.
    """
    if value is None:
        return default
    
    # Handle string type "nan"
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ("nan", "none", ""):
            return default
        try:
            result = float(value)
            # Check if converted result is nan
            if math.isnan(result):
                return default
            return result
        except (ValueError, TypeError):
            return default
    
    # Handle numeric types
    if isinstance(value, (int, float)):
        result = float(value)
        # Check if result is nan or inf
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    
    return default


def generate_score_json(items: List[Dict[str, Any]], score_json_path: Path) -> None:
    """Generate score.json file containing normalized scoring metrics for all items.
    
    After summing all items, normalize using fixed dataset total count (1411) and generate a total result containing:
    - VLM-T: Sum of eval_text_result divided by (DATASET_TOTAL_COUNT * 5) * 100
    - VLM-I-Mid: Sum of averages of "Step Accuracy" and "Process Consistency" divided by (DATASET_TOTAL_COUNT * 5) * 100
    - VLM-I-Res: Sum of "VLM_eval_image_result" divided by (DATASET_TOTAL_COUNT * 5) * 100
    - LPIPS ×10-2: Sum of "lpips_eval_image_result" (multiplied by 10^-2) divided by (DATASET_TOTAL_COUNT * 100)
    - PSNR: Sum of "psnr_eval_image_result" divided by (DATASET_TOTAL_COUNT * 100)
    - SSIM ×10-2: Sum of "ssim_eval_image_result" (multiplied by 10^-2) divided by (DATASET_TOTAL_COUNT * 100)
    - VLM-I: Average of VLM-I-Mid and VLM-I-Res
    All results are rounded to 2 decimal places.
    """
    # Initialize sum variables
    total_vlm_t = 0.0
    total_vlm_i_mid = 0.0
    total_vlm_i_res = 0.0
    total_lpips_scaled = 0.0
    total_psnr = 0.0
    total_ssim_scaled = 0.0
    total_count = 0
    
    for item in items:
        if item.get("_skip"):
            continue
        
        total_count += 1
        
        # Get original values
        eval_text = _safe_to_numeric(item.get("eval_text_result", 0))
        step_acc = _safe_to_numeric(item.get("Step Accuracy", 0))
        process_cons = _safe_to_numeric(item.get("Process Consistency", 0))
        vlm_image = _safe_to_numeric(item.get("VLM_eval_image_result", 0))
        lpips = _safe_to_numeric(item.get("lpips_eval_image_result", 0))
        psnr = _safe_to_numeric(item.get("psnr_eval_image_result", 0))
        ssim = _safe_to_numeric(item.get("ssim_eval_image_result", 0))
        
        # Calculate metrics for each item
        vlm_t = eval_text
        vlm_i_mid = (step_acc + process_cons) / 2.0  # Average of Step Accuracy and Process Consistency
        vlm_i_res = vlm_image
        lpips_scaled = lpips * (10 ** -2)  # Multiply by 10^-2, i.e., divide by 100
        psnr_value = psnr
        ssim_scaled = ssim * (10 ** -2)  # Multiply by 10^-2, i.e., divide by 100
        
        # Accumulate sums
        total_vlm_t += vlm_t
        total_vlm_i_mid += vlm_i_mid
        total_vlm_i_res += vlm_i_res
        total_lpips_scaled += lpips_scaled
        total_psnr += psnr_value
        total_ssim_scaled += ssim_scaled
    
    # Calculate final scores with normalization
    # Use fixed dataset total count (1411) instead of actual evaluated count
    # VLM-T, VLM-I-Mid, VLM-I-Res: divide by (DATASET_TOTAL_COUNT * 5) * 100
    # LPIPS ×10-2, PSNR, SSIM ×10-2: divide by (DATASET_TOTAL_COUNT * 100)
    if DATASET_TOTAL_COUNT > 0:
        vlm_t_score = (total_vlm_t / (DATASET_TOTAL_COUNT * 5)) * 100
        vlm_i_mid_score = (total_vlm_i_mid / (DATASET_TOTAL_COUNT * 5)) * 100
        vlm_i_res_score = (total_vlm_i_res / (DATASET_TOTAL_COUNT * 5)) * 100
        lpips_score = (total_lpips_scaled / DATASET_TOTAL_COUNT) * 100
        psnr_score = (total_psnr / DATASET_TOTAL_COUNT) * 100
        ssim_score = (total_ssim_scaled / DATASET_TOTAL_COUNT) * 100
        vlm_i_score = (vlm_i_mid_score + vlm_i_res_score) / 2.0
    else:
        vlm_t_score = 0.0
        vlm_i_mid_score = 0.0
        vlm_i_res_score = 0.0
        lpips_score = 0.0
        psnr_score = 0.0
        ssim_score = 0.0
        vlm_i_score = 0.0
    
    # Generate total result
    total_score = {
        "Total Samples": total_count,
        "VLM-T": round(vlm_t_score, 2),
        "VLM-I-Mid": round(vlm_i_mid_score, 2),
        "VLM-I-Res": round(vlm_i_res_score, 2),
        "LPIPS ×10-2": round(lpips_score, 2),
        "PSNR": round(psnr_score, 2),
        "SSIM ×10-2": round(ssim_score, 2),
        "VLM-I": round(vlm_i_score, 2),
    }
    
    score_json_path.parent.mkdir(parents=True, exist_ok=True)
    with score_json_path.open("w", encoding="utf-8") as f:
        json.dump(total_score, f, ensure_ascii=False, indent=2)
    logging.info("Score JSON saved to %s", score_json_path)


def write_outputs(
    items: List[Dict[str, Any]],
    output_json: Path,
    output_jsonl: Optional[Path],
) -> None:
    """Write evaluation results, automatically filter auxiliary fields and generate JSON/JSONL files."""
    serializable = [
        {k: v for k, v in item.items() if not k.startswith("_")}
        for item in items
        if not item.get("_skip")
    ]

    merged_map: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    if output_json.exists():
        try:
            with output_json.open("r", encoding="utf-8") as f:
                existing_items = json.load(f)
            if isinstance(existing_items, list):
                for existing in existing_items:
                    key = str(existing.get("id"))
                    merged_map[key] = existing
        except Exception as exc:
            logging.warning("Failed to read existing results (%s); will overwrite.", exc)

    for item in serializable:
        key = str(item.get("id"))
        merged_map[key] = item

    merged_items = list(merged_map.values())

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(merged_items, f, ensure_ascii=False, indent=2)
    logging.info("Evaluation JSON saved to %s", output_json)

    if output_jsonl:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as f:
            for item in merged_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logging.info("Evaluation JSONL saved to %s", output_jsonl)
    
    # Generate score.json
    score_json_path = output_json.parent / "score.json"
    generate_score_json(items, score_json_path)


def main() -> None:
    _log_setup(LOG_FILE)

    dataset_path = DATASET_PATH
    model_output_path = MODEL_OUTPUT_PATH
    dataset_root = DATASET_ROOT
    pred_root = PRED_ROOT if PRED_ROOT else None

    logging.info("Loading dataset from %s", dataset_path)
    dataset_map = load_dataset(dataset_path)

    logging.info("Loading model outputs from %s", model_output_path)
    items = load_model_results(model_output_path)
    ensure_fields(items, dataset_map, dataset_root)

    judge = JudgeClient(JUDGE_URL, JUDGE_API_KEY, JUDGE_MODEL)

    if ENABLE_IMAGE_JUDGE:
        evaluate_image_consistency(items, judge, pred_root, MAX_WORKERS)
    if ENABLE_TEXT_JUDGE:
        evaluate_text_chain(items, judge, MAX_WORKERS)

    if ENABLE_LPIPS:
        evaluate_lpips(items, pred_root)
    if ENABLE_PSNR:
        evaluate_psnr(items, pred_root)
    if ENABLE_SSIM:
        evaluate_ssim(items, pred_root)

    if ENABLE_MID_PROCESS_JUDGE:
        evaluate_mid_process(items, judge, pred_root, MAX_WORKERS)

    output_jsonl = OUTPUT_JSONL if OUTPUT_JSONL else None
    write_outputs(items, OUTPUT_JSON, output_jsonl)


if __name__ == "__main__":
    main()

