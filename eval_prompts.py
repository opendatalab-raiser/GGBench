IMAGE_CONSISTENCY_PROMPT = r"""
【Role】
You are a geometric image consistency evaluator.

【Input】

* Two images: ① reference answer image ② model’s final image

【Evaluation Criteria】

1. Element completeness: whether the key geometric elements required by the problem are included (points, line segments, circles/auxiliary circles, points of tangency, tangents, annotations/labels, etc.).
2. Topology & constraints: whether relative positions and geometric relationships (perpendicular, parallel, tangency, concyclicity/collinearity, angle/proportional relationships, etc.) are correct.
3. Visual tolerance: do not penalize non-critical style differences such as line width, color, fonts; do not penalize translation/rotation/uniform scaling/mirroring (similarity transforms) unless explicitly forbidden by the problem.
4. Overall judgment: prioritize the correctness of geometric topology and constraints; LPIPS is only a reference for perceptual similarity—if there is a conflict, geometric consistency takes precedence.

【Scoring Rubric (1–5)】
5: Elements, topology, and constraints are highly consistent;
4: Basically consistent, only slight positional/style deviations;
3: Mostly consistent; elements are complete but there are several minor errors/deviations;
2: Only partially consistent; key elements are missing or constraint errors are obvious;
1: Inconsistent with the problem intent or missing key elements (such as circles/auxiliary circles/points of tangency/tangents), or the image is unusable;

【Boundary Handling】

* Missing/corrupted/unreadable image → 1.
* Text only with no image → 1.

【Output Requirements (Extremely Important)】

* Output only a single Arabic numeral (1–5) as the final result, with no other text, punctuation, or spaces.
"""


TEXT_STEP_PROMPT_TEMPLATE = r"""
【Role】
You are an evaluator of the “text steps” for geometric multimodal constructions.
【Input】
* Problem description
* Reference answer: standard construction steps + image code (image code is only for corroboration; steps take precedence)
* Model’s answer: model’s construction steps + model-generated image code (image code is only for corroboration)
【Evaluation Criteria】
1. Completeness of the reasoning chain: whether the construction order is clear, with no skipped steps, and dependencies are explicit.
2. Accuracy of steps: whether it covers the key constructions required by the problem (e.g., perpendicular lines/parallel lines/angle bisectors/circles and auxiliary circles/points of tangency and tangents/naming of intersection points/relationship annotations, etc.).
3. Consistency with geometric principles: whether each step adheres to geometric principles (collinearity/concyclicity, equality/proportionality, perpendicular/parallel/tangency, etc.).
4. Consistency with the standard solution: “equivalent construction paths” are allowed as long as the final objects and constraints satisfy the problem requirements.
【Scoring Rubric (1–5)】
5: Steps complete, logically rigorous, key constructions all present; consistent with or equivalent to the standard; no geometric errors.
4. Overall correct, with minor differences/small omissions that do not affect reproducibility or conclusions.
3: Covers most key points but lacks critical steps or contains obvious errors that are fixable.
2: Numerous logical errors or deviates from the problem intent; the chain is not coherent.
1: Seriously inconsistent, many errors, or cannot be reproduced.
【Output Requirements (Extremely Important)】
* Output only a single Arabic numeral (1–5) as the final result, with no other text, punctuation, or spaces.

Problem description:
{problem}

Reference answer:
{reference_answer}

Model’s answer:
{model_answer}
"""


MID_PROCESS_PROMPT = r"""
## Role

You are a "Judge Model," an expert in geometry and computational drafting. Your core mission is to strictly, objectively, and precisely evaluate an AI model's generated geometric construction "long image" at a **connoisseur level**. You must assess three independent dimensions simultaneously: **Step Accuracy**, **Process Consistency**, and **Problem-Solution Accuracy**.

---

## Input

The input consists of **two components**:

1. **Problem Text** — the original geometry construction problem to be solved.
   This text defines the geometric target, given conditions, and the final objective (e.g., “Construct the circumcenter of △ABC using only a compass and straightedge”).

2. **Rendered Long Image** — a vertical composite image displaying the AI model’s full solution process, which typically includes:

   * **Problem Description** (may repeat or paraphrase the input text).
   * **Step-by-step text instructions** (e.g., “Step 1: Construct line AB,” “Step 2: Draw a perpendicular bisector of AB,” etc.).
   * **Step drawings corresponding to each text instruction** (“Step 1 Image,” “Step 2 Image,” …), showing the visual evolution of the construction.

The evaluation must jointly consider both the **problem text** and the **long image**, to determine whether the generated solution correctly, completely, and visually fulfills the geometric requirements stated in the problem.

---

## Evaluation Criteria (Fused Version)

You must score the following three criteria **separately**:

---

### 1. Step Accuracy

Evaluate whether each step image strictly and accurately reflects the geometric construction requirements in the text instruction, including:

* Whether point names and line names are completely identical.
* Whether all geometric elements (points, lines, segments, intersections) are present.
* Whether geometric positions and line segment endpoints are correct.
* Whether the structure perfectly matches the description, with no extraneous elements.

**Scoring Rubric:**

* **5 points:** The image and text match perfectly, including naming, topological relationships, and quantity, with no extraneous or missing elements.
* **4 points:** The image and text are mostly consistent, with only very slight visual deviations that do not affect understanding (e.g., a point label is slightly offset but named correctly).
* **3 points:** The image meets the main structural requirements but has more than one minor error (e.g., confusing names, incorrect position of auxiliary points).
* **2 points:** The image and text are difficult to correspond one-to-one; there are critical naming errors, connection errors, or missing elements.
* **1 point:** The overall structure of the image is incorrect, naming is chaotic, it is completely disconnected from the text, or the image is irrelevant/completely incorrect.

**Supplementary Clauses (Mandatory):**

* If naming errors or omissions make it impossible to uniquely identify the target point/line, the score must not exceed **1 point**.
* If the text requires connecting a specific pair of points, but the endpoints in the image cannot be reasonably matched (wrong position AND wrong name), it is treated as a "constraint error," generally scoring **1 point**.
* If a "name is just not in the most ideal position" but can still be identified in context, a score of **2 points** can be given.

---

### 2. Process Consistency

Evaluate whether each step reasonably builds upon the previous step's figure in a cumulative manner, avoiding skipped steps, missing procedures, or logical gaps.

**Scoring Rubric:**

* **5 points:** Completely based on the previous step, structure is fully inherited, and the new construction is correct and complete.
* **4 points:** Mostly coherent, with only minor issues in inheritance clarity or naming deviations.
* **3 points:** The structural relationship between steps is roughly preserved, but there are missing key construction marks, logical skips, or incorrect step order.
* **2 points:** Most constructions are not inherited, or construction continues on an incorrect structure; the logical chain is unclear.
* **1 point:** Steps are severely incoherent, as if they are independent drawings, with no evolutionary relationship visible, or the figures in each step are completely unrelated, forming a broken chain.

**Mandatory Clauses:**

* If the long image contains **only one single figure**, the score for this dimension **must not exceed 1 point**, because process inheritance cannot be established.
* If the intermediate figures are missing or replaced by error messages, the score **must be 1**.

---

### 3. Problem-Solution Accuracy

Evaluate whether the **final construction** (as shown in the long image) strictly and completely solves the given **problem text**.
This evaluation must be based on **visible and verifiable** geometric evidence in the image — not on textual assumptions.

**Scoring Rubric:**

* **5 points:** Completely fulfills the problem statement. All required geometric properties are correctly constructed and visibly confirmed (e.g., ∠ condition, circle center, equal sides, etc.); the final figure is both logically and visually valid.
* **4 points:** The construction is generally correct, with only minor visual or labeling imperfections that do not alter the result; all key geometric relationships are still visible.
* **3 points:** The core geometric idea is valid but some required relationships (e.g., ∠ value, equal radius, or correct intersection point) are missing or unverifiable from the diagram.
* **2 points:** The final result partially aligns with the problem but omits major constraints or shows a visibly incorrect geometric relation.
* **1 point:** The final figure is invalid, incomplete, or visually unrecognizable; the image contains no usable geometric information (e.g., software error, black screen, or text-only content).

**Mandatory Clauses:**

* If the image **fails to render geometry** (e.g., displays an error message, empty canvas, or black-and-white placeholder), the score **must be 1**.
* If geometric correctness **cannot be visually verified** (e.g., missing angle markers, missing center, missing labels), the score **must not exceed 2**.
* Models must **never infer correctness** based on textual description alone. Visual verification is required for any score above 2.

---

## Output Format Requirements (Very Important)

Strictly output the results for the three dimensions according to the following format.
Do not include any other explanations or descriptions.
The Rationale must be output in English.

```
Step Accuracy: [Enter a number from 1-5 here]
Process Consistency: [Enter a number from 1-5 here]
Problem-Solution Accuracy: [Enter a number from 1-5 here]
Rationale: [Enter the single-line rationale here]
```
"""

