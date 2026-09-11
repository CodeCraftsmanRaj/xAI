# ResNet18 Explainable Image Classification (Experiment 5)

Interactive app: upload an image → ResNet18 (pretrained on ImageNet-1K) predicts the class
→ Integrated Gradients (Captum) explains *why*, with a choice of baseline.

```
resnet18-xai-app/
├── backend/            FastAPI + PyTorch + Captum
│   ├── main.py
│   ├── model_utils.py
│   └── requirements.txt
├── frontend/            React + Vite + Tailwind UI
│   └── src/...
├── scripts/
│   └── generate_report.py   batch-runs your whole test-image folder, for the report
└── README.md
```

---

## 1. Backend setup (Python)

You need Python 3.9–3.12 (3.13 may not yet have prebuilt PyTorch wheels — if you're on 3.13
and installation fails, create a 3.11/3.12 venv instead).

```bash
cd resnet18-xai-app/backend

# create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

The first time you run the server, torchvision will **download the pretrained ResNet18
weights** (~45 MB) automatically — make sure you have internet access for that one-time step.

Run the API:

```bash
uvicorn main:app --reload --port 8000
```

Check it's alive: open http://localhost:8000 → you should see `{"status": "ok", ...}`.
Interactive API docs: http://localhost:8000/docs

---

## 2. Frontend setup (React)

In a **second terminal**:

```bash
cd resnet18-xai-app/frontend
npm install
npm run dev
```

Open the URL it prints (usually http://localhost:5173).

The frontend calls the backend at `http://localhost:8000` (hard-coded in
`src/App.jsx` as `API_BASE` — change it there if you run the backend elsewhere).

---

## 3. Using the app

1. Drag-and-drop or click to upload an image (any of your test images work, e.g.
   `good_01_dog.jpg` or `challenge_05_sketch_elephant.jpg`).
2. Click **Predict** → see the top-5 ImageNet-1K classes with confidence bars.
3. Click a predicted class to select it as the explanation target (top-1 is selected by default).
4. Pick a **baseline** (black / white / random noise / blurred input) and number of
   integration steps.
5. Click **Explain Prediction** → see the original image, the raw Integrated Gradients
   heatmap, and the heatmap overlaid on the original.
6. Change the baseline and re-run **Explain** on the same image/class to compare how the
   explanation shifts — this is exactly the "effect of different baselines" analysis the
   experiment asks for.

Works with **any** uploaded image, not just a fixed one — try your `Good_Cases` (should
localize cleanly on the object) versus `Challenging_Cases` (occlusion, camouflage, sketches,
unusual crops) and compare where the model actually looks.

---

## 4. Batch report generator (optional, great for the write-up)

Instead of clicking through the UI one image at a time, you can batch-process your whole
`ResNet18_Test_Images` folder and get a comparison figure (all 4 baselines side-by-side) plus
a CSV of predictions for every image — with the **same backend venv already active**:

```bash
cd resnet18-xai-app/backend
python ../scripts/generate_report.py \
  --input_dir "/home/raj_99/Projects/Sem7_Labs/xAI/Exp5_Project/ResNet18_Test_Images/ResNet18_Test_Images" \
  --output_dir ../report_outputs
```

This walks both `Good_Cases/` and `Challenging_Cases/`, and for each image saves
`<name>_explanation.png` (original + overlay for black/white/noise/blur baselines) into
`report_outputs/`, plus `report_outputs/predictions_summary.csv` with top-1/top-5 results.
Drop these PNGs straight into your lab report.

---

## 5. What to analyze in your report

- **Good cases** (`good_01_dog.jpg`, `good_02_elephant.jpg`, ...): does the IG heatmap
  concentrate on the animal/object itself, or does it leak into the background?
- **Challenging cases**: does the model get fooled by context (e.g. `challenge_02_laptop_forest.jpg`
  — does it focus on the laptop or on forest texture that might suggest a different class)?
  Does camouflage (`challenge_03_bird_camouflage.jpg`) spread attribution across the whole
  scene instead of the bird? Does the sketch (`challenge_05_sketch_elephant.jpg`) still get
  meaningful edge-based attribution despite not being a photo?
- **Baseline comparison**: for the same image/class, how different are the black vs. white
  vs. noise vs. blur attribution maps? A large convergence delta or very different-looking
  maps across baselines is a sign the explanation is less trustworthy / more baseline-sensitive.

---

## Troubleshooting

- **CORS / "Prediction request failed"**: make sure the backend is running on port 8000
  before using the UI.
- **Torch install issues**: if `pip install torch` is slow/fails, grab the right command for
  your OS/CUDA version from https://pytorch.org/get-started/locally/ and run that first,
  then `pip install -r requirements.txt` for the rest.
- **Model download blocked**: torchvision downloads weights from
  `download.pytorch.org` on first run — needs outbound internet once; after that it's cached
  in `~/.cache/torch/hub/checkpoints/`.
