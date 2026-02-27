import io
from typing import Sequence

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from service.compute_salad import init_model, extract_embedding, extract_embeddings

app = FastAPI(title="SALAD Embed Service", version="0.1")
MAX_BATCH = 8


@app.on_event("startup")
def _startup() -> None:
    init_model()


@app.get("/healthz")
def healthz():
    return {"ok": True, "impl": "SALAD"}


async def _load_upload(image: UploadFile) -> Image.Image:
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty image upload")

    try:
        return Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc


def _to_json_payload(vectors) -> dict:
    return {
        "dim": int(vectors.shape[1]),
        "dtype": "f32",
        "count": int(vectors.shape[0]),
        "embeddings": vectors.tolist(),
    }


@app.post("/embed")
async def embed(image: UploadFile = File(...)):
    pil_img = await _load_upload(image)

    try:
        vec = extract_embedding(pil_img)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({"dim": int(vec.shape[0]), "dtype": "f32", "embedding": vec.tolist()})


@app.post("/embed_batch")
async def embed_batch(files: Sequence[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")
    if len(files) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Batch too large ({len(files)}); max supported is {MAX_BATCH}",
        )

    pil_imgs = [await _load_upload(upload) for upload in files]
    try:
        vectors = extract_embeddings(pil_imgs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(_to_json_payload(vectors))
