import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPERMEAL_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import main
    main.store.init()
    main._hits.clear()
    return TestClient(main.app)


def png(size=(64, 48), color=(200, 120, 40), mode="RGB") -> bytes:
    b = io.BytesIO()
    Image.new(mode, size, color).save(b, "PNG")
    return b.getvalue()


def jpeg_with_gps() -> bytes:
    im = Image.new("RGB", (120, 80), (10, 120, 60))
    ex = Image.Exif()
    ex[0x010F] = "TestPhone"                         # Make
    ex[0x8825] = {1: "N", 2: (28.0, 36.0, 0.0), 3: "E", 4: (77.0, 12.0, 0.0)}      # GPS block
    b = io.BytesIO()
    im.save(b, "JPEG", exif=ex)
    return b.getvalue()


def upload(client, data, name="meal.png", ctype="image/png"):
    return client.post("/api/analyze", files={"file": (name, data, ctype)})
