"""Extra tests for the CardCrop fields and load_input. Safe to add alongside test_preprocess.py."""
import cv2
import numpy as np

from src.common.preprocess import normalise_card
from src.common.result import PhaseResult
from src.common.utils import load_input


def _card_on_background(angle=12):
    bg = np.full((900, 1200, 3), 200, np.uint8)
    card = np.full((540, 856, 3), 40, np.uint8)
    cv2.putText(card, "TEST", (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
    m = cv2.getRotationMatrix2D((428, 270), angle, 0.7)
    warped = cv2.warpAffine(card, m, (856, 540), borderValue=(200, 200, 200))
    bg[150:690, 150:1006] = warped
    return bg


def test_detected_crop_keeps_original_and_quad():
    img = _card_on_background()
    crop = normalise_card(img)
    assert crop.image.shape[:2] == (540, 856)
    assert crop.original is img
    if crop.detected:
        assert crop.quad is not None and crop.quad.shape == (4, 2)


def test_portrait_card_shaped_image_is_reliable():
    img = np.random.randint(0, 255, (856, 540, 3), np.uint8)   # portrait, ratio 1.585
    crop = normalise_card(img)
    assert crop.image.shape[:2] == (540, 856)
    assert crop.reliable


def test_square_image_is_not_reliable():
    img = np.random.randint(0, 255, (600, 600, 3), np.uint8)
    crop = normalise_card(img)
    assert not crop.detected
    assert not crop.reliable


def test_neutral_result():
    r = PhaseResult.neutral("could not check")
    assert r.score == 0.5


def test_load_input_image(tmp_path):
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), np.zeros((50, 80, 3), np.uint8))
    img, digital = load_input(p)
    assert img.shape[:2] == (50, 80) and digital is False


def test_load_input_pdf(tmp_path):
    import fitz
    p = tmp_path / "a.pdf"
    doc = fitz.open()
    doc.new_page(width=300, height=400).insert_text((50, 100), "hello")
    doc.save(str(p))
    img, digital = load_input(p)
    assert digital is True and img.ndim == 3