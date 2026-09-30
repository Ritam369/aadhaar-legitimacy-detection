import cv2
import numpy as np
import yaml
from src.phase0_checksum import phase0

CFG = yaml.safe_load(open("config.yaml"))["phase0"]


def fake_card(seed=0, bg=(235, 220, 200)):
    """Synthetic 'card': coloured background, photo block, logo, QR-like noise."""
    img = np.full((540, 856, 3), bg, np.uint8)
    cv2.rectangle(img, (40, 150), (240, 400), (90, 90, 90), -1)
    cv2.circle(img, (110, 70), 40, (30, 60, 200), -1)
    rng = np.random.default_rng(seed)
    img[300:500, 620:820] = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    cv2.putText(img, "SAMPLE", (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    return img


def test_same_card_scores_high():
    ref = fake_card()
    assert phase0.run(None, fake_card(), [ref], CFG).score > 0.9


def test_brightness_change_tolerated():
    ref = fake_card()
    dark = cv2.convertScaleAbs(fake_card(), alpha=0.75, beta=-10)
    assert phase0.run(None, dark, [ref], CFG).score > 0.6


def test_different_layout_and_colour_scores_low():
    ref = fake_card()
    other = np.full((540, 856, 3), (20, 200, 20), np.uint8)
    assert phase0.run(None, other, [ref], CFG).score < 0.3


def test_bad_number_forces_zero():
    ref = fake_card()
    assert phase0.run("234567890125", fake_card(), [ref], CFG).score == 0.0
