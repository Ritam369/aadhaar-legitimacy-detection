import cv2
import numpy as np
import yaml
from src.common.preprocess import CARD_SIZE, normalise_card
from src.phase0_checksum.layout_colour import layout_similarity
from test_phase0_image import fake_card


def card_on_table(angle=15, persp=True):
    card = fake_card()
    canvas = np.full((1000, 1400, 3), (40, 90, 40), np.uint8)
    src = np.float32([[0, 0], [855, 0], [855, 539], [0, 539]])
    dst = np.float32([[250, 200], [1050, 150], [1100, 700], [200, 650]]) if persp else \
        np.float32([[300, 200], [1156, 200], [1156, 740], [300, 740]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(card, m, (1400, 1000), dst=canvas,
                               borderMode=cv2.BORDER_TRANSPARENT)


def test_detects_and_straightens_tilted_card():
    crop = normalise_card(card_on_table())
    assert crop.detected, crop.note
    assert crop.image.shape[1::-1] == CARD_SIZE
    assert layout_similarity(crop.image, fake_card()) > 0.7


def test_portrait_card_rotated_to_landscape():
    photo = cv2.rotate(card_on_table(persp=False), cv2.ROTATE_90_CLOCKWISE)
    crop = normalise_card(photo)
    assert crop.detected, crop.note
    h, w = crop.image.shape[:2]
    assert w > h


def test_already_cropped_card_is_just_resized():
    crop = normalise_card(fake_card())
    assert crop.image.shape[1::-1] == CARD_SIZE
    assert crop.detected is False


def test_blank_image_falls_back_without_crashing():
    crop = normalise_card(np.full((600, 900, 3), 128, np.uint8))
    assert crop.detected is False and crop.image.shape[1::-1] == CARD_SIZE


def test_upside_down_card_still_matches():
    import cv2 as _cv
    from src.phase0_checksum import phase0
    cfg = yaml.safe_load(open("config.yaml"))["phase0"]
    ref = fake_card()
    flipped = _cv.rotate(card_on_table(persp=False), _cv.ROTATE_180)
    assert phase0.run(None, flipped, [ref], cfg).score > 0.6
