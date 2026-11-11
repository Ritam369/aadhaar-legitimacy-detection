import numpy as np
import yaml
from src.phase0_checksum import phase0
from test_phase0_image import fake_card

CFG = yaml.safe_load(open("config.yaml"))["phase0"]


def test_masked_number_is_neutral_not_zero():
    r = phase0.run("XXXX XXXX 0124")
    assert r.score == 0.5 and r.details["checksum"] == "masked"


def test_masked_number_does_not_change_image_score():
    ref, img = fake_card(), fake_card()
    plain = phase0.run(None, img, [ref], CFG).score
    masked = phase0.run("XXXX XXXX 0124", img, [ref], CFG).score
    assert masked == plain


def test_invalid_number_still_hard_zero():
    r = phase0.run("234567890125", fake_card(), [fake_card()], CFG)
    assert r.score == 0.0 and r.details["checksum"] == "invalid"


def test_non_card_shaped_image_is_skipped_not_scored():
    blank = np.full((600, 600, 3), 128, np.uint8)      # square, no outline
    r = phase0.run(None, blank, [fake_card()], CFG)
    assert r.details["skipped"] is True and r.score == 0.5


def test_details_report_layout_score():
    r = phase0.run(None, fake_card(), [fake_card()], CFG)
    assert r.details["layout"] is not None and r.details["skipped"] is False