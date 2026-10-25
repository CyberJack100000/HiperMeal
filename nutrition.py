"""Nutrition lookup.

The values below are rounded, approximate per-100-g figures for common foods (cooked or ready-to-eat unless the name says otherwise). They are
good enough to demonstrate the workflow and give a rough idea; they are NOT a verified food-composition database. Replace them with a cited
source (for example IFCT or USDA FoodData Central) before anyone relies on the numbers.

Matching is deliberately conservative: a name is only matched when it is the same food, a known alias, a trivial spelling variant, or the
food plus harmless words ("boiled white rice"). Anything else (for example "egg curry", which is not a boiled egg) is reported as unknown so
the person can type their own values, rather than silently attaching the wrong food's numbers.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

KEYS = ("kcal", "protein_g", "carbs_g", "fat_g")

# name: (kcal, protein_g, carbs_g, fat_g) per 100 g
_DATA: dict[str, tuple[float, float, float, float]] = {
    # Indian staples
    "roti": (297, 8.2, 55.0, 7.5), "paratha": (326, 7.0, 45.0, 13.0), "naan": (290, 9.0, 50.0, 5.5), "puri": (350, 7.5, 45.0, 16.0),
    "dal": (116, 7.6, 20.1, 0.4), "cooked rice": (130, 2.7, 28.2, 0.3), "white rice": (130, 2.7, 28.2, 0.3), "brown rice": (112, 2.3, 23.5, 0.8),
    "chickpeas": (164, 8.9, 27.4, 2.6), "rajma": (127, 8.7, 22.8, 0.5), "sambar": (50, 2.5, 8.0, 1.0), "khichdi": (110, 4.0, 18.0, 2.5),
    "upma": (120, 3.0, 18.0, 4.0), "poha": (130, 2.5, 25.0, 2.5), "idli": (146, 4.0, 29.0, 0.7), "dosa": (168, 4.0, 29.0, 3.7),
    "samosa": (308, 6.0, 32.0, 17.0), "paneer": (265, 18.3, 1.2, 20.8), "mixed vegetables": (75, 2.5, 12.0, 2.2),
    "chicken curry": (150, 12.0, 5.0, 9.0), "fish curry": (130, 14.0, 3.0, 7.0), "egg curry": (150, 9.0, 4.0, 11.0), "biryani": (170, 9.0, 22.0, 5.0),
    "gulab jamun": (300, 5.0, 45.0, 12.0), "jalebi": (380, 2.0, 60.0, 15.0),
    # protein
    "egg": (155, 13.0, 1.1, 11.0), "boiled egg": (155, 13.0, 1.1, 11.0), "omelette": (154, 11.0, 0.6, 12.0), "chicken breast": (165, 31.0, 0.0, 3.6),
    "fried chicken": (246, 19.0, 10.0, 14.0), "salmon": (206, 22.0, 0.0, 12.0), "fish": (140, 24.0, 0.0, 4.0), "prawns": (99, 24.0, 0.2, 0.3),
    "tofu": (76, 8.0, 1.9, 4.8),
    # dairy and fats
    "yogurt": (61, 3.5, 4.7, 3.3), "curd": (61, 3.5, 4.7, 3.3), "milk": (61, 3.2, 4.8, 3.3), "cheese": (403, 25.0, 1.3, 33.0),
    "butter": (717, 0.9, 0.1, 81.0), "ghee": (900, 0.0, 0.0, 100.0),
    # grains, bread, pasta, fast food
    "oats": (389, 16.9, 66.3, 6.9), "cornflakes": (357, 7.0, 84.0, 0.4), "white bread": (265, 9.0, 49.0, 3.2), "whole wheat bread": (247, 13.0, 41.0, 3.4),
    "pasta": (158, 5.8, 31.0, 0.9), "noodles": (138, 4.5, 25.0, 2.1), "pizza": (266, 11.0, 33.0, 10.0), "burger": (295, 17.0, 24.0, 14.0),
    "french fries": (312, 3.4, 41.0, 15.0), "sandwich": (250, 10.0, 30.0, 9.0),
    # vegetables
    "potato": (87, 1.9, 20.0, 0.1), "sweet potato": (90, 2.0, 21.0, 0.2), "salad": (20, 1.5, 3.5, 0.2), "cucumber": (15, 0.7, 3.6, 0.1),
    "tomato": (18, 0.9, 3.9, 0.2), "carrot": (41, 0.9, 9.6, 0.2), "broccoli": (35, 2.4, 7.2, 0.4), "spinach": (23, 3.0, 3.8, 0.3),
    # fruit and nuts
    "banana": (89, 1.1, 22.8, 0.3), "apple": (52, 0.3, 13.8, 0.2), "mango": (60, 0.8, 15.0, 0.4), "orange": (47, 0.9, 12.0, 0.1),
    "grapes": (69, 0.7, 18.0, 0.2), "watermelon": (30, 0.6, 7.6, 0.2), "papaya": (43, 0.5, 11.0, 0.3), "pomegranate": (83, 1.7, 19.0, 1.2),
    "peanuts": (567, 26.0, 16.0, 49.0), "almonds": (579, 21.0, 22.0, 50.0),
    # sweets and drinks
    "ice cream": (207, 3.5, 24.0, 11.0), "dark chocolate": (546, 4.9, 61.0, 31.0), "cake": (350, 5.0, 50.0, 15.0), "biscuits": (450, 6.0, 70.0, 16.0),
    "cola": (42, 0.0, 10.6, 0.0), "orange juice": (45, 0.7, 10.4, 0.2), "black coffee": (1, 0.1, 0.0, 0.0),
}

ALIASES = {
    "rice": "cooked rice", "steamed rice": "cooked rice", "chapati": "roti", "chapati/roti": "roti", "phulka": "roti", "flatbread": "roti",
    "lentils": "dal", "lentil curry": "dal", "lentil soup": "dal", "dal tadka": "dal", "toor dal": "dal", "moong dal": "dal", "dal fry": "dal",
    "chole": "chickpeas", "chana": "chickpeas", "kidney beans": "rajma", "vegetables": "mixed vegetables", "veg": "mixed vegetables",
    "sabzi": "mixed vegetables", "curd/yogurt": "curd", "dahi": "curd", "yoghurt": "yogurt", "eggs": "egg", "scrambled egg": "omelette",
    "chicken": "chicken breast", "grilled chicken": "chicken breast", "chips": "french fries", "fries": "french fries", "bread": "white bread",
    "toast": "white bread", "spaghetti": "pasta", "macaroni": "pasta", "juice": "orange juice", "coffee": "black coffee", "peanut": "peanuts",
    "almond": "almonds", "pomegranate seeds": "pomegranate", "idly": "idli", "dosai": "dosa", "masala dosa": "dosa", "porridge": "oats",
    "oatmeal": "oats", "cottage cheese": "paneer", "prawn": "prawns", "shrimp": "prawns", "potatoes": "potato", "tomatoes": "tomato",
    "carrots": "carrot", "grape": "grapes", "chickpea": "chickpeas", "cookies": "biscuits", "cookie": "biscuits", "soda": "cola",
}

# Words that describe how a food is served without changing what it is.
_HARMLESS = {"cooked", "boiled", "steamed", "fresh", "plain", "homemade", "home", "made", "sliced", "raw", "small", "large", "medium", "bowl", "plate",
             "of", "a", "an", "the", "some", "piece", "pieces", "serving", "portion", "half", "whole", "baked", "roasted", "indian", "style"}

NUTRITION: dict[str, dict[str, float]] = {k: dict(zip(KEYS, v)) for k, v in _DATA.items()}


@dataclass(frozen=True)
class Match:
    name: str
    per_100g: dict[str, float]
    how: str          # exact | alias | variant | close

    def as_dict(self) -> dict:
        return {"matched_name": self.name, "nutrition": self.per_100g, "match": self.how}


def normalise(name: str) -> str:
    s = re.sub(r"\([^)]*\)", " ", name.lower())
    s = re.sub(r"[^a-z/ ]+", " ", s)
    return " ".join(s.split())


def _get(key: str, how: str) -> Match | None:
    if key in NUTRITION:
        return Match(key, NUTRITION[key], how)
    if key in ALIASES:
        return Match(ALIASES[key], NUTRITION[ALIASES[key]], "alias")
    return None


def find(name: str) -> Match | None:
    """The best honest match for a food name, or None when it is not clearly one of the known foods."""
    key = normalise(name)
    if not key:
        return None
    hit = _get(key, "exact")
    if hit:
        return hit
    words = key.split()
    # the food plus harmless words: "boiled white rice", "a bowl of dal"
    core = " ".join(w for w in words if w not in _HARMLESS)
    if core and core != key:
        hit = _get(core, "variant")
        if hit:
            return hit
    # trailing plural
    for cand in {core or key, key}:
        if cand.endswith("es") and _get(cand[:-2], "variant"):
            return _get(cand[:-2], "variant")
        if cand.endswith("s") and _get(cand[:-1], "variant"):
            return _get(cand[:-1], "variant")
    # a small typo ("chapathi", "panner")
    pool = list(NUTRITION) + list(ALIASES)
    close = difflib.get_close_matches(core or key, pool, n=1, cutoff=0.86)
    if close:
        m = _get(close[0], "close")
        return Match(m.name, m.per_100g, "close") if m else None      # always labelled as a guess, whatever route found it
    return None


def scale(per_100g: dict[str, float], grams: float) -> dict[str, float]:
    return {k: per_100g[k] * grams / 100.0 for k in KEYS}
