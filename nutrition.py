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

# More foods, grouped. Same rules as above: rounded per-100-g figures for the food as usually eaten, approximate, not verified.
_MORE: dict[str, tuple[float, float, float, float]] = {
    # Indian breakfasts, snacks and street food
    "pav bhaji": (150, 3.5, 22.0, 5.5), "vada pav": (260, 6.5, 36.0, 10.0), "medu vada": (300, 8.0, 30.0, 16.0), "uttapam": (150, 4.5, 24.0, 4.0),
    "pesarattu": (140, 6.0, 20.0, 4.0), "dhokla": (160, 6.0, 25.0, 4.0), "thepla": (300, 9.0, 42.0, 11.0), "aloo paratha": (260, 5.5, 38.0, 10.0),
    "bhatura": (330, 8.0, 46.0, 13.0), "kachori": (400, 7.0, 44.0, 22.0), "pakora": (300, 8.0, 28.0, 17.0), "bhel puri": (190, 4.0, 32.0, 5.0),
    "pani puri": (200, 3.5, 33.0, 6.0), "sabudana khichdi": (190, 2.0, 32.0, 6.0), "appam": (160, 3.0, 30.0, 2.5), "puttu": (150, 3.0, 30.0, 2.0),
    "idiyappam": (130, 2.5, 28.0, 0.8), "besan chilla": (190, 9.0, 22.0, 7.0), "moong dal chilla": (170, 9.0, 22.0, 5.0), "sprouts": (30, 3.0, 6.0, 0.2),
    "roasted chana": (360, 20.0, 58.0, 6.0), "makhana": (350, 10.0, 77.0, 0.5), "papad": (370, 25.0, 50.0, 3.0), "namkeen": (500, 14.0, 45.0, 30.0),
    # Indian meals: breads, rice, curries, non-veg
    "garlic naan": (300, 9.0, 48.0, 8.0), "kulcha": (280, 8.0, 48.0, 5.0), "rumali roti": (270, 8.0, 52.0, 3.0), "bhakri": (250, 6.5, 50.0, 3.0),
    "curd rice": (120, 3.5, 18.0, 3.5), "lemon rice": (160, 3.0, 28.0, 4.0), "tamarind rice": (170, 3.0, 28.0, 5.0), "jeera rice": (150, 3.0, 28.0, 3.0),
    "pulao": (160, 3.5, 27.0, 4.0), "fried rice": (170, 4.0, 27.0, 5.0), "egg fried rice": (180, 5.5, 27.0, 5.5),
    "aloo gobi": (100, 2.5, 11.0, 5.5), "palak paneer": (150, 8.0, 6.0, 11.0), "paneer butter masala": (220, 9.0, 9.0, 17.0), "chana masala": (160, 7.0, 20.0, 6.0),
    "dal makhani": (140, 6.0, 14.0, 7.0), "aloo matar": (110, 3.5, 14.0, 4.5), "baingan bharta": (90, 2.0, 8.0, 6.0), "bhindi masala": (90, 2.0, 9.0, 5.0),
    "kadhi": (90, 3.0, 8.0, 5.0), "raita": (60, 3.0, 5.0, 3.0), "aloo sabzi": (110, 2.0, 15.0, 5.0), "paneer tikka": (230, 14.0, 6.0, 17.0),
    "mutton curry": (190, 14.0, 4.0, 13.0), "butter chicken": (160, 12.0, 6.0, 10.0), "tandoori chicken": (190, 25.0, 3.0, 9.0), "chicken tikka": (180, 24.0, 3.0, 8.0),
    "seekh kebab": (230, 17.0, 4.0, 16.0), "keema": (220, 15.0, 5.0, 15.0), "fish fry": (230, 18.0, 10.0, 13.0), "prawn curry": (120, 13.0, 4.0, 6.0),
    "egg bhurji": (170, 11.0, 3.0, 12.0), "chicken biryani": (180, 10.0, 22.0, 6.0), "veg biryani": (150, 3.5, 25.0, 4.0),
    # Indian sweets and drinks
    "rasgulla": (186, 4.0, 37.0, 2.0), "kheer": (130, 3.5, 20.0, 4.0), "suji halwa": (330, 4.0, 45.0, 15.0), "gajar halwa": (230, 4.0, 30.0, 11.0),
    "besan laddu": (400, 6.0, 55.0, 18.0), "barfi": (400, 8.0, 55.0, 17.0), "rasmalai": (200, 6.0, 25.0, 8.0), "kulfi": (190, 5.0, 22.0, 9.0),
    "lassi": (70, 3.0, 10.0, 2.0), "masala chai": (45, 1.5, 7.0, 1.2), "buttermilk": (40, 3.0, 5.0, 1.0), "coconut water": (19, 0.7, 3.7, 0.2),
    "sugarcane juice": (70, 0.0, 18.0, 0.0), "tea": (1, 0.0, 0.3, 0.0), "coffee with milk": (40, 1.2, 6.0, 1.2),
    # East and Southeast Asian
    "veg momos": (170, 5.0, 26.0, 5.0), "chicken momos": (200, 10.0, 24.0, 7.0), "hakka noodles": (150, 4.0, 25.0, 4.0), "veg manchurian": (180, 4.0, 16.0, 11.0),
    "spring roll": (250, 6.0, 30.0, 12.0), "sushi": (145, 5.0, 28.0, 1.5), "ramen": (140, 6.0, 18.0, 4.5), "pad thai": (170, 7.0, 22.0, 6.0),
    "pho": (60, 4.0, 8.0, 1.0), "thai curry": (130, 7.0, 5.0, 9.0), "miso soup": (35, 2.5, 4.0, 1.2), "tempura": (230, 6.0, 22.0, 13.0),
    "dumplings": (200, 8.0, 26.0, 7.0), "tempeh": (190, 19.0, 9.0, 11.0), "kimchi": (15, 1.1, 2.4, 0.5),
    # Western breakfast, mains and fast food
    "fried egg": (196, 14.0, 0.8, 15.0), "scrambled eggs": (170, 11.0, 2.0, 13.0), "bacon": (540, 37.0, 1.4, 42.0), "sausage": (300, 12.0, 2.0, 27.0),
    "ham": (145, 21.0, 1.5, 6.0), "pancakes": (230, 6.0, 28.0, 10.0), "waffle": (290, 8.0, 37.0, 12.0), "french toast": (230, 8.0, 25.0, 11.0),
    "croissant": (406, 8.0, 46.0, 21.0), "bagel": (250, 10.0, 49.0, 1.5), "granola": (450, 10.0, 64.0, 18.0), "muesli": (360, 10.0, 66.0, 6.0),
    "hot dog": (290, 10.0, 18.0, 18.0), "chicken nuggets": (290, 15.0, 17.0, 18.0), "taco": (220, 9.0, 22.0, 11.0), "burrito": (200, 8.0, 27.0, 6.5),
    "quesadilla": (280, 12.0, 24.0, 15.0), "nachos": (345, 8.0, 36.0, 19.0), "lasagna": (135, 8.0, 13.0, 6.0), "spaghetti bolognese": (150, 7.0, 17.0, 5.5),
    "mac and cheese": (165, 7.0, 18.0, 7.0), "risotto": (150, 3.0, 24.0, 4.5), "garlic bread": (350, 8.0, 45.0, 15.0), "grilled cheese sandwich": (330, 12.0, 28.0, 19.0),
    "fish and chips": (230, 11.0, 20.0, 12.0), "steak": (250, 26.0, 0.0, 17.0), "roast chicken": (200, 27.0, 0.0, 10.0), "beef": (250, 26.0, 0.0, 15.0),
    "pork": (240, 27.0, 0.0, 14.0), "turkey": (190, 29.0, 0.0, 7.5), "mutton": (250, 25.0, 0.0, 17.0), "meatballs": (240, 14.0, 9.0, 16.0),
    "tuna": (116, 26.0, 0.0, 1.0), "sardines": (208, 25.0, 0.0, 11.0), "crab": (97, 19.0, 0.0, 1.5), "grilled fish": (130, 22.0, 0.0, 4.5),
    "mashed potato": (90, 2.0, 15.0, 3.0), "roast potatoes": (150, 2.5, 22.0, 6.0), "baked beans": (95, 5.0, 18.0, 0.4), "coleslaw": (150, 1.2, 14.0, 10.0),
    "caesar salad": (180, 6.0, 8.0, 14.0), "potato chips": (540, 6.5, 53.0, 34.0), "popcorn": (380, 12.0, 74.0, 4.0), "pizza slice": (270, 12.0, 33.0, 10.0),
    # Middle Eastern, Mediterranean and grains
    "hummus": (166, 8.0, 14.0, 10.0), "falafel": (330, 13.0, 32.0, 18.0), "shawarma": (200, 13.0, 17.0, 9.0), "pita": (275, 9.0, 55.0, 1.2),
    "tortilla": (300, 8.0, 50.0, 7.0), "couscous": (112, 3.8, 23.0, 0.2), "quinoa": (120, 4.4, 21.0, 1.9), "vermicelli": (140, 4.5, 28.0, 0.8),
    "multigrain bread": (265, 13.0, 43.0, 4.0), "rusk": (400, 10.0, 75.0, 6.0), "soup": (40, 1.5, 7.0, 1.0), "tomato soup": (40, 1.0, 8.0, 1.0),
    "chicken soup": (40, 3.0, 4.0, 1.5), "sweet corn soup": (55, 2.0, 10.0, 1.0),
    # Sweet things and desserts
    "cheesecake": (320, 5.5, 26.0, 22.0), "brownie": (410, 5.0, 55.0, 20.0), "apple pie": (240, 2.0, 34.0, 11.0), "muffin": (380, 5.0, 50.0, 18.0),
    "donut": (420, 5.0, 50.0, 22.0), "pudding": (120, 3.0, 20.0, 3.0), "custard": (120, 3.5, 17.0, 4.0), "milkshake": (110, 3.5, 18.0, 3.0),
    "smoothie": (70, 1.0, 16.0, 0.5), "chocolate": (535, 7.0, 59.0, 30.0), "candy": (390, 0.0, 98.0, 0.2),
    # Fruit, nuts and seeds
    "pineapple": (50, 0.5, 13.0, 0.1), "strawberries": (32, 0.7, 7.7, 0.3), "blueberries": (57, 0.7, 14.0, 0.3), "kiwi": (61, 1.1, 15.0, 0.5),
    "pear": (57, 0.4, 15.0, 0.1), "peach": (39, 0.9, 9.5, 0.3), "guava": (68, 2.6, 14.0, 1.0), "lychee": (66, 0.8, 17.0, 0.4), "jackfruit": (95, 1.7, 23.0, 0.6),
    "custard apple": (94, 2.0, 24.0, 0.6), "coconut": (354, 3.3, 15.0, 33.0), "dates": (282, 2.5, 75.0, 0.4), "raisins": (299, 3.0, 79.0, 0.5),
    "avocado": (160, 2.0, 9.0, 15.0), "cashews": (553, 18.0, 30.0, 44.0), "walnuts": (654, 15.0, 14.0, 65.0), "pistachios": (560, 20.0, 28.0, 45.0),
    "chia seeds": (486, 17.0, 42.0, 31.0), "dried fruit": (290, 3.0, 75.0, 0.5), "fruit salad": (50, 0.6, 13.0, 0.2), "lemon": (29, 1.1, 9.3, 0.3),
    # Vegetables
    "cabbage": (25, 1.3, 6.0, 0.1), "cauliflower": (25, 1.9, 5.0, 0.3), "peas": (81, 5.0, 14.0, 0.4), "corn": (96, 3.4, 21.0, 1.5), "onion": (40, 1.1, 9.3, 0.1),
    "capsicum": (20, 0.9, 4.6, 0.2), "brinjal": (25, 1.0, 6.0, 0.2), "okra": (33, 1.9, 7.5, 0.2), "pumpkin": (26, 1.0, 6.5, 0.1), "beetroot": (43, 1.6, 10.0, 0.2),
    "mushrooms": (22, 3.1, 3.3, 0.3), "green beans": (31, 1.8, 7.0, 0.2), "lettuce": (15, 1.4, 2.9, 0.2), "bitter gourd": (17, 1.0, 3.7, 0.2),
    "bottle gourd": (14, 0.6, 3.4, 0.0), "radish": (16, 0.7, 3.4, 0.1), "olives": (115, 0.8, 6.0, 11.0), "pickle": (150, 1.5, 8.0, 13.0),
    "boiled vegetables": (45, 2.0, 8.0, 0.4), "stir fried vegetables": (80, 2.5, 9.0, 4.0),
    # Dairy, spreads, sweeteners and staples
    "peanut butter": (588, 25.0, 20.0, 50.0), "jam": (250, 0.4, 65.0, 0.1), "honey": (304, 0.3, 82.0, 0.0), "sugar": (387, 0.0, 100.0, 0.0), "jaggery": (383, 0.4, 98.0, 0.1),
    "mayonnaise": (680, 1.0, 0.6, 75.0), "ketchup": (100, 1.0, 25.0, 0.1), "cream": (340, 2.0, 3.0, 36.0), "mozzarella": (280, 28.0, 3.0, 17.0), "egg white": (52, 11.0, 0.7, 0.2),
    "skimmed milk": (35, 3.4, 5.0, 0.1), "soy milk": (54, 3.3, 6.0, 1.8), "almond milk": (17, 0.6, 0.3, 1.1), "cheese slice": (310, 18.0, 6.0, 24.0), "condensed milk": (320, 8.0, 54.0, 8.7),
}
_clash = _MORE.keys() & _DATA.keys()
assert not _clash, f"foods defined twice: {sorted(_clash)}"
_DATA.update(_MORE)


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

ALIASES.update({
    "momos": "veg momos", "vegetable momos": "veg momos", "dim sum": "dumplings", "gyoza": "dumplings", "potsticker": "dumplings", "noodle soup": "ramen", "rice noodles": "noodles",
    "chow mein": "hakka noodles", "chowmein": "hakka noodles", "gobi manchurian": "veg manchurian", "manchurian": "veg manchurian", "sushi roll": "sushi", "maki": "sushi",
    "samosas": "samosa", "pakoda": "pakora", "pakoras": "pakora", "bhajji": "pakora", "bhaji": "pakora", "vada": "medu vada", "wada": "medu vada", "vada pao": "vada pav",
    "uthappam": "uttapam", "oothappam": "uttapam", "parotta": "paratha", "parantha": "paratha", "aloo ka paratha": "aloo paratha", "kulche": "kulcha", "butter naan": "naan",
    "tandoori roti": "roti", "missi roti": "roti", "jowar roti": "bhakri", "bajra roti": "bhakri", "rotis": "roti", "chapatis": "roti", "poori": "puri", "puris": "puri",
    "steamed rice": "cooked rice", "basmati rice": "cooked rice", "plain rice": "cooked rice", "boiled rice": "cooked rice", "veg pulao": "pulao", "vegetable pulao": "pulao",
    "veg fried rice": "fried rice", "vegetable fried rice": "fried rice", "chicken fried rice": "fried rice", "biryani rice": "biryani", "dum biryani": "biryani", "hyderabadi biryani": "chicken biryani",
    "vegetable biryani": "veg biryani", "paneer makhani": "paneer butter masala", "shahi paneer": "paneer butter masala", "saag paneer": "palak paneer", "mattar paneer": "aloo matar",
    "chole masala": "chana masala", "chickpea curry": "chana masala", "chana": "chickpeas", "black dal": "dal makhani", "maa ki dal": "dal makhani", "gobi aloo": "aloo gobi",
    "aloo curry": "aloo sabzi", "potato curry": "aloo sabzi", "brinjal curry": "baingan bharta", "okra curry": "bhindi masala", "bhindi": "okra", "ladyfinger": "okra", "lauki": "bottle gourd",
    "karela": "bitter gourd", "baingan": "brinjal", "eggplant": "brinjal", "aubergine": "brinjal", "bell pepper": "capsicum", "green pepper": "capsicum", "sweetcorn": "corn", "maize": "corn",
    "green peas": "peas", "mushroom": "mushrooms", "beans": "green beans", "french beans": "green beans", "gourd": "bottle gourd", "pickles": "pickle", "achar": "pickle",
    "yoghurt dip": "raita", "boondi raita": "raita", "mutton masala": "mutton curry", "lamb curry": "mutton curry", "goat curry": "mutton curry", "mince": "keema", "minced meat": "keema",
    "tandoori": "tandoori chicken", "chicken kebab": "chicken tikka", "kebab": "seekh kebab", "kabab": "seekh kebab", "murgh makhani": "butter chicken", "fried fish": "fish fry",
    "shrimp curry": "prawn curry", "egg scramble": "egg bhurji", "anda bhurji": "egg bhurji", "rasagulla": "rasgulla", "rice pudding": "kheer", "payasam": "kheer", "sheera": "suji halwa",
    "sooji halwa": "suji halwa", "rava kesari": "suji halwa", "carrot halwa": "gajar halwa", "laddoo": "besan laddu", "ladoo": "besan laddu", "laddu": "besan laddu", "burfi": "barfi",
    "sweet lassi": "lassi", "chai": "masala chai", "milk tea": "masala chai", "indian tea": "masala chai", "chaas": "buttermilk", "tender coconut water": "coconut water",
    "green tea": "tea", "black tea": "tea", "latte": "coffee with milk", "cappuccino": "coffee with milk", "milk coffee": "coffee with milk", "cold coffee": "coffee with milk",
    "scrambled egg": "scrambled eggs", "sunny side up": "fried egg", "egg fry": "fried egg", "bacon strips": "bacon", "sausages": "sausage", "pancake": "pancakes", "waffles": "waffle",
    "croissants": "croissant", "bagels": "bagel", "hotdog": "hot dog", "nuggets": "chicken nuggets", "tacos": "taco", "burritos": "burrito", "nacho": "nachos", "lasagne": "lasagna",
    "bolognese": "spaghetti bolognese", "spaghetti with meat sauce": "spaghetti bolognese", "macaroni and cheese": "mac and cheese", "mac n cheese": "mac and cheese",
    "cheese sandwich": "grilled cheese sandwich", "fish n chips": "fish and chips", "beef steak": "steak", "grilled steak": "steak", "roasted chicken": "roast chicken", "chicken roast": "roast chicken",
    "meatball": "meatballs", "canned tuna": "tuna", "sardine": "sardines", "grilled salmon": "salmon", "baked fish": "grilled fish", "mashed potatoes": "mashed potato",
    "roasted potatoes": "roast potatoes", "potato wedges": "roast potatoes", "beans on toast": "baked beans", "slaw": "coleslaw", "chicken caesar salad": "caesar salad", "crisps": "potato chips",
    "pizza slices": "pizza slice", "slice of pizza": "pizza slice", "humus": "hummus", "falafels": "falafel", "gyro": "shawarma", "wrap": "shawarma", "shawarma wrap": "shawarma", "chicken shawarma": "shawarma", "chicken wrap": "shawarma", "kebab roll": "shawarma", "pitta": "pita", "pita bread": "pita",
    "flour tortilla": "tortilla", "wholewheat bread": "whole wheat bread", "brown bread": "whole wheat bread", "sliced bread": "white bread", "grain bread": "multigrain bread",
    "seviyan": "vermicelli", "sevai": "vermicelli", "rusks": "rusk", "veg soup": "soup", "vegetable soup": "soup", "cheesecakes": "cheesecake", "brownies": "brownie", "donuts": "donut",
    "doughnut": "donut", "doughnuts": "donut", "muffins": "muffin", "cupcake": "muffin", "dessert pudding": "pudding", "ice cream sundae": "ice cream", "shake": "milkshake",
    "protein shake": "milkshake", "fruit smoothie": "smoothie", "milk chocolate": "chocolate", "sweets": "candy", "pineapples": "pineapple", "strawberry": "strawberries", "blueberry": "blueberries",
    "kiwi fruit": "kiwi", "pears": "pear", "peaches": "peach", "guavas": "guava", "litchi": "lychee", "kathal": "jackfruit", "sitaphal": "custard apple", "sugar apple": "custard apple",
    "date": "dates", "khajur": "dates", "raisin": "raisins", "kishmish": "raisins", "avocados": "avocado", "cashew": "cashews", "cashew nuts": "cashews", "kaju": "cashews",
    "walnut": "walnuts", "pistachio": "pistachios", "pista": "pistachios", "chia": "chia seeds", "mixed nuts": "almonds", "nuts": "almonds", "dry fruits": "dried fruit",
    "mixed fruit": "fruit salad", "fruit bowl": "fruit salad", "cabbage curry": "cabbage", "cauliflower curry": "cauliflower", "gobi": "cauliflower", "pea": "peas", "onions": "onion",
    "pumpkin curry": "pumpkin", "beetroots": "beetroot", "salad greens": "lettuce", "green salad": "salad", "olive": "olives", "steamed vegetables": "boiled vegetables",
    "sauteed vegetables": "stir fried vegetables", "stir fry vegetables": "stir fried vegetables", "veg stir fry": "stir fried vegetables", "nut butter": "peanut butter", "fruit jam": "jam",
    "gur": "jaggery", "mayo": "mayonnaise", "tomato ketchup": "ketchup", "tomato sauce": "ketchup", "fresh cream": "cream", "mozzarella cheese": "mozzarella", "egg whites": "egg white",
    "toned milk": "milk", "full fat milk": "milk", "low fat milk": "skimmed milk", "soya milk": "soy milk", "processed cheese": "cheese slice", "cheddar": "cheese", "milkmaid": "condensed milk",
})

# Words that describe how a food is served without changing what it is.
_HARMLESS = {"cooked", "boiled", "steamed", "fresh", "plain", "homemade", "home", "made", "sliced", "raw", "small", "large", "medium", "bowl", "plate",
             "of", "a", "an", "the", "some", "piece", "pieces", "serving", "portion", "half", "whole", "baked", "roasted", "indian", "style"}

NUTRITION: dict[str, dict[str, float]] = {k: dict(zip(KEYS, v)) for k, v in _DATA.items()}
_bad_alias = {a: t for a, t in ALIASES.items() if t not in NUTRITION}
assert not _bad_alias, f"aliases pointing at unknown foods: {_bad_alias}"


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
