from typing import *
from numpy import ndarray

import math

import numpy as np
from shapely.geometry import Polygon, Point
from nltk.corpus import cmudict


def compute_loc_rel(corners1: ndarray, corners2: ndarray, name1: str, name2: str) -> Optional[str]:
    assert corners1.shape == corners2.shape == (8, 3), "Shape of corners should be (8, 3)."

    center1 = corners1.mean(axis=0)
    center2 = corners2.mean(axis=0)

    d = center1 - center2
    theta = math.atan2(d[2], d[0])  # range -pi to pi
    distance = (d[2]**2 + d[0]**2)**0.5  # center distance on the ground

    box1 = corners1[[0, 1, 4, 5], :][:, [0, 2]]  # 4 corners of the bottom face (0&5, 1&4 are opposite corners)
    box2 = corners2[[0, 1, 4, 5], :][:, [0, 2]]

    # Note that bounding boxes might not be axis-aligned
    polygon1, polygon2 = Polygon(box1[[0, 1, 3, 2], :]), Polygon(box2[[0, 1, 3, 2], :])  # change the order to be convex
    point1, point2 = Point(center1[[0, 2]]), Point(center2[[0, 2]])

    # Initialize the relationship
    p = None

    # Horizontal relationship: "left"/"right"/"front"/"behind"
    if theta >= 3 * math.pi / 4 or theta < -3 * math.pi / 4:
        p = "left of"
    elif -3 * math.pi / 4 <= theta < -math.pi / 4:
        p = "behind"
    elif -math.pi / 4 <= theta < math.pi / 4:
        p = "right of"
    elif math.pi / 4 <= theta < 3 * math.pi / 4:
        p = "in front of"

    # Vertical relationship: "above"/"below"
    if point1.within(polygon2) or point2.within(polygon1):
        delta1 = center1[1] - center2[1]
        delta2 = (
            corners1[:, 1].max() - corners1[:, 1].min() +
            corners2[:, 1].max() - corners2[:, 1].min()
        ) / 2.
        if (delta1 - delta2) >= 0. or "lamp" in name1:
            # Indicate that:
            # (1) delta1 > 0. (because always delta2 > 0.): `center1` is above `center2`
            # (2) delta1 >= delta2: `corners1` and `corners2` not intersect vertically
            # ==> `corners1` is completely above `corners2`
            # Or the subject is a lamp, which is always above other objects
            p = "above"
            return p
        if (-delta1 - delta2) >= 0. or "lamp" in name2:
            # ==> `corners1` is completely below `corners2`
            # Or the object is a lamp, which is always above other objects
            p = "below"
            return p

    if distance > 3.:
        return None  # too far away
    else:
        if distance < 1.:
            p = "closely " + p
        return p


"""
Taken from https://stackoverflow.com/questions/20336524/verify-correct-use-of-a-and-an-in-english-texts-python
"""

def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
    for syllables in pronunciations.get(word, []):
        return syllables[0][-1].isdigit()


def get_article(word):
    word = word.split(" ")[0]
    article = "an" if starts_with_vowel_sound(word) else "a"
    return article


################################################################


def reverse_rel(rel: str) -> str:
    return {
        "above": "below",
        "below": "above",
        "in front of": "behind",
        "behind": "in front of",
        "left of": "right of",
        "right of": "left of",
        "closely in front of": "closely behind",
        "closely behind": "closely in front of",
        "closely left of": "closely right of",
        "closely right of": "closely left of"
    }[rel]


def rotate_rel(rel: str, r: float) -> str:
    assert r in [0.0, np.pi * 0.5, np.pi, np.pi * 1.5]

    if rel in ["above", "below"]:
        return rel

    if r == 0.0:
        return rel
    elif r == np.pi * 0.5:
        return ("closely " if "closely " in rel else "") + \
            {
                "in front of": "right of",
                "behind": "left of",
                "left of": "in front of",
                "right of": "behind"
            }[rel.replace("closely ", "")]
    elif r == np.pi:
        return ("closely " if "closely " in rel else "") + \
            {
                "in front of": "behind",
                "behind": "in front of",
                "left of": "right of",
                "right of": "left of"
            }[rel.replace("closely ", "")]
    elif r == np.pi * 1.5:
        return ("closely " if "closely " in rel else "") + \
            {
                "in front of": "left of",
                "behind": "right of",
                "left of": "behind",
                "right of": "in front of"
            }[rel.replace("closely ", "")]


def model_desc_from_info(cate: str, style: str, theme: str, material: str, seed=None):
    cate_name = {
        "desk":                                    "desk",
        "nightstand":                              "nightstand",
        "king-size bed":                           "double bed",
        "single bed":                              "single bed",
        "kids bed":                                "kids bed",
        "ceiling lamp":                            "ceiling lamp",
        "pendant lamp":                            "pendant lamp",
        "bookcase/jewelry armoire":                "bookshelf",
        "tv stand":                                "tv stand",
        "wardrobe":                                "wardrobe",
        "lounge chair/cafe chair/office chair":    "lounge chair",
        "dining chair":                            "dining chair",
        "classic chinese chair":                   "classic chinese chair",
        "armchair":                                "armchair",
        "dressing table":                          "dressing table",
        "dressing chair":                          "dressing chair",
        "corner/side table":                       "corner side table",
        "dining table":                            "dining table",
        "round end table":                         "round end table",
        "drawer chest/corner cabinet":             "cabinet",
        "sideboard/side cabinet/console table":    "console table",
        "children cabinet":                        "children cabinet",
        "shelf":                                   "shelf",
        "footstool/sofastool/bed end stool/stool": "stool",
        "coffee table":                            "coffee table",
        "loveseat sofa":                           "loveseat sofa",
        "three-seat/multi-seat sofa":              "multi-seat sofa",
        "l-shaped sofa":                           "l-shaped sofa",
        "lazy sofa":                               "lazy sofa",
        "chaise longue sofa":                      "chaise longue sofa",
        "barstool":                                "barstool",
        "wine cabinet":                            "wine cabinet"
    }[cate.lower().replace(" / ", "/")]

    attrs = []
    if style is not None and style != "Others":
        attrs.append(style.replace(" ", "-").lower())
    if material is not None and material != "Others":
        attrs.append(material.replace(" ", "-").lower())
    if theme is not None:
        attrs.append(theme.replace(" ", "-").lower())

    if seed is not None:
        np.random.seed(seed)
    attr = np.random.choice(attrs) + " " if len(attrs) > 0 else ""

    return attr + cate_name


################################################################


def fill_templates(
    desc: Dict[str, List],
    object_types: List[str], predicate_types: List[str],
    object_descs: Optional[List[str]]=None,
    seed: Optional[int]=None,
    return_obj_ids=False
) -> Tuple[str, Dict[int, int], List[Tuple[int, int, int]], List[Tuple[str, str]]]:
    if object_descs is None:
        assert object_types is not None

    if seed is not None:
        np.random.seed(seed)

    obj_class_ids = desc["obj_class_ids"]  # map from object index to class id

    # Describe the relations between the main objects and others
    selected_relation_indices = np.random.choice(
        len(desc["obj_relations"]),
        min(np.random.choice([1, 2]), len(desc["obj_relations"])),  # select 1 or 2 relations
        replace=False
    )
    selected_relations = [desc["obj_relations"][idx] for idx in selected_relation_indices]
    selected_relations = [
        (int(obj_class_ids[s]), int(p), int(obj_class_ids[o]))
        for s, p, o in selected_relations
    ]  # e.g., [(4, 2, 18), ...]; 4, 18 are class ids; 2 is predicate id
    selected_descs = []
    selected_sentences = []
    selected_object_ids = []  # e.g., [0, ...]; 0 is object id
    for idx in selected_relation_indices:
        s, p, o = desc["obj_relations"][idx]
        s, p, o = int(s), int(p), int(o)
        if object_descs is None:
            s_name = object_types[obj_class_ids[s]].replace("_", " ")
            o_name = object_types[obj_class_ids[o]].replace("_", " ")
            p_str = predicate_types[p]
            if np.random.rand() > 0.5:
                subject = f"{get_article(s_name).replace('a', 'A')} {s_name}"
                predicate = f" is {p_str} "
                object = f"{get_article(o_name)} {o_name}."
            else:  # 50% of the time to reverse the order
                subject = f"{get_article(o_name).replace('a', 'A')} {o_name}"
                predicate = f" is {reverse_rel(p_str)} "
                object = f"{get_article(s_name)} {s_name}."
        else:
            if np.random.rand() < 0.75:
                s_name = object_descs[s]
            else:  # 25% of the time to use the object type as the description
                s_name = object_types[obj_class_ids[s]].replace("_", " ")
                s_name = f"{get_article(s_name)} {s_name}"  # "a" or "an" is added
            if np.random.rand() < 0.75:
                o_name = object_descs[o]
            else:
                o_name = object_types[obj_class_ids[o]].replace("_", " ")
                o_name = f"{get_article(o_name)} {o_name}"

            p_str = predicate_types[p]
            rev_p_str = reverse_rel(p_str)

            if p_str in ["left of", "right of"]:
                if np.random.rand() < 0.5:
                    p_str = "to the " + p_str
                    rev_p_str = "to the " + rev_p_str
            elif p_str in ["closely left of", "closely right of"]:
                if np.random.rand() < 0.25:
                    p_str = "closely to the " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "closely to the " + rev_p_str.split(" ")[-2] + " of"
                elif np.random.rand() < 0.5:
                    p_str = "to the close " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "to the close " + rev_p_str.split(" ")[-2] + " of"
                elif np.random.rand() < 0.75:
                    p_str = "to the near " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "to the near " + rev_p_str.split(" ")[-2] + " of"

            if np.random.rand() < 0.5:
                verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                if "lamp" in s_name:
                    verbs += ["Hang", "Install"]
                verb = verbs[np.random.choice(len(verbs))]
                subject = f"{verb} {s_name}"
                predicate = f" {p_str} "
                object = f"{o_name}."
                selected_descs.append((s_name, o_name))
                selected_object_ids.append(s)
            else:  # 50% of the time to reverse the order
                verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                if "lamp" in o_name:
                    verbs += ["Hang", "Install"]
                verb = verbs[np.random.choice(len(verbs))]
                subject = f"{verb} {o_name}"
                predicate = f" {rev_p_str} "
                object = f"{s_name}."
                selected_descs.append((o_name, s_name))
                selected_object_ids.append(o)
        selected_sentences.append(subject + predicate + object)

    text = ""
    conjunctions = [" Then, ", " Next, ", " Additionally, ", " Finnally, ", " And ", " "]
    for i, sentence in enumerate(selected_sentences):
        if i == 0:
            text += sentence
        else:
            conjunction = conjunctions[np.random.choice(len(conjunctions))]
            while conjunction == " Finnally, " and i != len(selected_sentences)-1:
                # "Finally" should be used only in the last sentence
                conjunction = conjunctions[np.random.choice(len(conjunctions))]
            if conjunction != " ":
                sentence = sentence[0].lower() + sentence[1:]
            text += conjunction + sentence

    if return_obj_ids:
        return text, selected_relations, selected_descs, selected_object_ids
    else:
        return text, selected_relations, selected_descs  # return `selected_relations`, `selected_descs` for evaluation
