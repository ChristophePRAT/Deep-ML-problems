import math
from collections import Counter
from typing import Any, Dict, List, Union


def calculate_entropy(labels: List[Any]) -> float:
    """
    Compute the Shannon entropy of the list of labels.
    labels: list of any hashable items.
    Returns a Python float.
    """
    categories = set(labels)

    entropy = 0.0
    total_count = len(labels)

    for category in categories:
        prob = labels.count(category) / total_count
        entropy -= prob * math.log2(prob)
    return entropy


def calculate_information_gain(
    s_set: List[Dict[str, Any]], attr: str, target_attr: str = "PlayTennis"
) -> float:
    """
    Computes the information gain of splitting `s_set` on `attr`.
    ID(S, A) = H(S) - Σ (|S_v| / |S|) * H(S_v)
    where S_v is the subset of S for which attribute A has value v.
    """
    labels = [example[target_attr] for example in s_set]
    H_s = calculate_entropy(labels)
    information_gain = H_s

    categories = set(example[attr] for example in s_set)

    for category in categories:
        s_v = [example for example in s_set if example[attr] == category]
        prob = len(s_v) / len(s_set)
        s_v_labels = [example[target_attr] for example in s_v]
        H_s_v = calculate_entropy(s_v_labels)
        information_gain -= prob * H_s_v
    return information_gain


def get_most_information_gain(
    examples: List[Dict[str, Any]], attributes: List[str], target_attr: str
) -> str:
    max_gain = -1
    best_attr = ""
    for attr in attributes:
        gain = calculate_information_gain(examples, attr, target_attr)
        if gain > max_gain:
            max_gain = gain
            best_attr = attr
    return best_attr


def is_trivial(examples: List[Dict[str, Any]], target_attr) -> Union[Any, None]:
    """
    Check if all examples have the same target attribute value.
    If so, return that value; otherwise, return None.
    """
    first_value = examples[0][target_attr]
    for example in examples:
        if example[target_attr] != first_value:
            return None
    return first_value


def learn_decision_tree(
    examples: List[Dict[str, Any]], attributes: List[str], target_attr: str
) -> Union[Dict[str, Any], Any]:
    """
    Learn a decision tree using the ID3 algorithm.
    Returns either a nested dict representing the tree or a class label at the leaves.
    """
    trivial_class = is_trivial(examples, target_attr)
    if trivial_class is not None:
        return trivial_class

    if not attributes:
        return Counter(example[target_attr] for example in examples).most_common(1)[0][
            0
        ]

    best_attr = get_most_information_gain(examples, attributes, target_attr)
    tree = {best_attr: {}}

    categories = set(example[best_attr] for example in examples)

    for category in categories:
        subset = [ex for ex in examples if ex[best_attr] == category]
        if not subset:
            majority_class = Counter(
                example[target_attr] for example in examples
            ).most_common(1)[0][0]
            tree[best_attr][category] = majority_class
        else:
            remaining_attrs = [attr for attr in attributes if attr != best_attr]
            subtree = learn_decision_tree(subset, remaining_attrs, target_attr)
            tree[best_attr][category] = subtree

    return tree


if __name__ == "__main__":
    print(
        print(
            learn_decision_tree(
                [
                    {"Outlook": "Sunny", "Wind": "Weak", "PlayTennis": "No"},
                    {"Outlook": "Overcast", "Wind": "Strong", "PlayTennis": "Yes"},
                    {"Outlook": "Rain", "Wind": "Weak", "PlayTennis": "Yes"},
                    {"Outlook": "Sunny", "Wind": "Strong", "PlayTennis": "No"},
                    {"Outlook": "Sunny", "Wind": "Weak", "PlayTennis": "Yes"},
                    {"Outlook": "Overcast", "Wind": "Weak", "PlayTennis": "Yes"},
                    {"Outlook": "Rain", "Wind": "Strong", "PlayTennis": "No"},
                    {"Outlook": "Rain", "Wind": "Weak", "PlayTennis": "Yes"},
                ],
                ["Outlook", "Wind"],
                "PlayTennis",
            )
        )
    )
