import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_exp0009r import measurement_order, prefix_answers, scorable_baselines


class FakeTokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens
        fragments = {
            1: "Work",
            2: "continues.",
            3: "The answer is",
            4: "34.",
        }
        return " ".join(fragments[token] for token in tokens)


def test_prefix_answers_respects_cap_and_existing_parser():
    scores = prefix_answers([1, 2, 3, 4], FakeTokenizer(), (2, 4, 8))

    assert scores["2"]["predicted"] is None
    assert scores["2"]["generated_tokens"] == 2
    assert scores["4"]["predicted"] == "34"
    assert scores["8"]["generated_tokens"] == 4


def test_gate_requires_all_three_baselines_to_parse():
    assert scorable_baselines([{"predicted": "1"}] * 3)
    assert not scorable_baselines([
        {"predicted": "1"}, {"predicted": None}, {"predicted": "3"}
    ])


def test_timing_order_balances_modes_across_questions():
    orders = [
        measurement_order(question_index, repeat)
        for question_index in (8, 9, 10)
        for repeat in (1, 2)
    ]

    assert orders[0] == ("full", "mlp_zero", "mlp_predicted")
    assert orders[1] == ("mlp_zero", "mlp_predicted", "full")
    assert all(len(set(order)) == 3 for order in orders)
    for position in range(3):
        assert sorted(order[position] for order in orders) == [
            "full", "full", "mlp_predicted", "mlp_predicted", "mlp_zero", "mlp_zero"
        ]
