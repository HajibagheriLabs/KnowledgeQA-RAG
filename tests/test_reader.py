import pytest

from knowledgeqa.ingest import SquadQuestion, split_by_article
from knowledgeqa.reader import Answer, score_answers


def test_answer_metrics_match_hand_computed_example():
    golds = {
        "q1": ["Denver Broncos", "Broncos"],
        "q2": ["Santa Clara, California"],
        "q3": ["1990"],
        "q4": [],
        "q5": [],
    }
    answers = {
        "q1": Answer("the Denver Broncos", 2.0),
        "q2": Answer("Santa Clara", 1.0),
        "q3": Answer("1990", -1.0),
        "q4": Answer("Paris", -0.5),
        "q5": Answer("Rome", 0.5),
    }
    # q1 matches once the article is dropped; q2 has precision 1 and recall 2/3;
    # q3 falls below the threshold, so it abstains and scores zero
    scores = score_answers(answers, golds, threshold=0.0)
    assert scores == pytest.approx(
        {
            "answerable_em": (1 + 0 + 0) / 3,
            "answerable_f1": (1 + 0.8 + 0) / 3,
            "abstention_accuracy": 1 / 2,
            "answerable": 3,
            "unanswerable": 2,
        }
    )
    # a lower threshold answers more: q3 becomes right and q4 stops abstaining
    looser = score_answers(answers, golds, threshold=-2.0)
    assert looser["answerable_em"] == pytest.approx(2 / 3)
    assert looser["abstention_accuracy"] == 0.0


def test_squad_tuning_articles_never_reach_evaluation():
    questions = [
        SquadQuestion(
            f"{title}-{i}", title, f"{title} context", "q", ("a",) if i else ()
        )
        for title in "ABCDEFG"
        for i in range(3)
    ]
    tune, evaluation = split_by_article(questions, tune_articles=2, seed=13)
    tune_titles, eval_titles = {q.title for q in tune}, {q.title for q in evaluation}
    assert len(tune_titles) == 2
    assert not tune_titles & eval_titles
    assert not {q.context for q in tune} & {q.context for q in evaluation}
    assert len(tune) + len(evaluation) == len(questions)
    assert split_by_article(questions, 2, 13) == (tune, evaluation)
