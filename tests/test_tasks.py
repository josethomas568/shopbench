from shopbench.tasks.build import compile_task
from shopbench.tasks.specs import TASKS


def test_task_file_matches_specs(tasks, catalog):
    assert len(tasks) == len(TASKS) >= 50
    for spec in TASKS:
        assert compile_task(spec, catalog) == tasks[spec["id"]], f"{spec['id']} is stale; rebuild tasks.json"


def test_task_mix(tasks):
    types = {t["type"] for t in tasks.values()}
    assert {"standard", "trap_oos", "trap_price", "ambiguous", "infeasible", "injection", "cart_only", "multi"} <= types
    traps = [t for t in tasks.values() if t["type"] not in ("standard", "named", "multi")]
    assert len(traps) >= 20


def test_purchase_tasks_have_answers(tasks):
    for t in tasks.values():
        if t["expect"] in ("purchase", "cart_only"):
            assert all(it["acceptable"] for it in t["items"]), t["id"]
