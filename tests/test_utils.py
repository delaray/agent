from src import utils


def test_timing_returns_value_and_reports_elapsed_time(monkeypatch, capsys):
    ticks = iter([60.0, 150.0])
    monkeypatch.setattr(utils, "time", lambda: next(ticks))

    @utils.timing
    def add(left, right=0):
        return left + right

    assert add(2, right=3) == 5
    assert add.__name__ == "add"
    assert capsys.readouterr().out == "add took 1.5 minutes.\n"
