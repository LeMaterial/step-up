"""Package-level import smoke test."""


def test_package_imports() -> None:
    import step_up

    assert step_up.__version__
