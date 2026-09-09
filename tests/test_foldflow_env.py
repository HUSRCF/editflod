from ospedit.foldflow_env import inspect_foldflow_environment


def test_foldflow_environment_report_is_machine_readable():
    report = inspect_foldflow_environment()
    assert "python" in report
    assert set(report["packages"]) == {"numpy", "torch", "foldflow", "openfold", "einops", "dm_tree"}
    expected = (
        not report["missing_packages"]
        and report["python_supported_by_official_foldflow_env"]
        and report["torch_supported_by_official_foldflow_env"]
        and report["numpy_trapz_available"]
    )
    assert report["ready_for_foldflow_import"] is expected


def test_foldflow_environment_reports_configured_source_root(tmp_path, monkeypatch):
    monkeypatch.setenv("OSPEDIT_FOLDFLOW_ROOT", str(tmp_path))
    report = inspect_foldflow_environment()
    assert report["foldflow_root"] == str(tmp_path)
    assert report["foldflow_root_exists"] is True


def test_foldflow_environment_warns_for_missing_source_root(tmp_path, monkeypatch):
    missing = tmp_path / "missing-foldflow"
    monkeypatch.setenv("OSPEDIT_FOLDFLOW_ROOT", str(missing))
    report = inspect_foldflow_environment()
    assert report["foldflow_root_exists"] is False
    assert any("does not exist" in warning for warning in report["compatibility_warnings"])
