from unittest.mock import Mock

from click.testing import CliRunner
from thothcraft_cli.main import main
from thothcraft_cli import daemon


def test_help():
    runner = CliRunner()
    for args in (['--help'], ['models', '--help'], ['models', 'upload', '--help'], ['predictions', '--help']):
        assert runner.invoke(main, args).exit_code == 0


def test_predictions_snapshot(monkeypatch):
    client = Mock()
    client.device.return_value.predictions.return_value = [{'label': 'occupied'}]
    monkeypatch.setattr('thothcraft_cli.main._client', lambda: client)
    result = CliRunner().invoke(main, ['predictions', 'pi', '--minute', '20260922_0000'])
    assert result.exit_code == 0
    assert 'occupied' in result.output
    client.device.return_value.predictions.assert_called_once_with('20260922_0000')


def test_probe_command(monkeypatch):
    monkeypatch.setattr('thothcraft_cli.probe.scan', lambda: {'radar': (False, 'not attached')})
    result = CliRunner().invoke(main, ['sensors', 'list'])
    assert result.exit_code == 0
    assert 'not attached' in result.output


def test_daemon_uuid_and_missing_credential(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(daemon, 'CONFIG_DIR', tmp_path)
    monkeypatch.setattr(daemon, 'DEVICE_FILE', tmp_path / 'device.json')
    assert daemon._device_uuid() == daemon._device_uuid()
    assert daemon.run() == 2
    assert 'No device credential' in capsys.readouterr().err
