import io
import json
import zipfile
from unittest.mock import Mock
from urllib.error import HTTPError

import numpy as np
import pytest

from thothcraft import Client, Dataset, Device, Minute, SensorData
from thothcraft.client import _Http
from thothcraft.models import Deployment
from thothcraft.errors import AuthError, EntitlementError, NotFoundError


def test_credentials_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr('thothcraft.client._CREDENTIALS_PATH', str(tmp_path / 'credentials.json'))
    Client.store_token('test-token', 'https://example.invalid')
    client = Client.login()
    assert client._http.token == 'test-token'
    assert client._http.base_url == 'https://example.invalid'
    Client.clear_token()
    assert Client.load_token() is None


@pytest.mark.parametrize('code,error', [(401, AuthError), (403, EntitlementError), (404, NotFoundError)])
def test_http_errors(monkeypatch, code, error):
    def fail(*args, **kwargs):
        raise HTTPError('https://example.invalid', code, 'denied', {}, io.BytesIO(b'denied'))
    monkeypatch.setattr('urllib.request.urlopen', fail)
    with pytest.raises(error):
        _Http('https://example.invalid').get_json('/test')


def test_devices_and_deploy_payload():
    client = Client()
    client._http = Mock()
    client._http.get_json.return_value = {'devices': [{'device_uuid': 'pi', 'device_name': 'Lab'}]}
    assert client.devices()[0].name == 'Lab'
    client._http.post_json.return_value = {'deployment_id': 'd1'}
    assert client.deploy_model(7, 'pi', {'threshold': .5}).id == 'd1'
    client._http.post_json.assert_called_once_with('/api/datasets/models/7/deploy',
        {'model_id': 7, 'device_id': 'pi', 'config': {'threshold': .5}})


def test_upload_multipart(tmp_path):
    path = tmp_path / 'model.pt'
    path.write_bytes(b'torchscript-test-archive')
    client = Client()
    client._http._request = Mock(return_value=b'{"model":{"id":7,"name":"test"}}')
    model = client.upload_model(path, name='test', classes=['empty', 'occupied'],
        input_spec={'sensor': 'radar', 'representation': 'raw_adc', 'frames': 1, 'shape': [1, 8]})
    assert model.id == 7
    args, kwargs = client._http._request.call_args
    assert args == ('POST', '/api/datasets/models/upload')
    assert b'name="metadata"' in kwargs['body']
    assert b'"schema": "thoth-model/v1"' in kwargs['body']
    assert b'name="model"; filename="model.pt"' in kwargs['body']
    assert b'torchscript-test-archive' in kwargs['body']
    with pytest.raises(ValueError):
        client.upload_model(tmp_path / 'bad.onnx', name='bad', classes=['x'], input_spec={})


@pytest.mark.parametrize('status', ['delivered', 'declined'])
def test_deployment_wait(status):
    client = Mock()
    deployment = Deployment(client, {'deployment_id': 'd1'})
    client.deployments.return_value = [Deployment(client, {'deployment_id': 'd1', 'status': status})]
    assert deployment.wait(timeout=1).status == status


def test_deployment_timeout_and_cancelled():
    deployment = Deployment(Mock(), {'deployment_id': 'd1'})
    with pytest.raises(TimeoutError):
        deployment.wait(timeout=0)
    deployment._client.deployments.return_value = []
    with pytest.raises(NotFoundError):
        deployment.refresh()


def test_predictions_and_stream_cursor(monkeypatch):
    http = Mock()
    http.get_json.side_effect = [
        {'chunks': [{'model_predictions': [{'label': 'occupied'}]}]},
        {'chunks': [{'id': 1}], 'cursor': '2026-09-22T00:00:00Z'},
        {'chunks': [{'id': 2}], 'cursor': '2026-09-22T00:00:01Z'},
    ]
    device = Device(http, {'device_uuid': 'pi'})
    assert device.predictions() == [{'label': 'occupied'}]
    monkeypatch.setattr('thothcraft.devices.time.sleep', lambda _: None)
    assert list(device.stream(max_items=2)) == [{'id': 1}, {'id': 2}]
    assert http.get_json.call_args.args[1] == {'after': '2026-09-22T00:00:00Z'}


def test_numpy_and_xy():
    data = SensorData([[1, 2], [3, 4]])
    assert data.to_numpy().shape == (2, 2)
    minute = Mock(id='one', labels={'activity': 1})
    minute.__getitem__ = Mock(return_value=data)
    x, y = Dataset().add(minute).to_xy(sensor='radar', label='activity')
    np.testing.assert_array_equal(x, [[1, 2, 3, 4]])
    np.testing.assert_array_equal(y, [1])
    with pytest.raises(KeyError):
        Dataset().add(minute).to_xy(label='missing')
    with pytest.raises(ValueError):
        SensorData([[1], [2, 3]]).to_numpy()


def test_minute_zip_repeated_access():
    npz = io.BytesIO()
    np.savez(npz, metadata_json=np.frombuffer(b'{"model_predictions":[]}', dtype=np.uint8))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('capture.npz', npz.getvalue())
    http = Mock()
    http.get_bytes.return_value = archive.getvalue()
    with Minute(http, '20260922_0000', 'pi') as minute:
        assert list(minute) == ['radar', 'csi', 'camera', 'sense']
        assert len(minute) == 4
        assert len(minute['radar']) == len(minute['csi']) == 0
    http.get_bytes.assert_called_once()
