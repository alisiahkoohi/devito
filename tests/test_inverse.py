import numpy as np
import pytest
from conftest import skipif

from devito import clear_cache, Operator
from examples.seismic import demo_model, TimeAxis, RickerSource, Receiver
from examples.seismic.acoustic import AcousticWaveSolver

pytestmark = skipif(['yask', 'ops'])

presets = {
    'constant': {'preset': 'constant-isotropic'},
    'layers': {'preset': 'layers-isotropic', 'ratio': 3},
}


@pytest.mark.parametrize('mkey, shape, kernel, space_order, nbpml', [
    # 1 tests with varying time and space orders
    ('layers', (61, ), 'OT2', 4, 10), ('layers', (61, ), 'OT2', 8, 10),
    ('layers', (61, ), 'OT4', 4, 10), ('layers', (61, ), 'OT4', 8, 10),
    # 2D tests with varying time and space orders
    ('layers', (61, 71), 'OT2', 4, 10), ('layers', (61, 71), 'OT2', 8, 10),
    ('layers', (61, 71), 'OT2', 12, 10), ('layers', (61, 71), 'OT4', 4, 10),
    ('layers', (61, 71), 'OT4', 8, 10), ('layers', (61, 71), 'OT4', 12, 10),
    # 3D tests with varying time and space orders
    ('layers', (61, 71, 81), 'OT2', 4, 10), ('layers', (61, 71, 81), 'OT2', 8, 10),
    ('layers', (61, 71, 81), 'OT2', 12, 10), ('layers', (61, 71, 81), 'OT4', 4, 10),
    ('layers', (61, 71, 81), 'OT4', 8, 10), ('layers', (61, 71, 81), 'OT4', 12, 10),
    # Constant model in 2D and 3D
    ('constant', (61, 71), 'OT2', 8, 14), ('constant', (61, 71, 81), 'OT2', 8, 14),
])
def test_inverse_A(mkey, shape, kernel, space_order, nbpml):
    """
    Inverse test for the forward modeling operator.
    The forward modeling operator F generates a wavefield
    from a source while the inverse of F generates a source wavefield that is
    zero everywhere and the source at the source position.
    """
    clear_cache()
    t0 = 0.0  # Start time
    tn = 500.  # Final time
    nrec = 130  # Number of receivers

     # Create model from preset
    model = demo_model(spacing=[15. for _ in shape], dtype=np.float64,
                       space_order=space_order, shape=shape, nbpml=nbpml,
                       **(presets[mkey]))

     # Derive timestepping from model spacing
    dt = model.critical_dt * (1.73 if kernel == 'OT4' else 1.0)
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

     # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time_range=time_range)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    if len(shape) > 1:
        src.coordinates.data[0, -1] = 30.

     # Define receiver geometry (same as source, but spread across x)
    rec = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    if len(shape) > 1:
        rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

     # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                kernel=kernel, space_order=space_order)

     # Run forward and adjoint operators
    rec, u, _ = solver.forward(save=True)
    q, _, _ = solver.A(u)
    if len(shape) == 1:
        src_inv = q.data[1:, 40]
    elif len(shape) == 2:
        src_inv = q.data[1:, 40, 12]
    else:
        src_inv = q.data[1:, 40, 45 ,12]

    error = np.linalg.norm(src.data[1:, 0] - src_inv)/ np.linalg.norm(src.data[1:, 0])
    # Check that srci == src
    assert np.isclose(error, 0.)