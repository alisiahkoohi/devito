from devito import Function, TimeFunction, memoized_meth
from examples.seismic import PointSource, Receiver
from examples.seismic.acoustic.operators import (
    ForwardOperator, AdjointOperator, GradientOperator, BornOperator, A
)
from examples.checkpointing.checkpoint import DevitoCheckpoint, CheckpointOperator
from pyrevolve import Revolver


class AcousticWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    :param model: Physical model with domain parameters
    :param source: Sparse point symbol providing the injected wave
    :param receiver: Sparse point symbol describing an array of receivers
    :param time_order: Order of the time-stepping scheme (default: 2, choices: 2,4)
                       time_order=4 will not implement a 4th order FD discretization
                       of the time-derivative as it is unstable. It implements instead
                       a 4th order accurate wave-equation with only second order
                       time derivative. Full derivation and explanation of the 4th order
                       in time can be found at:
                       http://www.hl107.math.msstate.edu/pdfs/rein/HighANM_final.pdf
    :param space_order: Order of the spatial stencil discretisation (default: 4)

    Note: space_order must always be greater than time_order
    """
    def __init__(self, model, geometry, kernel='OT2', space_order=2, **kwargs):
        self.model = model
        self.geometry = geometry

        assert self.model == geometry.model

        self.space_order = space_order
        self.kernel = kernel

        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        self.dt = self.model.critical_dt
        if self.kernel == 'OT4':
            self.dt *= 1.73

        # Cache compiler options
        self._kwargs = kwargs

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_adj(self, save=None):
        """Cached operator for adjoint runs"""

        return AdjointOperator(self.model, save=None, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_grad(self, save=True):
        """Cached operator for gradient runs"""
        return GradientOperator(self.model, save=save, geometry=self.geometry,
                                kernel=self.kernel, space_order=self.space_order,
                                **self._kwargs)

    @memoized_meth
    def op_born(self, save=None):
        """Cached operator for born runs"""

        return BornOperator(self.model, save=save, geometry=self.geometry,
                            kernel=self.kernel, space_order=self.space_order,
                            **self._kwargs)

    @memoized_meth
    def op_A(self):
        """Cached operator for applying the wave equation to a wavefield"""
        return A(self.model, source=self.source, kernel=self.kernel,
                 space_order=self.space_order, **self._kwargs)

    def forward(self, src=None, rec=None, u=None, m=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        :param src: Symbol with time series data for the injected source term
        :param rec: Symbol to store interpolated receiver data
        :param u: (Optional) Symbol to store the computed wavefield
        :param m: (Optional) Symbol for the time-constant square slowness
        :param save: Option to store the entire (unrolled) wavefield

        :returns: Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or Receiver(name='rec', grid=self.model.grid,
                              time_range=self.geometry.time_axis,
                              coordinates=self.geometry.rec_positions)

        # Create the forward wavefield if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              save=self.geometry.nt if save else None,
                              time_order=2, space_order=self.space_order)

        # Pick m from model unless explicitly provided
        m = m or self.model.m

        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec=rec, u=u, m=m,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, u, summary

    def adjoint(self, rec, srca=None, v=None, m=None, save=None, **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        :param rec: Symbol with stored receiver data. Please note that
                    these act as the source term in the adjoint run.
        :param srca: Symbol to store the resulting data for the
                     interpolated at the original source location.
        :param v: (Optional) Symbol to store the computed wavefield
        :param m: (Optional) Symbol for the time-constant square slowness

        :returns: Adjoint source, wavefield and performance summary
        """
        # Create a new adjoint source and receiver symbol
        srca = srca or PointSource(name='srca', grid=self.model.grid,
                                   time_range=self.geometry.time_axis,
                                   coordinates=self.geometry.src_positions)

        # Create the adjoint wavefield if not provided

        v = v or TimeFunction(name='v', grid=self.model.grid,
                              save=self.source.nt if save else None,          
                              time_order=2, space_order=self.space_order)
        # Pick m from model unless explicitly provided
        m = m or self.model.m

        # Execute operator and return wavefield and receiver data
        summary = self.op_adj(save).apply(srca=srca, rec=rec, v=v, m=m,
                                      dt=kwargs.pop('dt', self.dt), **kwargs)
        return srca, v, summary

    def gradient(self, rec, u, v=None, grad=None, m=None, isic=False, checkpointing=False, **kwargs):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modelling function, ie. the action of the
        Jacobian adjoint on an input data.

        :param recin: Receiver data as a numpy array
        :param u: Symbol for full wavefield `u` (created with save=True)
        :param v: (Optional) Symbol to store the computed wavefield
        :param grad: (Optional) Symbol to store the gradient field

        :returns: Gradient field and performance summary
        """
        dt = kwargs.pop('dt', self.dt)
        # Gradient symbol
        grad = grad or Function(name='grad', grid=self.model.grid)

        # Create the forward wavefield
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        # Pick m from model unless explicitly provided
        m = m or self.model.m

        if checkpointing:
            u = TimeFunction(name='u', grid=self.model.grid,
                             time_order=2, space_order=self.space_order)
            cp = DevitoCheckpoint([u])
            n_checkpoints = None
            wrap_fw = CheckpointOperator(self.op_fwd(save=False), src=self.geometry.src,
                                         u=u, m=m, dt=dt)
            wrap_rev = CheckpointOperator(self.op_grad(save=False), u=u, v=v,
                                          m=m, rec=rec, dt=dt, grad=grad)

            # Run forward
            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, rec.data.shape[0]-2)
            wrp.apply_forward()
            summary = wrp.apply_reverse()
        else:
            summary = self.op_grad().apply(rec=rec, grad=grad, v=v, u=u, m=m,
                                           dt=dt, **kwargs)
        return grad, summary

    def born(self, dmin, src=None, rec=None, u=None, U=None, m=None, save=False, **kwargs):
        """
        Linearized Born modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        :param src: Symbol with time series data for the injected source term
        :param rec: Symbol to store interpolated receiver data
        :param u: (Optional) Symbol to store the computed wavefield
        :param U: (Optional) Symbol to store the computed wavefield
        :param m: (Optional) Symbol for the time-constant square slowness
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or Receiver(name='rec', grid=self.model.grid,
                              time_range=self.geometry.time_axis,
                              coordinates=self.geometry.rec_positions)

        # Create the forward wavefields u and U if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              save= src.nt if save else None,
                              time_order=2, space_order=self.space_order)
        U = U or TimeFunction(name='U', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)
        # Pick m from model unless explicitly provided
        m = m or self.model.m

        # Execute operator and return wavefield and receiver data
        summary = self.op_born(save).apply(dm=dmin, u=u, U=U, src=src, rec=rec,
                                           m=m, dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, u, U, summary


    def A(self, u, q=None, srci=None, m=None, **kwargs):
        """
        Wave equation application function that creates the necessary
        data objects for running a forward modelling operator.
         :param src: Symbol with time series data for the injected source term
        :param u: Symbol to store the computed wavefield
        :param q: (Optional) Symbol to store the Au object
        :param m: (Optional) Symbol for the time-constant square slowness
         :returns: Source (Au) as a wavefield, src at the source position
        and performance summary
        """
        # Source term is read-only, so re-use the default
        srci = srci or PointSource(name='srci', grid=self.model.grid,
                                   time_range=self.source.time_range,
                                   coordinates=self.source.coordinates.data)


         # Create the forward wavefield if not provided
        q = q or TimeFunction(name='q', grid=self.model.grid,
                              save=self.source.nt,
                              time_order=2, space_order=0)
        # Pick m from model unless explicitly provided
        m = m or self.model.m

         # Execute operator and return wavefield and receiver data
        summary = self.op_A().apply(srci=srci, u=u, q=q, m=m,
                                    dt=kwargs.pop('dt', self.dt), **kwargs)
        return q, srci, summary
