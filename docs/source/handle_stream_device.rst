
.. toctree::
   :maxdepth: 4
   :caption: Contents:

=====================================
Handle, Stream, and Device Management
=====================================

Handle
======

A rocBLAS handle must be created before calling other rocBLAS functions.
This can be done with:

::

    rocblas_handle handle;
    if(rocblas_create_handle(&handle) != rocblas_status_success) return EXIT_FAILURE;

The handle should be destroyed when the user has completed calling rocBLAS functions with:

::

    if(rocblas_destroy_handle(handle) != rocblas_status_success) return EXIT_FAILURE;

The above will use the default stream and the default device. If you want a non-default
stream and device, then call:

::

    int deviceId = non_default_device_id;
    if(hipSetDevice(deviceId) != hipSuccess) return EXIT_FAILURE;

    //optional call to rocblas_initialize
    rocblas_initialize();

    // note the order, call hipSetDevice before hipStreamCreate
    hipStream_t stream;
    if(hipStreamCreate(&stream) != hipSuccess) return EXIT_FAILURE;

    rocblas_handle handle;
    if(rocblas_create_handle(&handle) != rocblas_status_success) return EXIT_FAILURE;

    if(rocblas_set_stream(handle, stream) != rocblas_status_success) return EXIT_FAILURE;

If you created a stream it is your responsibility to destroy it when
you have completed using it with:

::

    if(hipStreamDestroy(stream) != hipSuccess) return EXIT_FAILURE;

If you change the stream from one non-default stream to another non-default stream, it
is your responsibility to synchronize the old stream before setting the new stream, and
before optionally destroying the old stream:

::

    // Synchronize the old stream
    if(hipStreamSynchronize(stream) != hipSuccess) return EXIT_FAILURE;

    // Destroy the old stream (this step is optional but must come after synchronization)
    if(hipStreamDestroy(stream) != hipSuccess) return EXIT_FAILURE;

    // Create a new stream (this step can be done before the steps above)
    if(hipStreamCreate(&stream) != hipSuccess) return EXIT_FAILURE;

    // Set the handle to use the new stream (must come after synchronization)
    if(rocblas_set_stream(handle, stream) != rocblas_status_success) return EXIT_FAILURE;

This synchronization is necessary because the rocBLAS handle contains allocated device
memory which must not be shared by multiple asynchronous streams at the same time.

If either the old or new stream is the default (NULL) stream, it is not necessary to
synchronize the old stream before destroying it, or before setting the new stream,
because the synchronization is implicit.

Creating the handle will incur a startup cost. There is an additional startup cost for
gemm functions. This is to load gemm kernels for a specific device. You can shift the
gemm startup cost to occur after setting the device by calling ``rocblas_initialize()``
after calling ``hipSetDevice()``. This needs to be done once for each device.
If you have two rocBLAS handles using the same device, you only need to call ``rocblas_initialize()``
once. If ``rocblas_initialize()`` is not called, then the first gemm call will have
the startup cost.

The rocBLAS handle stores the following:

- stream
- logging mode
- pointer mode
- atomics mode

Stream and device management
============================

HIP kernels are launched in a queue, otherwise known as a stream. A stream is a queue of
work for a particular device. A rocBLAS handle always has one and only one stream, and a
stream is always associated with one and only one device. The rocBLAS handle is passed
as an argument to all rocBLAS functions that launch kernels, and the kernels are
launched in that handle's stream, to run on that stream's device.

If the user does not create a stream, then the rocBLAS handle uses the default (NULL)
stream, maintained by the system. Users cannot create or destroy the default
stream. However, users can create a new stream and bind it to the rocBLAS handle with
the two commands: ``hipStreamCreate()`` and ``rocblas_set_stream()``. If the user creates
a stream, they are responsible for destroying it with ``hipStreamDestroy()``. If the
handle is switching from one non-default stream to another, then the old stream should
be synchronized before calling `rocblas_set_stream()` with the new stream, and before
optionally destroying the old stream. The order of calls would be:
``hipStreamSynchronize()`` on the old stream first, and then optionally
``hipStreamDestroy()`` on the old stream, and then for the new stream to be passed to
``rocblas_set_stream()``.

HIP has device management functions ``hipSetDevice()`` and ``hipGetDevice()``.
The device that is associated with a stream is whatever device was set at the time that
``hipStreamCreate()`` is called, or if no device was set, the default device.

Users cannot switch the device in a stream between ``hipStreamCreate()`` and ``hipStreamDestroy()``.
If users want to use another device, they should create another stream.

rocBLAS never sets a device, it only queries using ``hipGetDevice()``). If rocBLAS does not see a
valid device, it returns an error message to users.

Multiple streams and multiple devices
=====================================

If a machine has num devices, they will have deviceID numbers 0, 1, 2, ... (num - 1). The
default device has deviceID == 0. Users can run ``num`` rocBLAS handles
on ``num`` devices concurrently but can not span a single rocBLAS
handle on ``num`` devices. Each handle is associated with one and only one device.
