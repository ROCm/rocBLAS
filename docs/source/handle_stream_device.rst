
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

The rocBLAS handle stores the following:

- stream
- logging mode
- pointer mode
- atomics mode

Creating the handle will incur a startup cost. There is an additional startup cost for
calling gemm functions. You can call ``rocblas_initialize()`` immediately after calling
``rocblas_create_handle()`` to incur the cost of initializing gemm immediately after the
handle is created. If ``rocblas_initialize()`` is not called, then the gemm startup cost 
will occur with the first gemm call. 

Stream and device management
============================

HIP kernels are launched in a queue, otherwise known as a stream. A stream is a queue of
work for a particular device. A rocBLAS handle always has one and only one stream, and a stream
is always associated with one and only one device. The rocBLAS handle is passed as an argument to all 
rocBLAS functions that launch kernels, and the kernels are launched in that handle's stream, 
to run on that stream's device.

If the user does not create a stream, the rocBLAS handle uses the default stream, maintained 
by the system. Users cannot create or destroy the default stream. However, users can create 
a new stream and bind it to the rocBLAS handle with the two commands:``hipStreamCreate()``
and ``rocblas_set_stream``.  If the user creates a stream, they are responsible for destroying 
it with ``hipStreamDestroy()``.

HIP has device management functions ``hipSetDevice()`` and ``hipGetDevice()``. 
The device that is associated with a stream is whatever device was set at the time that 
``hipStreamCreate()`` is called, or if no device was set, the default device.

Users cannot switch the device in a handle between ``hipStreamCreate()`` and ``hipStreamDestroy()``. 
If users want to use another device, they should create another handle.

rocBLAS never sets a device, it only queries using ``hipGetDevice()``). 
If rocBLAS does not see a valid device, it returns an error message to 
users.

Multiple streams and multiple devices
=====================================

If a machine has num devices, they will have deviceID numbers 0, 1, 2, ... (num - 1). The 
default device has deviceID == 0. Users can run ``num`` rocBLAS handles 
on ``num`` devices concurrently but can not span a single rocBLAS 
handle on ``num`` devices. Each handle is associated with a one and only one device, 
and a new handle should be created if you want to run on a new device.
