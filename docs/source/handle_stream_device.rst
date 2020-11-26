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

The created handle should be destroyed when the user has completed calling rocBLAS functions. This can be done with:

::

    if(rocblas_destroy_handle(handle) != rocblas_status_success) return EXIT_FAILURE;

The above-created handle will use the default stream and the default device. If the user wants the non-default
stream and the non-default device, then call:

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

If the user created a non-default stream, it is the user's responsibility to destroy it when the user has completed using it with:

::

    if(hipStreamDestroy(stream) != hipSuccess) return EXIT_FAILURE;

When a user changes the stream from one non-default stream to another non-default stream, it is the user's responsibility to synchronize the old stream before setting the new stream. Then, the user can optionally destroy the old stream:

::

    // Synchronize the old stream
    if(hipStreamSynchronize(stream) != hipSuccess) return EXIT_FAILURE;

    // Destroy the old stream (this step is optional but must come after synchronization)
    if(hipStreamDestroy(stream) != hipSuccess) return EXIT_FAILURE;

    // Create a new stream (this step can be done before the steps above)
    if(hipStreamCreate(&stream) != hipSuccess) return EXIT_FAILURE;

    // Set the handle to use the new stream (must come after synchronization)
    if(rocblas_set_stream(handle, stream) != rocblas_status_success) return EXIT_FAILURE;

The above ``hipStreamSynchronize`` is necessary because the rocBLAS handle contains allocated device
memory which must not be shared by multiple asynchronous streams at the same time.

If either the old or new stream is the default (NULL) stream, it is not necessary to
synchronize the old stream before destroying it, or before setting the new stream,
because the synchronization is implicit.

Creating the handle will incur a startup cost. There is an additional startup cost for
gemm functions to load gemm kernels for a specific device. Users can shift the
gemm startup cost to occur after setting the device by calling ``rocblas_initialize()``
after calling ``hipSetDevice()``. This action needs to be done once for each device.
If the user has two rocBLAS handles which use the same device, then the  user only needs to call ``rocblas_initialize()``
once. If ``rocblas_initialize()`` is not called, then the first gemm call will have
the startup cost.

The rocBLAS handle stores the following:

- Stream
- Logging mode
- Pointer mode
- Atomics mode

Stream and Device Management
============================

HIP kernels are launched in a queue. This queue is otherwise known as a stream. A stream is a queue of
work on a particular device.

A rocBLAS handle always has one stream, and a stream is always associated with one device. Furthermore, the rocBLAS handle is passed as an argument to all rocBLAS functions that launch kernels, and these kernels are
launched in that handle's stream to run on that stream's device.

If the user does not create a stream, then the rocBLAS handle uses the default (NULL)
stream, maintained by the system. Users cannot create or destroy the default
stream. However, users can create a new non-default stream and bind it to the rocBLAS handle with the
two commands: ``hipStreamCreate()`` and ``rocblas_set_stream()``.

If the user creates a
stream, they are responsible for destroying it with ``hipStreamDestroy()``. If the handle
is switching from one non-default stream to another, then the old stream needs to be synchronized. Next, the user needs to create and set the new non-default stream using ``hipStreamCreate()`` and ``rocblas_set_stream()`` respectively. Then the user can optionally destroy the old stream. The order of calls would be:
First, call ``hipStreamSynchronize()`` on the old stream. Then, in any order, call the ``hipStreamDestroy()`` on the old stream,  and the new stream being passed to ``rocblas_set_stream()``.

HIP has two important device management functions ``hipSetDevice()`` and ``hipGetDevice()``.

1. ``hipSetDevice()``: Set default device to be used for subsequent hip API calls from this thread.

2. ``hipGetDevice()``: Return the default device id for the calling host thread.

The device which was set using ``hipSetDevice()`` at the time of calling
``hipStreamCreate()`` is the one that is associated with a stream. But, if the device was not set using ``hipSetDevice()``, then, the default device will be used.

Users cannot switch the device in a stream between ``hipStreamCreate()`` and ``hipStreamDestroy()``.
If users want to use another device, they should create another stream.

rocBLAS never sets a device, it only queries using ``hipGetDevice()``. If rocBLAS does not see a
valid device, it returns an error message to users.

Multiple streams and Multiple devices
=====================================

If a machine has ``num`` GPU devices, they will have deviceID numbers 0, 1, 2, ... (``num`` - 1). The
default device has deviceID == 0. Users can run ``num`` rocBLAS handles
on ``num`` devices concurrently. But, users cannot run a single rocBLAS
handle on ``num`` devices. Each handle is associated with only one device.
