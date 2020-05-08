
.. toctree::
   :maxdepth: 4
   :caption: Contents:

============================
Device and Stream Management
============================

HIP Device management
=====================

hipSetDevice() & hipGetDevice() are HIP device management APIs. They are
NOT part of the rocBLAS API.

Before a HIP kernel invocation, users need to call hipSetDevice() to set
a device, e.g. device 1. If users do not explicitly call it, the system
by default sets it as device 0. Unless users explicitly call
hipSetDevice() to set to another device, their HIP kernels are always
launched on device 0.

The above is a HIP (and CUDA) device management approach and has nothing
to do with rocBLAS. rocBLAS honors the approach above and assumes users
have already set the device before a rocBLAS routine call.

Once users set the device, they create a handle with
``rocblas_status rocblas_create_handle(rocblas_handle *handle)``

Subsequent rocBLAS routines take this handle as an input parameter.
rocBLAS ONLY queries (by hipGetDevice) the user's device; rocBLAS
does NOT set the device for users. If rocBLAS does not see a valid
device, it returns an error message to users. It is the users'
responsibility to provide a valid device to rocBLAS and ensure the
device safety.

Users CANNOT switch devices between rocblas_create_handle() and
rocblas_destroy_handle() (the same as cuBLAS requires). If users want to
change device, they must destroy the current handle, and create another
rocBLAS handle (context).

Stream management
=================

HIP kernels are always launched in a queue (otherwise known as a stream).

If users do not explicitly specify a stream, the system provides a
default stream, maintained by the system. Users cannot create or destroy
the default stream. Howevers, users can freely create new streams (with
hipStreamCreate) and bind it to the rocBLAS handle:
``rocblas_set_stream(rocblas_handle handle, hipStream_t stream_id)`` HIP
kernels are invoked in rocBLAS routines. The rocBLAS handles are always
associated with a stream, and rocBLAS passes its stream to the kernels
inside the routine. One rocBLAS routine only takes one stream in a
single invocation. If users create a stream, they are responsible for
destroying it.

Multiple streams and multiple devices
=====================================

If the system under test has 4 HIP devices, users can run 4 rocBLAS
handles (also known as contexts) on 4 devices concurrently, but can NOT
span a single rocBLAS handle on 4 discrete devices. Each handle is
associated with a particular singular device, and a new handle should be
created for each additional device.
