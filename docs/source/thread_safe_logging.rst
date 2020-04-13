=====================
Thread Safe Logging
=====================

rocBLAS has thread safe logging. This prevents garbled output when multiple threads are writing to the same file.

Thread safe logging is obtained from using rocblas_ostream, a class which can be used similarly to std::ostream. It provides standardized methods for formatted output to either strings or files. The default constructor of rocblas_ostream writes to strings, which are thread-safe because they are owned by the calling thread. There are also rocblas_ostream constructors for writing to files. The rocblas_ostream::yaml_on and rocblas_ostream::yaml_off IO modifiers turn YAML formatting mode on and off.

rocblas_cout and rocblas_cerr are the thread-safe versions of std::cout and std::cerr.

Many output identifiers have been marked "poisoned" in rocblas-test and rocblas-bench, to catch the use of non-thread-safe IO. These include std::cout, std::cerr, printf, fprintf, fputs, puts, and others. The poisoning is not turned on in the library itself or in the samples, because we cannot impose restrictions on the use of these symbols on outside users.

rocblas_handle contains 3 rocblas_ostream pointers for logging output

- static rocblas_ostream* log_trace_os
- static rocblas_ostream* log_bench_os
- static rocblas_ostream* log_profile_os

The user can also create rocblas_ostream pointers/objects outside of the handle.

Each rocblas_ostream associated with a file points to a single rocblas_ostream::worker with a std::shared_ptr, for writing to the file. The worker is mapped from the device id and inode corresponding to the file. More than one rocblas_ostream can point to the same worker.

This means that if more than one rocblas_ostream is writing to a single output file, they will share the same rocblas_ostream::worker.

The << operator for rocblas_ostream is overloaded. Output is first accumulated in rocblas_ostream::os, a std::ostringstream buffer. Each rocblas_ostream has its own os std::ostringstream buffer, so strings in os will not be garbled.

When rocblas_ostream.os is flushed with either a std::endl or an explicit flush of rocblas_ostream, then rocblas_ostream::worker::send pushes the string contents of rocblas_ostream.os and a promise, the pair being called a task,  onto rocblas_ostream.worker.queue.

The send function uses promise/future to asynchronously transfer data from rocblas_ostream.os to rocblas_ostream.worker.queue, and to wait for the worker to finish writing the string to the file. It also locks a mutex to make sure the push of the task onto the queue is atomic.

The ostream.worker.queue will contain a number of tasks. When rocblas_ostream is destroyed, all the tasks.string in rocblas_ostream.worker.queue are printed to the rocblas_ostream file, the std::shared_ptr to the ostream.worker is destroyed, and if the reference count to the worker becomes 0, the worker's thread is sent a 0-length string to tell it to exit.

