/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef LOGGING_H
#define LOGGING_H
#include <fstream>
#include <string>
#include <unistd.h>
#include <sys/param.h>

/**
 *  @brief Logging function
 *
 *  @details
 *  open_log_stream Open stream log_os for logging.
 *                  If the environment variable with name environment_variable_name
 *                  is not set, then stream log_os to std::cerr.
 *                  Else open a file at the full logfile path contained in
 *                  the environment variable.
 *                  If opening the file suceeds, stream to the file
 *                  else stream to std::cerr.
 *
 *  @param[in]
 *  environment_variable_name   std::string
 *                              Name of environment variable that contains
 *                              the full logfile path.
 *
 *  @parm[out]
 *  log_os      std::ostream**
 *              Output stream. Stream to std:err if environment_variable_name
 *              is not set, else set to stream to log_ofs
 *
 *  @parm[out]
 *  log_ofs     std::ofstream*
 *              Output file stream. If log_ofs->is_open()==true, then log_os
 *              will stream to log_ofs. Else it will stream to std::cerr.
 */

inline void open_log_stream(std::ostream** log_os,
                            std::ofstream* log_ofs,
                            std::string environment_variable_name)
{
    *log_os = &std::cerr;

    char const* environment_variable_value = getenv(environment_variable_name.c_str());

    if(environment_variable_value != NULL)
    {
        // if environment variable is set, open file at logfile_pathname contained in the
        // environment variable
        std::string logfile_pathname = (std::string)environment_variable_value;
        log_ofs->open(logfile_pathname);

        // if log_ofs is open, then stream to log_ofs, else log_os is already
        // set equal to std::cerr
        if(log_ofs->is_open() == true)
        {
            *log_os = log_ofs;
        }
    }
}

/**
 * @brief Invoke functor for each argument in variadic parameter pack.
 * @detail
 * The variatic template function each_args applies the functor f
 * to each argument in the expansion of the parameter pack xs...

 * Note that in ((void)f(xs),0) the C/C++ comma operator evaluates
 * the first expression (void)f(xs) and discards the output, then
 * it evaluates the second expression 0 and returns the output 0.

 * It thus calls (void)f(xs) on each parameter in xs... as a bye-product of
 * building the initializer_list 0,0,0,...0. The initializer_list is discarded.
 *
 * @param f functor to apply to each argument
 *
 * @parm xs variadic parameter pack with list of arguments
 */
template <typename F, typename... Ts>
void each_args(F f, Ts&... xs)
{
    (void)std::initializer_list<int>{((void)f(xs), 0)...};
}

/**
 * @brief Workaround for gcc warnings when each_args called with single argument
 *        and no parameter pack.
 */
template <typename F>
void each_args(F)
{
}

/**
 * @brief Functor for logging arguments
 *
 * @details Functor to log single argument to ofs.
 * The overloaded () in log_arg is the function call operator.
 * The definition in log_arg says "objects of type log_arg can have
 * the function call operator () applied to them with operand x,
 * and it will output x to ofs and return void".
 */
struct log_arg
{
    log_arg(std::ostream& os, std::string& separator) : os_(os), separator_(separator) {}

    /// Generic overload for () operator.
    template <typename T>
    void operator()(T& x) const
    {
        os_ << separator_ << x;
    }

    /// Overload () operator for rocblas_float_complex.
    void operator()(const rocblas_float_complex complex_value) const
    {
        os_ << separator_ << complex_value.x << separator_ << complex_value.y;
    }

    /// Overload () operator for rocblas_double_complex.
    void operator()(const rocblas_double_complex complex_value) const
    {
        os_ << separator_ << complex_value.x << separator_ << complex_value.y;
    }

    private:
    std::ostream& os_;       ///< Output stream.
    std::string& separator_; ///< Separator: output preceding argument.
};

/**
 * @brief Logging function
 *
 * @details
 * log_arguments   Log arguments to output file stream. Arguments
 *                 are preceded by new line, and separated by separator.
 *
 * @param[in]
 * ofs             std::ofstream
 *                 Open output stream file.
 *
 * @param[in]
 * separator       std::string
 *                 Separator to print between arguments.
 *
 * @param[in]
 * head            <typename H>
 *                 First argument to log. It is preceded by newline.
 *
 * @param[in]
 * xs              <typename... Ts>
 *                 Variadic parameter pack. Each argument in variadic
 *                 parameter pack is logged, and it is preceded by
 *                 separator.
 */
template <typename H, typename... Ts>
void log_arguments(std::ostream& os, std::string& separator, H head, Ts&... xs)
{
    os << "\n" << head;
    each_args(log_arg{os, separator}, xs...);
}

/**
 * @brief Logging function
 *
 * @details
 * log_arguments   Log argument to output file stream. Argument
 *                 is preceded by new line.
 *
 * @param[in]
 * ofs             std::ofstream
 *                 open output stream file.
 *
 * @param[in]
 * separator       std::string
 *                 Not used.
 *
 * @param[in]
 * head            <typename H>
 *                 Argument to log. It is preceded by newline.
 */
template <typename H>
void log_argument(std::ostream& os, std::string& separator, H head)
{
    os << "\n" << head;
}

/**
 * @brief Logging function
 *
 * @details
 * log_arguments   Log argument to output file stream. Argument
 *                 is preceded by new line.
 *
 * @param[in]
 * ofs             std::ofstream
 *                 open output stream file.
 *
 * @param[in]
 * head            <typename H>
 *                 Argument to log. It is preceded by newline.
 */
template <typename H>
void log_argument(std::ostream& os, H head)
{
    os << "\n" << head;
}

#endif
