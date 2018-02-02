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
 * @brief Logging function
 *
 *  @details
 *  open_logfile    Open file for logging. The parameter
 *                  environment_variable_name contains the name of
 *                  an environment variable set to the full file path
 *                  for the file to open.  If the environment variable
 *                  is not set, the parameter filename contains the
 *                  name of a file that will be opened in the current
 *                  working directory.
 *
 *  @param[in]
 *  environment_variable_name   std::string
 *                              Name of environment variable that
 *                              is set to the full file path.
 *
 *  @param[in]
 *  filename    std::string
 *              Name of file to open in current working directory.
 *
 *  @return[out]
 *  std::ofstream
 *  Open output stream. If function fails, output stream will not.
 *  be open.
 */
inline std::ofstream open_logfile(std::string environment_variable_name, std::string filename)
{
    std::ofstream logfile_ofs;
    std::string logfile_pathname;

    // first try to open filepath in environment_variable_name
    char const* tmp = getenv(environment_variable_name.c_str());
    if(tmp != NULL)
    {
        logfile_pathname = (std::string)tmp;
        logfile_ofs.open(logfile_pathname);
    }

    // second option: open file in current working directory
    if(logfile_ofs.is_open() == false)
    {
        char temp[MAXPATHLEN];
        std::string curr_work_dir =
            (getcwd(temp, MAXPATHLEN) ? std::string(temp) : std::string(""));
        logfile_pathname = curr_work_dir + "/" + filename;
        logfile_ofs.open(logfile_pathname);
    }

    return logfile_ofs;
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
    log_arg(std::ofstream& ofs, std::string& separator) : ofs_(ofs), separator_(separator) {}

    /// Generic overload for () operator.
    template <typename T>
    void operator()(T& x) const
    {
        ofs_ << separator_ << x;
    }

    /// Overload () operator for rocblas_float_complex.
    void operator()(const rocblas_float_complex complex_value) const
    {
        ofs_ << separator_ << complex_value.x << separator_ << complex_value.y;
    }

    /// Overload () operator for rocblas_double_complex.
    void operator()(const rocblas_double_complex complex_value) const
    {
        ofs_ << separator_ << complex_value.x << separator_ << complex_value.y;
    }

    private:
    std::ofstream& ofs_;     ///< Output stream file.
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
void log_arguments(std::ofstream& ofs, std::string& separator, H head, Ts&... xs)
{
    ofs << "\n" << head;
    each_args(log_arg{ofs, separator}, xs...);
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
void log_argument(std::ofstream& ofs, std::string& separator, H head)
{
    ofs << "\n" << head;
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
void log_argument(std::ofstream& ofs, H head)
{
    ofs << "\n" << head;
}

#endif
