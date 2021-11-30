/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */
// This emulates the required functionality of boost::program_options

#pragma once

#include <cinttypes>
#include <cstdio>
#include <iomanip>
#include <map>
#include <ostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace roc
{
    // Regular expression for token delimiters (whitespace and commas)
    static const std::regex program_options_regex{"[, \\f\\n\\r\\t\\v]+",
                                                  std::regex_constants::optimize};

    // Polymorphic base class to use with dynamic_cast
    class value_base
    {
    protected:
        bool m_has_actual  = false;
        bool m_has_default = false;

    public:
        virtual ~value_base() = default;

        bool has_actual() const
        {
            return m_has_actual;
        }

        bool has_default() const
        {
            return m_has_default;
        }
    };

    // Value parameters
    template <typename T>
    class value : public value_base
    {
        T  m_var{}; // Variable to be modified if no pointer provided
        T* m_var_ptr; // Pointer to variable to be modified

    public:
        // Constructor
        explicit value()
            : m_var_ptr(nullptr)
        {
        }

        explicit value(const T& var, bool defaulted)
            : m_var(var)
            , m_var_ptr(nullptr)
        {
            m_has_actual  = !defaulted;
            m_has_default = defaulted;
        }

        explicit value(T* var_ptr)
            : m_var_ptr(var_ptr)
        {
        }

        // Allows actual_value() and default_value()
        value* operator->()
        {
            return this;
        }

        // Get the value
        const T& get_value() const
        {
            if(m_var_ptr)
                return *m_var_ptr;
            else
                return m_var;
        }

        // Set actual value
        value& actual_value(T val)
        {
            if(m_var_ptr)
                *m_var_ptr = std::move(val);
            else
                m_var = std::move(val);
            m_has_actual = true;
            return *this;
        }

        // Set default value
        value& default_value(T val)
        {
            if(!m_has_actual)
            {
                if(m_var_ptr)
                    *m_var_ptr = std::move(val);
                else
                    m_var = std::move(val);
                m_has_default = true;
            }
            return *this;
        }
    };

    // bool_switch is a value<bool>, which is handled specially
    using bool_switch = value<bool>;

    class variable_value
    {
        std::shared_ptr<value_base> m_val;

    public:
        // Constructor
        explicit variable_value() = default;

        template <typename T>
        explicit variable_value(const T& xv, bool xdefaulted)
            : m_val(std::make_shared<value<T>>(xv, xdefaulted))
        {
        }

        explicit variable_value(std::shared_ptr<value_base> val)
            : m_val(val)
        {
        }

        // Member functions
        bool empty() const
        {
            return !m_val.get() || (!m_val->has_actual() && !m_val->has_default());
        }

        bool defaulted() const
        {
            return m_val.get() && !m_val->has_actual() && m_val->has_default();
        }

        template <typename T>
        const T& as() const
        {
            if(value<T>* val = dynamic_cast<value<T>*>(m_val.get()))
                return val->get_value();
            else
                throw std::logic_error("Internal error: Invalid cast");
        }
    };

    using variables_map = std::map<std::string, variable_value>;

    class options_description
    {
        // desc_option describes a particular option
        class desc_option
        {
            std::string                 m_opts;
            std::shared_ptr<value_base> m_val;
            std::string                 m_desc;

        public:
            // Constructor with options, value and description
            template <typename T>
            desc_option(std::string opts, value<T> val, std::string desc)
                : m_opts(std::move(opts))
                , m_val(new auto(std::move(val)))
                , m_desc(std::move(desc))
            {
            }

            // Constructor with options and description
            desc_option(std::string opts, std::string desc)
                : m_opts(std::move(opts))
                , m_val(nullptr)
                , m_desc(std::move(desc))
            {
            }

            // Copy constructor is deleted
            desc_option(const desc_option&) = delete;

            // Move constructor
            desc_option(desc_option&& other) = default;

            // Accessors
            const std::string& get_opts() const
            {
                return m_opts;
            }

            const std::shared_ptr<value_base> get_val() const
            {
                return m_val;
            }

            const std::string& get_desc() const
            {
                return m_desc;
            }

            // Set a value
            void set_val(int& argc, char**& argv, const std::string& inopt) const
            {
                // We test all supported types with dynamic_cast and parse accordingly
                bool match = false;
                if(auto* ptr = dynamic_cast<value<int32_t>*>(m_val.get()))
                {
                    int32_t val;
                    match = argc && sscanf(*argv, "%" SCNd32, &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<uint32_t>*>(m_val.get()))
                {
                    uint32_t val;
                    match = argc && sscanf(*argv, "%" SCNu32, &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<int64_t>*>(m_val.get()))
                {
                    int64_t val;
                    match = argc && sscanf(*argv, "%" SCNd64, &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<uint64_t>*>(m_val.get()))
                {
                    uint64_t val;
                    match = argc && sscanf(*argv, "%" SCNu64, &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<float>*>(m_val.get()))
                {
                    float val;
                    match = argc && sscanf(*argv, "%f", &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<double>*>(m_val.get()))
                {
                    double val;
                    match = argc && sscanf(*argv, "%lf", &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<char>*>(m_val.get()))
                {
                    char val;
                    match = argc && sscanf(*argv, " %c", &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<int8_t>*>(m_val.get()))
                {
                    int8_t val;
                    match = argc && sscanf(*argv, " %c", &val) == 1;
                    ptr->actual_value(val);
                }
                else if(auto* ptr = dynamic_cast<value<bool>*>(m_val.get()))
                {
                    // We handle bool specially, setting the value to true without argument
                    ptr->actual_value(true);
                    return;
                }
                else if(auto* ptr = dynamic_cast<value<std::string>*>(m_val.get()))
                {
                    if(argc)
                    {
                        ptr->actual_value(*argv);
                        match = true;
                    }
                }
                else
                {
                    throw std::logic_error("Internal error: Unsupported data type (setting value)");
                }

                if(!match)
                    throw std::invalid_argument(argc ? "Invalid value for " + inopt
                                                     : "Missing required value for " + inopt);

                // Skip past the argument's value
                ++argv;
                --argc;
            }
        };

        // Description and option list
        std::string              m_desc;
        std::vector<desc_option> m_optlist;

        // desc_optionlist allows chains of options to be parenthesized
        class desc_optionlist
        {
            std::vector<desc_option>& m_list;

        public:
            explicit desc_optionlist(std::vector<desc_option>& list)
                : m_list(list)
            {
            }

            template <typename... Ts>
            desc_optionlist operator()(Ts&&... arg)
            {
                m_list.push_back(desc_option(std::forward<Ts>(arg)...));
                return *this;
            }
        };

        // Parse an option at the current (argc, argv) position
        void parse_option(int& argc, char**& argv, variables_map& vm, bool ignoreUnknown) const
        {
            // Iterate across all options
            for(const auto& opt : m_optlist)
            {
                // Canonical name used for map
                std::string canonical_name;

                // Iterate across tokens in the opts
                for(std::sregex_token_iterator tok{
                        opt.get_opts().begin(), opt.get_opts().end(), program_options_regex, -1};
                    tok != std::sregex_token_iterator();
                    ++tok)
                {
                    // The first option in a list of options is the canonical name
                    if(!canonical_name.length())
                        canonical_name = tok->str();

                    // If the length of the option is 1, it is single-dash; otherwise double-dash
                    const char* prefix = tok->length() == 1 ? "-" : "--";

                    // If option matches
                    if(*argv == prefix + tok->str())
                    {
                        ++argv;
                        --argc;

                        // If option has a value, set it
                        if(opt.get_val().get())
                            opt.set_val(argc, argv, prefix + tok->str());

                        // Add seen options to map
                        vm[canonical_name] = variable_value(opt.get_val());

                        return; // Return successfully
                    }
                }
            }

            // No options were matched
            if(ignoreUnknown)
            {
                ++argv;
                --argc;
            }
            else
                throw std::invalid_argument("Option " + std::string(argv[0]) + " is not defined.");
        }

    public:
        // Constructor
        explicit options_description(std::string desc)
            : m_desc(std::move(desc))
        {
        }

        // Start a desc_optionlist chain
        desc_optionlist add_options() &
        {
            return desc_optionlist(m_optlist);
        }

        // Parse all options
        void parse_options(int&           argc,
                           char**&        argv,
                           variables_map& vm,
                           bool           ignoreUnknown = false) const
        {
            // Add options with default values to map
            for(const auto& opt : m_optlist)
            {
                std::sregex_token_iterator tok{
                    opt.get_opts().begin(), opt.get_opts().end(), program_options_regex, -1};

                // Canonical name used for map
                std::string canonical_name = tok->str();

                if(opt.get_val().get() && opt.get_val()->has_default())
                    vm[canonical_name] = variable_value(opt.get_val());
            }

            // Parse options
            while(argc)
                parse_option(argc, argv, vm, ignoreUnknown);
        }

        // Formatted output of command-line arguments description
        friend std::ostream& operator<<(std::ostream& os, const options_description& d)
        {
            // Iterate across all options
            for(const auto& opt : d.m_optlist)
            {
                bool               first = true, printvalue = true;
                const char*        delim = "";
                std::ostringstream left;

                // Iterate across tokens in the opts
                for(std::sregex_token_iterator tok{opt.get_opts().begin(),
                                                   opt.get_opts().end(),
                                                   program_options_regex,
                                                   -1};
                    tok != std::sregex_token_iterator();
                    ++tok, first = false, delim = " ")
                {
                    // If the length of the option is 1, it is single-dash; otherwise double-dash
                    const char* prefix = tok->length() == 1 ? "-" : "--";
                    left << delim << (first ? "" : "|") << prefix << tok->str();

                    if(tok->str() == "help" || tok->str() == "h")
                        printvalue = false;
                }

                if(printvalue)
                    left << " <value>";
                os << std::setw(26) << std::left << left.str() << " " << opt.get_desc() << " ";
                left.str(std::string());

                // Print the default value of the variable type if it exists
                // We do not print the default value for bool
                const value_base* val = opt.get_val().get();
                if(val && !dynamic_cast<const value<bool>*>(val))
                {
                    if(val->has_default())
                    {
                        // We test all supported types with dynamic_cast and print accordingly
                        left << " (Default value is: ";
                        if(dynamic_cast<const value<int32_t>*>(val))
                            left << dynamic_cast<const value<int32_t>*>(val)->get_value();
                        else if(dynamic_cast<const value<uint32_t>*>(val))
                            left << dynamic_cast<const value<uint32_t>*>(val)->get_value();
                        else if(dynamic_cast<const value<int64_t>*>(val))
                            left << dynamic_cast<const value<int64_t>*>(val)->get_value();
                        else if(dynamic_cast<const value<uint64_t>*>(val))
                            left << dynamic_cast<const value<uint64_t>*>(val)->get_value();
                        else if(dynamic_cast<const value<float>*>(val))
                            left << dynamic_cast<const value<float>*>(val)->get_value();
                        else if(dynamic_cast<const value<double>*>(val))
                            left << dynamic_cast<const value<double>*>(val)->get_value();
                        else if(dynamic_cast<const value<char>*>(val))
                            left << dynamic_cast<const value<char>*>(val)->get_value();
                        else if(dynamic_cast<const value<int8_t>*>(val))
                            left << dynamic_cast<const value<int8_t>*>(val)->get_value();
                        else if(dynamic_cast<const value<std::string>*>(val))
                            left << dynamic_cast<const value<std::string>*>(val)->get_value();
                        else
                            throw std::logic_error(
                                "Internal error: Unsupported data type (printing value)");
                        left << ")";
                    }
                }
                os << left.str() << "\n\n";
            }
            return os << std::flush;
        }
    };

    // Class representing command line parser
    class parse_command_line
    {
        variables_map m_vm;

    public:
        parse_command_line(int                        argc,
                           char**                     argv,
                           const options_description& desc,
                           bool                       ignoreUnknown = false)
        {
            ++argv; // Skip argv[0]
            --argc;
            desc.parse_options(argc, argv, m_vm, ignoreUnknown);
        }

        // Copy the variables_map
        friend void store(const parse_command_line& p, variables_map& vm)
        {
            vm = p.m_vm;
        }

        // Move the variables_map
        friend void store(parse_command_line&& p, variables_map& vm)
        {
            vm = std::move(p.m_vm);
        }
    };

    // We can define the notify() function as a no-op for our purposes
    inline void notify(const variables_map&) {}

}
