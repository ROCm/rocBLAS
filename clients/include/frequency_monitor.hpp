#ifndef FREQ_MONITOR
#define FREQ_MONITOR

static const char* env = getenv("ROCBLAS_BENCH_FREQ");
#ifndef _WIN32 && !env

#include <rocm_smi/rocm_smi.h>

template <typename T>
inline std::ostream& stream_write(std::ostream& stream, T&& val)
{
    return stream << std::forward<T>(val);
}

template <typename T, typename... Ts>
inline std::ostream& stream_write(std::ostream& stream, T&& val, Ts&&... vals)
{
    return stream_write(stream << std::forward<T>(val), std::forward<Ts>(vals)...);
}

template <typename... Ts>
inline std::string concatenate(Ts&&... vals)
{
    std::ostringstream msg;
    stream_write(msg, std::forward<Ts>(vals)...);

    return msg.str();
}

#define HIP_CHECK_EXC(expr)                                                                       \
    do                                                                                            \
    {                                                                                             \
        hipError_t e = (expr);                                                                    \
        if(e)                                                                                     \
        {                                                                                         \
            const char*        errName = hipGetErrorName(e);                                      \
            const char*        errMsg  = hipGetErrorString(e);                                    \
            std::ostringstream msg;                                                               \
            msg << "Error " << e << "(" << errName << ") " << __FILE__ << ":" << __LINE__ << ": " \
                << std::endl                                                                      \
                << #expr << std::endl                                                             \
                << errMsg << std::endl;                                                           \
            throw std::runtime_error(msg.str());                                                  \
        }                                                                                         \
    } while(0)

#define RSMI_CHECK_EXC(expr)                                                                      \
    do                                                                                            \
    {                                                                                             \
        rsmi_status_t e = (expr);                                                                 \
        if(e)                                                                                     \
        {                                                                                         \
            const char* errName = nullptr;                                                        \
            rsmi_status_string(e, &errName);                                                      \
            std::ostringstream msg;                                                               \
            msg << "Error " << e << "(" << errName << ") " << __FILE__ << ":" << __LINE__ << ": " \
                << std::endl                                                                      \
                << #expr << std::endl;                                                            \
            throw std::runtime_error(msg.str());                                                  \
        }                                                                                         \
    } while(0)

class FrequencyMonitor
{
public:
    static FrequencyMonitor* getInstance()
    {
        if(m_instancePtr == NULL)
        {
            m_instancePtr = new FrequencyMonitor();
        }
        return m_instancePtr;
    }

    // deleting copy constructor
    FrequencyMonitor(const FrequencyMonitor& obj) = delete;

    ~FrequencyMonitor()
    {
        m_stop = true;
        m_exit = true;

        m_cv.notify_all();
        m_thread.join();
    }

    void set_device_id(int deviceId)
    {
        m_smiDeviceIndex = GetROCmSMIIndex(deviceId);
    }

    void start()
    {
        clearValues();
        runBetweenEvents();
    }

    void stop()
    {
        assertActive();
        m_stop = true;
        wait();
    }

    double getAverageFrequency()
    {
        assertNotActive();

        if(m_dataPoints == 0)
            throw std::runtime_error("No data points collected!");

        double averageFrequency = 0.0;
        averageFrequency        = static_cast<double>(m_frequencySum / m_dataPoints);
        return averageFrequency / 1000000;
    }

    double getMedianFrequency()
    {
        assertNotActive();

        if(m_dataPoints == 0)
            throw std::runtime_error("No data points collected!");

        std::sort(m_freqArray.begin(), m_freqArray.end());
        double medianFrequency = static_cast<double>(m_freqArray[(m_dataPoints - 1) / 2]);
        if(m_dataPoints % 2 == 0)
            medianFrequency
                = static_cast<double>(medianFrequency + m_freqArray[(m_dataPoints - 1) / 2 + 1])
                  / 2.0;
        return medianFrequency / 1000000;
    }

private:
    FrequencyMonitor()
    {
        initThread();
    }

    void initThread()
    {
        m_stop   = false;
        m_exit   = false;
        m_thread = std::thread([=]() { this->runLoop(); });
        return;
    }

    void runBetweenEvents()
    {
        assertNotActive();
        {
            std::unique_lock<std::mutex> lock(m_mutex);

            m_task   = std::move(Task([=]() { this->collect(); }));
            m_future = m_task.get_future();

            m_stop = false;
            m_exit = false;
        }
        m_cv.notify_all();
    }

    void runLoop()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        while(!m_exit)
        {

            while(!m_task.valid() && !m_exit)
            {
                m_cv.wait(lock);
            }

            if(m_exit)
            {
                return;
            }

            m_task();
            m_task = std::move(Task());
        }
        return;
    }

    void collect()
    {
        rsmi_frequencies_t freq;

        do
        {
            //XCD 0
            auto status1 = rsmi_dev_gpu_clk_freq_get(m_smiDeviceIndex, RSMI_CLK_TYPE_SYS, &freq);
            if(status1 != RSMI_STATUS_SUCCESS)
            {
                continue;
            }
            else
            {
                m_frequencySum += freq.frequency[freq.current];
                m_freqArray.push_back(freq.frequency[freq.current]);
                m_dataPoints++;
            }
            // collect freq every 50ms
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

        } while(!m_stop && !m_exit);
    }

    void assertActive()
    {
        if(!m_future.valid())
            throw std::runtime_error("Monitor is not active.");
    }

    void assertNotActive()
    {
        if(m_future.valid())
            throw std::runtime_error("Monitor is active.");
    }

    void clearValues()
    {
        m_frequencySum = 0;
        m_dataPoints   = 0;
    }

    void wait()
    {
        if(!m_future.valid())
            return;

        if(!m_stop)
            throw std::runtime_error("Waiting for monitoring to stop with no end condition.");

        m_future.wait();
        m_future = std::move(std::future<void>());
    }

    void InitROCmSMI()
    {
        static rsmi_status_t status = rsmi_init(0);
        RSMI_CHECK_EXC(status);
    }

    uint32_t GetROCmSMIIndex(int hipDeviceIndex)
    {
        InitROCmSMI();

        hipDeviceProp_t props;

        HIP_CHECK_EXC(hipGetDeviceProperties(&props, hipDeviceIndex));
#if HIP_VERSION >= 50220730
        int hip_version;
        HIP_CHECK_EXC(hipRuntimeGetVersion(&hip_version));
        if(hip_version >= 50220730)
        {
            HIP_CHECK_EXC(hipDeviceGetAttribute(&props.multiProcessorCount,
                                                hipDeviceAttributePhysicalMultiProcessorCount,
                                                hipDeviceIndex));
        }
#endif

        uint64_t hipPCIID = 0;
        // hipPCIID |= props.pciDeviceID & 0xFF;
        // hipPCIID |= ((props.pciBusID & 0xFF) << 8);
        // hipPCIID |= (props.pciDomainID) << 16;

        hipPCIID |= (((uint64_t)props.pciDomainID & 0xffffffff) << 32);
        hipPCIID |= ((props.pciBusID & 0xff) << 8);
        hipPCIID |= ((props.pciDeviceID & 0x1f) << 3);

        uint32_t smiCount = 0;

        RSMI_CHECK_EXC(rsmi_num_monitor_devices(&smiCount));

        std::ostringstream msg;
        msg << "PCI IDs: [" << std::endl;

        for(uint32_t smiIndex = 0; smiIndex < smiCount; smiIndex++)
        {
            uint64_t rsmiPCIID = 0;

            RSMI_CHECK_EXC(rsmi_dev_pci_id_get(smiIndex, &rsmiPCIID));

            msg << smiIndex << ": " << rsmiPCIID << std::endl;

            if(hipPCIID == rsmiPCIID)
                return smiIndex;
        }

        msg << "]" << std::endl;

        throw std::runtime_error(concatenate("RSMI Can't find a device with PCI ID ",
                                             hipPCIID,
                                             "(",
                                             props.pciDomainID,
                                             "-",
                                             props.pciBusID,
                                             "-",
                                             props.pciDeviceID,
                                             ")\n",
                                             msg.str()));
    }

    using Task = std::packaged_task<void(void)>;
    Task                     m_task;
    std::atomic<bool>        m_exit;
    std::atomic<bool>        m_stop;
    std::future<void>        m_future;
    std::thread              m_thread;
    std::condition_variable  m_cv;
    std::mutex               m_mutex;
    uint32_t                 m_smiDeviceIndex;
    size_t                   m_dataPoints;
    uint64_t                 m_frequencySum;
    std::vector<uint64_t>    m_freqArray;
    static FrequencyMonitor* m_instancePtr;
};

#else

class FrequencyMonitor
{
public:
    static FrequencyMonitor* getInstance()
    {
        if(m_instancePtr == NULL)
        {
            m_instancePtr = new FrequencyMonitor();
        }
        return m_instancePtr;
    }

    // deleting copy constructor
    FrequencyMonitor(const FrequencyMonitor& obj) = delete;

    ~FrequencyMonitor() {}

    void set_device_id(int deviceId) {}

    void start() {}

    void stop() {}

    double getAverageFrequency() {}

    double getMedianFrequency() {}

private:
    FrequencyMonitor() {}

    static FrequencyMonitor* m_instancePtr;
};

#endif

FrequencyMonitor*                        FrequencyMonitor::m_instancePtr = NULL;
static std::unique_ptr<FrequencyMonitor> freq_monitor(FrequencyMonitor ::getInstance());

#endif
