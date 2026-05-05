#ifndef BUTTON_DRIVER_H_
#define BUTTON_DRIVER_H_

#include <thread>
#include <array>
#include <functional>
#include <optional>

namespace button_driver {
    
using PressCallback = std::function<void()>;
using ReleaseCallback = std::function<void()>;
    
class IButtonDriver {
public:  
    virtual ~IButtonDriver() = default;
    
    virtual void registerPressCallback(PressCallback callback) = 0;
    virtual void deregisterPressCallback() = 0;
    virtual void registerReleaseCallback(ReleaseCallback callback) = 0;
    virtual void deregisterReleaseCallback() = 0;
};

class ButtonDriver : public IButtonDriver {
public:
    
    ButtonDriver();
    ~ButtonDriver();
    
    void registerPressCallback(PressCallback callback) override;
    void deregisterPressCallback() override;

    void registerReleaseCallback(ReleaseCallback callback) override;
    void deregisterReleaseCallback() override;

private: 
    static constexpr int BUTTON_PIN = 14;

    std::thread worker; 
    bool buttonStatus;
    
    std::optional<PressCallback> pressCallback;  
    std::optional<ReleaseCallback> releaseCallback; 
    
    bool running = true; 
};

}

#endif
